"""
Main agent session controller for factor mining.

Runs a budget-limited loop:
1. Agent proposes hypothesis + formula
2. System parses DSL, computes factor values, runs IS walk-forward
3. Gates checked; results fed back to agent
4. After all experiments: top 3 by IS IC get OOS check
5. Session summary written to output file

Usage:
    python -m src.agent.loop --market cn --budget 50 --output reports/session.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dsl.parser import parse as parse_dsl, DSLParseError
from src.dsl.compute import compute_factor
from src.backtest.walk_forward import walk_forward_backtest, run_oos_check, BacktestResult
from src.backtest.gates import check_gates, format_gate_result, GateResult
from src.agent.prompts import (
    build_system_prompt,
    build_feedback_prompt,
    parse_agent_response,
    ParsedResponse,
)
from src.agent.backends import call_agent
from src.paths import FACTOR_LAB_DB, QUANT_CN_DB, QUANT_US_DB

logger = logging.getLogger(__name__)
EXPERIMENTS_FILE = Path(__file__).resolve().parents[2] / "experiments.jsonl"

# ---------------------------------------------------------------------------
# Market configurations (consistent with evaluate/run_evaluation.py)
# ---------------------------------------------------------------------------

MARKET_CONFIGS = {
    "cn": {
        "market": "cn",
        "db_path": str(QUANT_CN_DB),
        "table": "prices",
        "sym_col": "ts_code",
        "date_col": "trade_date",
        "close_col": "close",
        "vol_col": "vol",
        "cost_per_trade": 0.003,
        "oos_start": "2025-10-01",
        "universe_top_n": 2000,
    },
    "us": {
        "market": "us",
        "db_path": str(QUANT_US_DB),
        "table": "prices_daily",
        "sym_col": "symbol",
        "date_col": "date",
        "close_col": "adj_close",
        "vol_col": "volume",
        "cost_per_trade": 0.001,
        "oos_start": "2025-10-01",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_prices(cfg: dict) -> pd.DataFrame:
    """Load raw prices from the pipeline DuckDB with enrichment."""
    import duckdb, shutil, tempfile

    db_path = cfg["db_path"]
    try:
        con = duckdb.connect(db_path, read_only=True)
    except Exception:
        tmp = tempfile.mktemp(suffix=".duckdb")
        shutil.copy2(db_path, tmp)
        con = duckdb.connect(tmp, read_only=True)

    sym = cfg["sym_col"]
    dt = cfg["date_col"]
    market = cfg.get("market", "")

    if market == "cn":
        sql = f"""
            SELECT p.{sym}, p.{dt},
                   p.{cfg['close_col']} AS close, p.open, p.high, p.low,
                   p.{cfg['vol_col']} AS volume, p.amount,
                   db.turnover_rate, db.pe_ttm, db.pb, db.ps_ttm,
                   db.total_mv AS market_cap, db.circ_mv AS circ_market_cap
            FROM {cfg['table']} p
            LEFT JOIN daily_basic db
                ON p.{sym} = db.ts_code AND p.{dt} = db.trade_date
            WHERE p.{cfg['close_col']} > 0
            ORDER BY p.{sym}, p.{dt}
        """
    else:
        sql = f"""
            SELECT {sym}, {dt},
                   {cfg['close_col']} AS close,
                   open, high, low,
                   {cfg['vol_col']} AS volume
            FROM {cfg['table']}
            WHERE {cfg['close_col']} > 0
            ORDER BY {sym}, {dt}
        """
    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()

    # Universe filter for CN
    top_n = cfg.get("universe_top_n")
    if top_n and "market_cap" in df.columns:
        df["_r"] = df.groupby(dt)["market_cap"].rank(ascending=False, method="first", na_option="bottom")
        df = df[df["_r"] <= top_n].drop(columns=["_r"]).reset_index(drop=True)

    return df


def _compute_forward_returns(cfg: dict) -> pd.DataFrame:
    """Compute 5-day forward returns."""
    import duckdb

    con = duckdb.connect(cfg["db_path"], read_only=True)
    sql = f"""
        SELECT {cfg['sym_col']}, {cfg['date_col']},
               LEAD({cfg['close_col']}, 5) OVER w / {cfg['close_col']} - 1 AS fwd_5d
        FROM {cfg['table']}
        WHERE {cfg['close_col']} > 0
        WINDOW w AS (PARTITION BY {cfg['sym_col']} ORDER BY {cfg['date_col']})
        ORDER BY {cfg['sym_col']}, {cfg['date_col']}
    """
    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()
    return df


def _compute_basic_features(prices: pd.DataFrame, sym_col: str, date_col: str) -> pd.DataFrame:
    """Compute basic features (returns, volume ratio, etc.) for DSL evaluation.

    This produces a DataFrame with columns:
        [sym_col, date_col, close, open, high, low, volume,
         ret_1d, ret_5d, ret_20d, volume_ratio, rsi_14, bb_position, ma20_dist]

    These are the features available to the DSL expressions.
    """
    results = []

    for sym, group in prices.groupby(sym_col):
        g = group.sort_values(date_col).reset_index(drop=True)
        n = len(g)
        if n < 60:
            continue

        closes = g["close"].values.astype(float)
        volumes = g["volume"].values.astype(float)
        dates = g[date_col].values

        # Returns
        ret_1d = np.full(n, np.nan)
        ret_5d = np.full(n, np.nan)
        ret_20d = np.full(n, np.nan)
        for i in range(1, n):
            if closes[i - 1] > 0:
                ret_1d[i] = closes[i] / closes[i - 1] - 1
        for i in range(5, n):
            if closes[i - 5] > 0:
                ret_5d[i] = closes[i] / closes[i - 5] - 1
        for i in range(20, n):
            if closes[i - 20] > 0:
                ret_20d[i] = closes[i] / closes[i - 20] - 1

        # Volume ratio
        vol_ratio = np.full(n, np.nan)
        for i in range(20, n):
            avg_vol = np.mean(volumes[i - 20:i])
            if avg_vol > 0:
                vol_ratio[i] = volumes[i] / avg_vol

        # CN-specific columns to pass through (from daily_basic JOIN)
        extra_cols = [c for c in g.columns if c in (
            "amount", "turnover_rate", "pe_ttm", "pb", "ps_ttm",
            "market_cap", "circ_market_cap",
        )]

        for i in range(20, n):
            row = {
                sym_col: sym,
                date_col: dates[i],
                "close": closes[i],
                "open": g["open"].values[i] if "open" in g.columns else closes[i],
                "high": g["high"].values[i] if "high" in g.columns else closes[i],
                "low": g["low"].values[i] if "low" in g.columns else closes[i],
                "volume": volumes[i],
                "ret_1d": ret_1d[i],
                "ret_5d": ret_5d[i],
                "ret_20d": ret_20d[i],
                "volume_ratio": vol_ratio[i],
            }
            # Pass through enrichment columns (amount, turnover_rate, etc.)
            for col in extra_cols:
                row[col] = g[col].values[i]
            results.append(row)

    return pd.DataFrame(results)


def _append_experiment_rows(
    session_id: str,
    market: str,
    experiments: list[dict],
    top3_oos: list[dict],
) -> None:
    if not experiments:
        return

    if EXPERIMENTS_FILE.exists():
        existing_text = EXPERIMENTS_FILE.read_text(encoding="utf-8", errors="replace")
        if f'"session_id": "{session_id}"' in existing_text:
            return

    oos_by_formula = {
        exp["formula"]: "PASS" if exp.get("oos_pass") else "FAIL"
        for exp in top3_oos
    }
    rows = []
    ts = datetime.now().astimezone().isoformat()
    for exp in experiments:
        row = {
            "ts": ts,
            "session_id": session_id,
            "market": market,
            "formula": exp.get("formula", ""),
            "name": exp.get("name", ""),
            "is_ic": round(float(exp.get("is_ic", 0.0)), 4),
            "is_ic_ir": round(float(exp.get("is_ic_ir", 0.0)), 3),
            "is_sharpe": round(float(exp.get("is_sharpe", 0.0)), 3),
            "is_turnover": round(float(exp.get("is_turnover", 0.0)), 3),
            "is_monotonicity": round(float(exp.get("is_monotonicity", 0.0)), 3),
            "gates": "PASS" if exp.get("gates_passed") else "FAIL",
            "status": exp.get("status", "evaluated"),
            "eval_seconds": round(float(exp.get("eval_seconds", 0.0)), 1),
        }
        oos = oos_by_formula.get(exp.get("formula", ""))
        if oos is not None:
            row["oos"] = oos
        rows.append(json.dumps(row, ensure_ascii=False))

    prefix = ""
    if EXPERIMENTS_FILE.exists() and EXPERIMENTS_FILE.stat().st_size > 0:
        last_byte = EXPERIMENTS_FILE.read_bytes()[-1:]
        if last_byte != b"\n":
            prefix = "\n"

    with EXPERIMENTS_FILE.open("a", encoding="utf-8") as fh:
        if prefix:
            fh.write(prefix)
        for row in rows:
            fh.write(row)
            fh.write("\n")


def _compute_dsl_factor(
    formula: str,
    features_df: pd.DataFrame,
    sym_col: str,
    date_col: str,
) -> pd.DataFrame | None:
    """Parse and evaluate a DSL formula against feature data.

    Returns DataFrame with columns [sym_col, date_col, 'factor_value'], or None on error.

    NOTE: This is a simplified evaluator for MVP. It handles the most common
    patterns: rank(x), delta(x, n), pct_change(x, n), ts_mean(x, n),
    and simple arithmetic between features. More complex DSL evaluation
    will be built in the compute engine module.
    """
    try:
        ast = parse_dsl(formula)
    except DSLParseError as e:
        logger.warning(f"DSL parse error for '{formula}': {e}")
        return None

    try:
        values = _eval_ast(ast, features_df, sym_col, date_col)
        if values is None:
            return None

        result = features_df[[sym_col, date_col]].copy()
        result["factor_value"] = values
        result = result.dropna(subset=["factor_value"])
        return result

    except Exception as e:
        logger.warning(f"Factor computation error for '{formula}': {e}")
        return None


def _eval_ast(node, df: pd.DataFrame, sym_col: str, date_col: str) -> np.ndarray | None:
    """Recursively evaluate an AST node against the features DataFrame."""
    from src.dsl.parser import Literal, Feature, FunctionCall, BinOp, UnaryOp

    if isinstance(node, Literal):
        return np.full(len(df), node.value)

    if isinstance(node, Feature):
        if node.name not in df.columns:
            logger.warning(f"Feature '{node.name}' not in data columns: {list(df.columns)}")
            return None
        return df[node.name].values.astype(float)

    if isinstance(node, UnaryOp):
        operand = _eval_ast(node.operand, df, sym_col, date_col)
        if operand is None:
            return None
        if node.op == "-":
            return -operand
        return operand

    if isinstance(node, BinOp):
        left = _eval_ast(node.left, df, sym_col, date_col)
        right = _eval_ast(node.right, df, sym_col, date_col)
        if left is None or right is None:
            return None
        if node.op == "+":
            return left + right
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        if node.op == "/":
            return np.where(np.abs(right) > 1e-12, left / right, np.nan)
        return None

    if isinstance(node, FunctionCall):
        return _eval_function(node, df, sym_col, date_col)

    return None


def _eval_function(node, df: pd.DataFrame, sym_col: str, date_col: str) -> np.ndarray | None:
    """Evaluate a function call AST node."""
    from src.dsl.parser import FunctionCall, Literal

    name = node.name
    args = node.args

    # --- Cross-sectional operators ---
    if name == "rank":
        inner = _eval_ast(args[0], df, sym_col, date_col)
        if inner is None:
            return None
        temp = df[[date_col]].copy()
        temp["_val"] = inner
        temp["_rank"] = temp.groupby(date_col)["_val"].rank(pct=True, method="average")
        return temp["_rank"].values

    if name == "zscore":
        inner = _eval_ast(args[0], df, sym_col, date_col)
        if inner is None:
            return None
        temp = df[[date_col]].copy()
        temp["_val"] = inner
        g = temp.groupby(date_col)["_val"]
        temp["_z"] = (temp["_val"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        return np.clip(temp["_z"].values, -3, 3)

    if name == "demean":
        inner = _eval_ast(args[0], df, sym_col, date_col)
        if inner is None:
            return None
        temp = df[[date_col]].copy()
        temp["_val"] = inner
        temp["_dm"] = temp["_val"] - temp.groupby(date_col)["_val"].transform("mean")
        return temp["_dm"].values

    # --- Universal functions ---
    if name == "abs":
        inner = _eval_ast(args[0], df, sym_col, date_col)
        return np.abs(inner) if inner is not None else None

    if name == "sign":
        inner = _eval_ast(args[0], df, sym_col, date_col)
        return np.sign(inner) if inner is not None else None

    if name == "log":
        inner = _eval_ast(args[0], df, sym_col, date_col)
        return np.log(np.maximum(inner, 1e-9)) if inner is not None else None

    if name == "sqrt":
        inner = _eval_ast(args[0], df, sym_col, date_col)
        return np.sqrt(np.maximum(inner, 0)) if inner is not None else None

    if name == "power":
        base = _eval_ast(args[0], df, sym_col, date_col)
        if base is None:
            return None
        p = args[1].value if isinstance(args[1], Literal) else 2.0
        return np.power(np.abs(base), p) * np.sign(base)

    if name in ("max", "min"):
        a = _eval_ast(args[0], df, sym_col, date_col)
        b = _eval_ast(args[1], df, sym_col, date_col)
        if a is None or b is None:
            return None
        return np.maximum(a, b) if name == "max" else np.minimum(a, b)

    if name == "clamp":
        inner = _eval_ast(args[0], df, sym_col, date_col)
        if inner is None:
            return None
        lo = args[1].value if isinstance(args[1], Literal) else -np.inf
        hi = args[2].value if isinstance(args[2], Literal) else np.inf
        return np.clip(inner, lo, hi)

    if name == "if_then":
        cond = _eval_ast(args[0], df, sym_col, date_col)
        a = _eval_ast(args[1], df, sym_col, date_col)
        b = _eval_ast(args[2], df, sym_col, date_col)
        if cond is None or a is None or b is None:
            return None
        return np.where(cond > 0, a, b)

    # --- Time-series operators (require groupby sym) ---
    if name in ("ts_mean", "ts_std", "ts_max", "ts_min", "ts_sum",
                "ts_rank", "ts_skew", "ts_kurt", "ts_product",
                "delta", "pct_change", "shift", "decay_linear", "decay_exp",
                "ts_argmax", "ts_argmin", "ts_count",
                "ts_corr", "ts_cov"):
        return _eval_ts_function(name, args, df, sym_col, date_col)

    logger.warning(f"Unimplemented function: {name}")
    return None


def _eval_ts_function(
    name: str,
    args: list,
    df: pd.DataFrame,
    sym_col: str,
    date_col: str,
) -> np.ndarray | None:
    """Evaluate time-series functions that operate per-stock along time."""
    from src.dsl.parser import Literal

    # Most TS functions take (series, window) as last arg
    inner = _eval_ast(args[0], df, sym_col, date_col)
    if inner is None:
        return None

    window = int(args[-1].value) if isinstance(args[-1], Literal) else 20

    temp = df[[sym_col, date_col]].copy()
    temp["_val"] = inner
    temp = temp.sort_values([sym_col, date_col])

    result = np.full(len(df), np.nan)
    idx_map = {idx: i for i, idx in enumerate(temp.index)}

    if name == "ts_mean":
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.rolling(window, min_periods=max(1, window // 2)).mean()
        )
    elif name == "ts_std":
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.rolling(window, min_periods=max(2, window // 2)).std()
        )
    elif name == "ts_max":
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
    elif name == "ts_min":
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
    elif name == "ts_sum":
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )
    elif name == "ts_rank":
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.rolling(window, min_periods=max(1, window // 2)).apply(
                lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=True
            )
        )
    elif name == "ts_skew":
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.rolling(window, min_periods=max(3, window // 2)).skew()
        )
    elif name == "ts_kurt":
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.rolling(window, min_periods=max(4, window // 2)).kurt()
        )
    elif name == "ts_argmax":
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.rolling(window, min_periods=1).apply(
                lambda w: np.argmax(w), raw=True
            )
        )
    elif name == "ts_argmin":
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.rolling(window, min_periods=1).apply(
                lambda w: np.argmin(w), raw=True
            )
        )
    elif name == "ts_count":
        # Count positive values in window (cond is the first arg > 0)
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: (x > 0).rolling(window, min_periods=1).sum()
        )
    elif name == "ts_product":
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.rolling(window, min_periods=1).apply(np.prod, raw=True)
        )
    elif name == "delta":
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.diff(window)
        )
    elif name == "pct_change":
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.pct_change(window)
        )
    elif name == "shift":
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.shift(window)
        )
    elif name == "decay_linear":
        weights = np.arange(1, window + 1, dtype=float)
        weights /= weights.sum()
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.rolling(window, min_periods=window).apply(
                lambda w: np.dot(w, weights), raw=True
            )
        )
    elif name == "decay_exp":
        # Half-life = window
        alpha = 1 - np.exp(-np.log(2) / max(window, 1))
        temp["_result"] = temp.groupby(sym_col)["_val"].transform(
            lambda x: x.ewm(alpha=alpha, min_periods=max(1, window // 2)).mean()
        )
    elif name == "ts_corr":
        # ts_corr(x, y, n) — need second series
        inner2 = _eval_ast(args[1], df, sym_col, date_col)
        if inner2 is None:
            return None
        temp["_val2"] = inner2
        temp["_result"] = np.nan
        for sym, grp in temp.groupby(sym_col):
            idx = grp.index
            rolling_corr = grp["_val"].rolling(window, min_periods=max(3, window // 2)).corr(
                grp["_val2"]
            )
            temp.loc[idx, "_result"] = rolling_corr.values
    elif name == "ts_cov":
        inner2 = _eval_ast(args[1], df, sym_col, date_col)
        if inner2 is None:
            return None
        temp["_val2"] = inner2
        temp["_result"] = np.nan
        for sym, grp in temp.groupby(sym_col):
            idx = grp.index
            rolling_cov = grp["_val"].rolling(window, min_periods=max(3, window // 2)).cov(
                grp["_val2"]
            )
            temp.loc[idx, "_result"] = rolling_cov.values
    else:
        logger.warning(f"Unimplemented TS function: {name}")
        return None

    # Map back to original DataFrame index order
    return temp["_result"].values


# ---------------------------------------------------------------------------
# Session result
# ---------------------------------------------------------------------------

@dataclass
class SessionResult:
    session_id: str
    market: str
    budget: int
    experiments: list[dict]
    top3_oos: list[dict]  # top 3 by IS IC with OOS results
    summary: str


# ---------------------------------------------------------------------------
# Agent session
# ---------------------------------------------------------------------------

class FactorSession:
    """Run a full agent factor mining session."""

    def __init__(
        self,
        market: str,
        budget: int = 50,
        db_path: str | None = None,
        model: str = "claude-opus-4-6",
    ):
        self.market = market
        self.budget = budget
        self.model = model
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.cfg = MARKET_CONFIGS[market]
        self.experiments: list[dict] = []
        self._client = None

    def _load_existing_factors(self) -> list[dict]:
        """Load promoted + recently retired factors from registry for agent context."""
        import duckdb
        db_path = str(FACTOR_LAB_DB)
        try:
            con = duckdb.connect(db_path, read_only=True)
            rows = con.execute("""
                SELECT name, formula, status, ic_7d, ic_ir_7d,
                       promoted_at, retired_at, retire_reason
                FROM factor_registry
                WHERE market = ?
                  AND (status = 'promoted'
                       OR (status IN ('retired', 'watchlist')
                           AND retired_at >= CURRENT_TIMESTAMP - INTERVAL 7 DAY))
                ORDER BY status, promoted_at DESC
                LIMIT 25
            """, [self.market]).fetchall()
            con.close()
        except Exception:
            return []

        factors = []
        for name, formula, status, ic, ic_ir, promoted_at, retired_at, retire_reason in rows:
            entry = {
                "name": name or "unnamed",
                "formula": formula,
                "is_ic": ic or 0,
                "status": status,
            }
            if status == "promoted":
                entry["note"] = f"IC_IR={ic_ir:.3f}" if ic_ir else ""
            elif status == "retired":
                entry["note"] = f"retired: {retire_reason or 'unknown'}"
            elif status == "watchlist":
                entry["note"] = "degrading — on watchlist"
            factors.append(entry)

        logger.info(f"Loaded {len(factors)} existing factors ({sum(1 for f in factors if f['status']=='promoted')} promoted)")
        return factors

    def run(self) -> SessionResult:
        """Run full session: budget experiments, then OOS check top 3."""
        print(f"=== Factor Lab Session: {self.market.upper()} ===")
        print(f"Session ID: {self.session_id}")
        print(f"Budget: {self.budget} experiments")
        print()

        # Load data
        print("Loading prices...")
        prices = _load_prices(self.cfg)
        print(f"  {len(prices)} rows, {prices[self.cfg['sym_col']].nunique()} symbols")

        print("Computing forward returns...")
        fwd_returns = _compute_forward_returns(self.cfg)
        print(f"  {len(fwd_returns)} rows")

        print("Computing features...")
        features = _compute_basic_features(prices, self.cfg["sym_col"], self.cfg["date_col"])
        print(f"  {len(features)} feature rows")
        print()

        # Load existing promoted factors so agent avoids rediscovery
        existing_factors = self._load_existing_factors()

        # Build system prompt
        system_prompt = build_system_prompt(
            market=self.market,
            regime_dist=None,  # TODO: compute from data
            existing_factors=existing_factors,
        )

        # Main experiment loop
        for exp_i in range(self.budget):
            budget_remaining = self.budget - exp_i - 1
            exp_started = time.time()

            print(f"--- Experiment {exp_i + 1}/{self.budget} ---")

            # Build feedback prompt
            feedback = build_feedback_prompt(self.experiments, budget_remaining + 1)
            full_prompt = system_prompt + "\n\n" + feedback

            # Ask agent
            try:
                response_text = self._ask_agent(full_prompt)
            except Exception as e:
                logger.error(f"Agent call failed: {e}")
                print(f"  Agent call failed: {e}")
                continue

            # Parse response
            parsed = parse_agent_response(response_text)
            if parsed is None:
                print(f"  Could not parse agent response. Skipping.")
                self.experiments.append({
                    "name": "parse_error",
                    "formula": "",
                    "hypothesis": "Parse error",
                    "direction": "long",
                    "is_ic": 0.0,
                    "is_ic_ir": 0.0,
                    "is_sharpe": 0.0,
                    "is_turnover": 0.0,
                    "is_monotonicity": 0.0,
                    "gates_passed": False,
                    "gate_details": {},
                    "status": "parse_error",
                    "eval_seconds": time.time() - exp_started,
                    "error": "parse_error",
                })
                continue

            print(f"  Name: {parsed.name}")
            print(f"  Formula: {parsed.formula}")
            print(f"  Direction: {parsed.direction}")
            print(f"  Hypothesis: {parsed.hypothesis[:80]}...")

            # Compute factor values using the canonical DSL compute engine
            try:
                ast = parse_dsl(parsed.formula)
                factor_df = compute_factor(
                    ast, features,
                    sym_col=self.cfg["sym_col"],
                    date_col=self.cfg["date_col"],
                )
                factor_df = factor_df.dropna(subset=["factor_value"])
            except (DSLParseError, Exception) as e:
                logger.warning(f"Factor computation error for '{parsed.formula}': {e}")
                factor_df = None
            if factor_df is None or len(factor_df) == 0:
                print(f"  Factor computation failed.")
                self.experiments.append({
                    "name": parsed.name,
                    "formula": parsed.formula,
                    "hypothesis": parsed.hypothesis,
                    "direction": parsed.direction,
                    "is_ic": 0.0,
                    "is_ic_ir": 0.0,
                    "is_sharpe": 0.0,
                    "is_turnover": 0.0,
                    "is_monotonicity": 0.0,
                    "gates_passed": False,
                    "gate_details": {},
                    "status": "compute_error",
                    "eval_seconds": time.time() - exp_started,
                    "error": "compute_error",
                })
                continue

            # If direction is "short", flip the factor values
            if parsed.direction == "short":
                factor_df["factor_value"] = -factor_df["factor_value"]

            # Walk-forward IS backtest
            bt_result = walk_forward_backtest(
                factor_values=factor_df,
                forward_returns=fwd_returns,
                sym_col=self.cfg["sym_col"],
                date_col=self.cfg["date_col"],
                oos_start=self.cfg["oos_start"],
                n_folds=2,
                min_train_days=120,
                min_test_days=60,
                cost_per_trade=self.cfg["cost_per_trade"],
            )

            # Check gates
            gate_result = check_gates(bt_result, self.market)

            print(f"  IC={bt_result.avg_ic:.4f}  IC_IR={bt_result.avg_ic_ir:.3f}  "
                  f"Sharpe={bt_result.avg_sharpe:.3f}  Turnover={bt_result.avg_turnover:.3f}  "
                  f"Mono={bt_result.avg_monotonicity:.3f}")
            print(f"  Gates: {'PASS' if gate_result.passed else 'FAIL'}")

            exp_record = {
                "name": parsed.name,
                "formula": parsed.formula,
                "hypothesis": parsed.hypothesis,
                "direction": parsed.direction,
                "is_ic": bt_result.avg_ic,
                "is_ic_ir": bt_result.avg_ic_ir,
                "is_sharpe": bt_result.avg_sharpe,
                "is_turnover": bt_result.avg_turnover,
                "is_monotonicity": bt_result.avg_monotonicity,
                "gates_passed": gate_result.passed,
                "gate_details": gate_result.details,
                "fold_metrics": [
                    {
                        "train": f"{f.train_start}~{f.train_end}",
                        "test": f"{f.test_start}~{f.test_end}",
                        "ic": f.ic,
                        "ic_ir": f.ic_ir,
                        "sharpe": f.sharpe,
                    }
                    for f in bt_result.fold_metrics
                ],
                "status": "evaluated",
                "eval_seconds": time.time() - exp_started,
                # Keep factor_df reference for potential OOS check
                "_factor_df": factor_df,
            }
            self.experiments.append(exp_record)
            print()

        # --- OOS check: top 3 by IS IC ---
        print("=" * 60)
        print("Session complete. Running OOS checks on top 3 by IS IC...")

        # Filter to experiments that have valid factor data AND passed gates
        valid_exps = [
            e for e in self.experiments
            if "_factor_df" in e and e["_factor_df"] is not None and e.get("gates_passed")
        ]
        top3 = sorted(valid_exps, key=lambda e: abs(e["is_ic"]), reverse=True)[:3]

        top3_oos = []
        for exp in top3:
            oos_pass = run_oos_check(
                factor_values=exp["_factor_df"],
                forward_returns=fwd_returns,
                sym_col=self.cfg["sym_col"],
                date_col=self.cfg["date_col"],
                oos_start=self.cfg["oos_start"],
                market=self.market,
                cost_per_trade=self.cfg["cost_per_trade"],
            )
            print(f"  {exp['name']}: IS IC={exp['is_ic']:.4f}, OOS={'PASS' if oos_pass else 'FAIL'}")
            top3_oos.append({
                "name": exp["name"],
                "formula": exp["formula"],
                "hypothesis": exp["hypothesis"],
                "is_ic": exp["is_ic"],
                "is_ic_ir": exp["is_ic_ir"],
                "oos_pass": oos_pass,
            })

        # Clean up internal references before returning
        for exp in self.experiments:
            exp.pop("_factor_df", None)

        summary = self._build_summary(top3_oos)
        _append_experiment_rows(self.session_id, self.market, self.experiments, top3_oos)
        print()
        print(summary)

        return SessionResult(
            session_id=self.session_id,
            market=self.market,
            budget=self.budget,
            experiments=self.experiments,
            top3_oos=top3_oos,
            summary=summary,
        )

    def _ask_agent(self, prompt: str) -> str:
        return call_agent(
            prompt,
            model=self.model,
            max_tokens=1000,
            repo_root=Path(__file__).resolve().parents[2],
            claude_timeout=120,
            codex_timeout=300,
        )

    def _build_summary(self, top3_oos: list[dict]) -> str:
        """Build a human-readable session summary."""
        lines = [
            f"# Factor Lab Session Report",
            f"",
            f"- Session ID: {self.session_id}",
            f"- Market: {self.market.upper()}",
            f"- Budget: {self.budget} experiments",
            f"- Experiments run: {len(self.experiments)}",
            f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"",
            f"## All Experiments",
            f"",
            f"| # | Name | Formula | IC | IC_IR | Sharpe | Gates |",
            f"|---|------|---------|-----|-------|--------|-------|",
        ]

        for i, exp in enumerate(self.experiments, 1):
            gates_str = "PASS" if exp.get("gates_passed") else "FAIL"
            formula_short = exp.get("formula", "")[:40]
            lines.append(
                f"| {i} | {exp.get('name', '?')} | `{formula_short}` | "
                f"{exp.get('is_ic', 0):.4f} | {exp.get('is_ic_ir', 0):.3f} | "
                f"{exp.get('is_sharpe', 0):.3f} | {gates_str} |"
            )

        lines.extend([
            f"",
            f"## OOS Results (Top 3 by IS IC)",
            f"",
            f"| Name | Formula | IS IC | OOS |",
            f"|------|---------|-------|-----|",
        ])

        for exp in top3_oos:
            oos_str = "PASS" if exp.get("oos_pass") else "FAIL"
            lines.append(
                f"| {exp['name']} | `{exp['formula'][:50]}` | {exp['is_ic']:.4f} | {oos_str} |"
            )

        passed_oos = [e for e in top3_oos if e.get("oos_pass")]
        lines.extend([
            f"",
            f"## Summary",
            f"",
            f"- Factors passing OOS: {len(passed_oos)}/{len(top3_oos)}",
        ])

        if passed_oos:
            lines.append(f"- Candidates for promotion:")
            for e in passed_oos:
                fid = hashlib.sha256(e["formula"].encode()).hexdigest()[:12]
                lines.append(f"  - {e['name']} (id={fid}): `{e['formula']}`")
        else:
            lines.append(f"- No factors passed OOS. Consider different hypotheses next session.")

        lines.append(f"\n---\n*Generated by Factor Lab agent loop.*")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Factor Lab agent session")
    parser.add_argument("--market", choices=["cn", "us"], required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--output", default=None, help="Output markdown path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    session = FactorSession(
        market=args.market,
        budget=args.budget,
        model=args.model,
    )
    result = session.run()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(result.summary)
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
