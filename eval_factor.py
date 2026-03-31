#!/usr/bin/env python3
"""Factor evaluation CLI — single entry point for autoresearch mode.

Usage:
    uv run python eval_factor.py --market cn --formula "rank(delta(volume, 5))"
    uv run python eval_factor.py --market cn --formula "..." --oos-check
    uv run python eval_factor.py --market cn --formula "..." --promote --name "x" --hypothesis "y" --direction long
    uv run python eval_factor.py --show-registry --market cn
    uv run python eval_factor.py --eval-composite --market cn
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pickle
import duckdb
import numpy as np
import pandas as pd

from src.dsl.parser import parse, DSLParseError
from src.dsl.compute import compute_factor
from src.backtest.walk_forward import walk_forward_backtest, run_oos_check
from src.backtest.gates import check_gates, GateResult
from src.paths import FACTOR_LAB_DB, FACTOR_LAB_ROOT, QUANT_CN_DB, QUANT_US_DB

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = FACTOR_LAB_ROOT / "data" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Market configuration — mirrors src/agent/loop.py:51-72
# ---------------------------------------------------------------------------
MARKET_CONFIGS = {
    "cn": {
        "db_path": str(QUANT_CN_DB),
        "table": "prices",
        "sym_col": "ts_code",
        "date_col": "trade_date",
        "close_col": "close",
        "vol_col": "vol",
        "cost_per_trade": 0.003,
        "oos_start": "2025-10-01",
        "universe_top_n": 2000,  # Rolling top N by market_cap per day
    },
    "us": {
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
# Data caching — avoids 10-30s DB loads on every experiment
# ---------------------------------------------------------------------------
def _cached_load(market: str, loader, name: str, max_age_hours: int = 24):
    """Cache DataFrames to pickle for fast repeated access."""
    cache_file = CACHE_DIR / f"{market}_{name}.pkl"
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            return pickle.load(open(cache_file, "rb"))
    data = loader(market)
    pickle.dump(data, open(cache_file, "wb"))
    return data


# ---------------------------------------------------------------------------
# Data loading — adapted from src/agent/loop.py:79-117
# Uses SELECT * to include all available columns (amount, turnover_rate, etc.)
# so DSL formulas can reference any column that exists in the pipeline DB.
# ---------------------------------------------------------------------------
def _open_db_readonly(db_path: str):
    """Open DuckDB read-only, falling back to file copy if write-locked.

    Temp files are registered for cleanup at process exit via atexit.
    """
    try:
        return duckdb.connect(db_path, read_only=True)
    except Exception:
        import shutil, tempfile, atexit
        tmp = tempfile.mktemp(suffix=".duckdb")
        shutil.copy2(db_path, tmp)
        atexit.register(lambda p=tmp: os.unlink(p) if os.path.exists(p) else None)
        return duckdb.connect(tmp, read_only=True)


def _load_prices_raw(market: str) -> pd.DataFrame:
    """Load raw prices from pipeline DuckDB with enrichment from auxiliary tables."""
    cfg = MARKET_CONFIGS[market]
    con = _open_db_readonly(cfg["db_path"])
    sym = cfg["sym_col"]
    dt = cfg["date_col"]
    close_col = cfg["close_col"]

    if market == "cn":
        # CN: JOIN prices with daily_basic (fundamentals) + moneyflow + margin_detail
        sql = f"""
            SELECT
                p.*,
                db.turnover_rate,
                db.volume_ratio,
                db.pe_ttm,
                db.pb,
                db.ps_ttm,
                db.total_mv AS market_cap,
                db.circ_mv AS circ_market_cap,
                mf.net_mf_amount,
                mf.buy_elg_amount - mf.sell_elg_amount AS large_net_in,
                mg.rzye AS margin_balance
            FROM {cfg['table']} p
            LEFT JOIN daily_basic db
                ON p.{sym} = db.ts_code AND p.{dt} = db.trade_date
            LEFT JOIN moneyflow mf
                ON p.{sym} = mf.ts_code AND p.{dt} = mf.trade_date
            LEFT JOIN margin_detail mg
                ON p.{sym} = mg.ts_code AND p.{dt} = mg.trade_date
            WHERE p.{close_col} > 0
            ORDER BY p.{sym}, p.{dt}
        """
    else:
        # US: explicit column list to avoid duplicate 'close' from adj_close rename
        sql = f"""
            SELECT {sym}, {dt}, open, high, low, {close_col}, volume
            FROM {cfg['table']}
            WHERE {close_col} > 0
            ORDER BY {sym}, {dt}
        """

    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()

    # Normalize key column names for compute_factor compatibility
    rename = {}
    if close_col != "close":
        rename[close_col] = "close"
    if cfg["vol_col"] != "volume":
        rename[cfg["vol_col"]] = "volume"
    if rename:
        df = df.rename(columns=rename)

    # Compute margin_delta_5d for CN if margin_balance exists
    if market == "cn" and "margin_balance" in df.columns:
        df["margin_delta_5d"] = df.groupby(sym)["margin_balance"].transform(
            lambda s: s - s.shift(5)
        )

    # Universe filter: keep only top N stocks by market_cap each day
    top_n = cfg.get("universe_top_n")
    if top_n and "market_cap" in df.columns:
        df["_mcap_rank"] = df.groupby(dt)["market_cap"].rank(
            ascending=False, method="first", na_option="bottom"
        )
        df = df[df["_mcap_rank"] <= top_n].drop(columns=["_mcap_rank"]).reset_index(drop=True)

    return df


def load_prices(market: str) -> pd.DataFrame:
    return _cached_load(market, _load_prices_raw, "prices")


def _compute_fwd_raw(market: str) -> pd.DataFrame:
    """Compute 5-day forward returns via DuckDB window function."""
    cfg = MARKET_CONFIGS[market]
    con = _open_db_readonly(cfg["db_path"])
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


def compute_forward_returns(market: str) -> pd.DataFrame:
    return _cached_load(market, _compute_fwd_raw, "fwd")


# ---------------------------------------------------------------------------
# Promoted factor loading (for correlation gate)
# ---------------------------------------------------------------------------
def load_promoted_factor_values(market: str, prices: pd.DataFrame) -> tuple[list[pd.Series], list[str]]:
    """Load promoted factors from registry, re-compute their values, return as list of Series.

    Returns (factor_value_series_list, factor_name_list).
    """
    cfg = MARKET_CONFIGS[market]
    con = duckdb.connect(FACTOR_LAB_DB, read_only=True)
    try:
        promoted = con.execute(
            "SELECT factor_id, name, formula FROM factor_registry WHERE market=? AND status='promoted'",
            [market],
        ).fetchdf()
    except duckdb.CatalogException:
        return [], []
    finally:
        con.close()

    if promoted.empty:
        return [], []

    series_list = []
    name_list = []
    for _, row in promoted.iterrows():
        try:
            ast = parse(row["formula"])
            vals = compute_factor(ast, prices, sym_col=cfg["sym_col"], date_col=cfg["date_col"])
            series_list.append(vals["factor_value"])
            name_list.append(row["name"] or row["factor_id"][:7])
        except Exception:
            continue  # skip broken promoted factors

    return series_list, name_list


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_results(
    market: str,
    formula: str,
    bt,
    gate_result: GateResult,
    elapsed: float,
) -> None:
    """Print grep-friendly results to stdout."""
    print("---")
    print(f"market:           {market}")
    print(f"formula:          {formula}")
    print(f"is_ic:            {bt.avg_ic:.4f}")
    print(f"is_ic_ir:         {bt.avg_ic_ir:.3f}")
    print(f"is_sharpe:        {bt.avg_sharpe:.3f}")
    print(f"is_turnover:      {bt.avg_turnover:.4f}")
    print(f"is_monotonicity:  {bt.avg_monotonicity:.3f}")

    corr_info = gate_result.details.get("correlation", {})
    print(f"max_corr:         {corr_info.get('value', 0.0):.3f}")
    print(f"max_corr_with:    {corr_info.get('most_correlated', 'none')}")

    gate_str = "PASS" if gate_result.passed else "FAIL"
    detail_parts = []
    for gname, ginfo in gate_result.details.items():
        detail_parts.append(f"{gname}={'PASS' if ginfo['passed'] else 'FAIL'}")
    print(f"gates:            {gate_str}")
    print(f"gate_detail:      {' '.join(detail_parts)}")
    print(f"eval_seconds:     {elapsed:.1f}")


# ---------------------------------------------------------------------------
# Ensure DuckDB tables exist — reuses schema from daily_pipeline.py:69
# ---------------------------------------------------------------------------
def ensure_tables():
    """Create factor_registry table if it doesn't exist."""
    from src.mining.daily_pipeline import init_db
    init_db()


# ---------------------------------------------------------------------------
# Promote to DuckDB — adapts src/mining/daily_pipeline.py:265
# ---------------------------------------------------------------------------
def promote_factor(
    market: str,
    formula: str,
    name: str,
    hypothesis: str,
    direction: str,
    bt,
) -> str:
    """Write factor to DuckDB registry as promoted. Populates IS metrics. Returns factor_id."""
    ensure_tables()
    factor_id = hashlib.sha256(f"{market}:{formula}".encode()).hexdigest()[:16]
    con = duckdb.connect(FACTOR_LAB_DB)
    try:
        existing = con.execute(
            "SELECT factor_id FROM factor_registry WHERE factor_id=?", [factor_id]
        ).fetchone()

        # Use 5d IS metrics for the 7d columns (closest horizon we have)
        ic_val = bt.avg_ic if bt else None
        icir_val = bt.avg_ic_ir if bt else None
        mono_val = bt.avg_monotonicity if bt else None

        if existing:
            con.execute("""
                UPDATE factor_registry SET
                    name=?, hypothesis=?, direction=?,
                    status='promoted', promoted_at=CURRENT_TIMESTAMP,
                    watchlist_at=NULL, retired_at=NULL, retire_reason=NULL,
                    health_watch_count=0,
                    ic_7d=?, ic_ir_7d=?, mono_7d=?
                WHERE factor_id=?
            """, [name, hypothesis, direction, ic_val, icir_val, mono_val, factor_id])
        else:
            con.execute("""
                INSERT INTO factor_registry
                    (factor_id, market, name, hypothesis, formula, direction,
                     status, promoted_at, ic_7d, ic_ir_7d, mono_7d)
                VALUES (?, ?, ?, ?, ?, ?, 'promoted', CURRENT_TIMESTAMP, ?, ?, ?)
            """, [factor_id, market, name, hypothesis, formula, direction,
                  ic_val, icir_val, mono_val])
    finally:
        con.close()

    return factor_id


# ---------------------------------------------------------------------------
# Show registry
# ---------------------------------------------------------------------------
def show_registry(market: str) -> None:
    """Print promoted factors for a market."""
    con = duckdb.connect(FACTOR_LAB_DB, read_only=True)
    try:
        df = con.execute("""
            SELECT name, formula, status, composite_score,
                   ic_7d, ic_ir_7d, promoted_at
            FROM factor_registry
            WHERE market=?
            ORDER BY composite_score DESC NULLS LAST
        """, [market]).fetchdf()
    except duckdb.CatalogException:
        print("---")
        print("registry:         empty (no tables)")
        return
    finally:
        con.close()

    print("---")
    print(f"market:           {market}")
    print(f"promoted_count:   {len(df[df['status'] == 'promoted'])}")
    print(f"total_count:      {len(df)}")
    print()
    for _, row in df.iterrows():
        status = row["status"]
        name = row["name"] or "unnamed"
        formula = row["formula"]
        print(f"  [{status:>10s}] {name}: {formula}")


# ---------------------------------------------------------------------------
# Eval composite
# ---------------------------------------------------------------------------
def eval_composite(market: str) -> None:
    """Evaluate the IC_IR-weighted composite of all promoted factors."""
    cfg = MARKET_CONFIGS[market]
    sym_col, date_col = cfg["sym_col"], cfg["date_col"]

    prices = load_prices(market)
    fwd = compute_forward_returns(market)

    con = duckdb.connect(FACTOR_LAB_DB, read_only=True)
    try:
        promoted = con.execute(
            "SELECT factor_id, name, formula, ic_ir_7d FROM factor_registry WHERE market=? AND status='promoted'",
            [market],
        ).fetchdf()
    except duckdb.CatalogException:
        print("---")
        print("composite:        no promoted factors")
        return
    finally:
        con.close()

    if promoted.empty:
        print("---")
        print("composite:        no promoted factors")
        return

    # Compute each factor and combine
    factor_dfs = {}
    weights = {}
    for _, row in promoted.iterrows():
        try:
            ast = parse(row["formula"])
            vals = compute_factor(ast, prices, sym_col=sym_col, date_col=date_col)
            fid = row["factor_id"]
            factor_dfs[fid] = vals
            # IC_IR-weighted (fallback to 1.0 if NULL/NaN)
            ir = row["ic_ir_7d"]
            w = abs(ir) if ir is not None and not (isinstance(ir, float) and np.isnan(ir)) else 1.0
            weights[fid] = max(w, 0.01)
        except Exception:
            continue

    if not factor_dfs:
        print("---")
        print("composite:        all factor computations failed")
        return

    # Normalize weights
    total_w = sum(weights.values())
    for k in weights:
        weights[k] /= total_w

    # Merge all factor values into composite
    base = None
    for fid, fdf in factor_dfs.items():
        renamed = fdf.rename(columns={"factor_value": f"f_{fid}"})
        if base is None:
            base = renamed
        else:
            base = base.merge(renamed, on=[sym_col, date_col], how="outer")

    # Weighted sum
    composite_vals = pd.Series(0.0, index=base.index)
    for fid in factor_dfs:
        col = f"f_{fid}"
        if col in base.columns:
            composite_vals += base[col].fillna(0) * weights[fid]

    base["factor_value"] = composite_vals
    composite_df = base[[sym_col, date_col, "factor_value"]].dropna()

    # Backtest composite
    bt = walk_forward_backtest(
        composite_df, fwd,
        sym_col=sym_col, date_col=date_col,
        oos_start=cfg["oos_start"],
        cost_per_trade=cfg["cost_per_trade"],
    )

    print("---")
    print(f"composite_ic:     {bt.avg_ic:.4f}")
    print(f"composite_ic_ir:  {bt.avg_ic_ir:.3f}")
    print(f"composite_sharpe: {bt.avg_sharpe:.3f}")
    print(f"n_factors:        {len(factor_dfs)}")
    weights_str = ", ".join(f"{fid[:7]}={w:.2f}" for fid, w in weights.items())
    print(f"weights:          {weights_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Factor evaluation CLI for autoresearch mode")
    parser.add_argument("--market", choices=["cn", "us"], help="Target market")
    parser.add_argument("--formula", type=str, help="DSL formula to evaluate")
    parser.add_argument("--name", type=str, default="", help="Factor name")
    parser.add_argument("--hypothesis", type=str, default="", help="Economic hypothesis")
    parser.add_argument("--direction", choices=["long", "short"], default="long")
    parser.add_argument("--oos-check", action="store_true", help="Run OOS validation (PASS/FAIL)")
    parser.add_argument("--promote", action="store_true", help="Promote factor to registry")
    parser.add_argument("--show-registry", action="store_true", help="Show promoted factors")
    parser.add_argument("--eval-composite", action="store_true", help="Evaluate promoted factor composite")

    args = parser.parse_args()

    # Dispatch modes
    if args.show_registry:
        if not args.market:
            print("ERROR: --market required", file=sys.stderr)
            return 1
        show_registry(args.market)
        return 0

    if args.eval_composite:
        if not args.market:
            print("ERROR: --market required", file=sys.stderr)
            return 1
        eval_composite(args.market)
        return 0

    # Formula-based modes require --market and --formula
    if not args.market or not args.formula:
        print("ERROR: --market and --formula required", file=sys.stderr)
        return 1

    cfg = MARKET_CONFIGS[args.market]
    sym_col, date_col = cfg["sym_col"], cfg["date_col"]

    # Timing starts before data load (fix: t0 before parse, not after)
    t0 = time.time()

    # 1. Parse DSL
    try:
        ast = parse(args.formula)
    except DSLParseError as e:
        print(f"PARSE_ERROR: {e}", file=sys.stderr)
        return 1

    # 2. Load data (cached — first call ~10s, subsequent <0.5s)
    try:
        prices = load_prices(args.market)
        fwd = compute_forward_returns(args.market)
    except Exception as e:
        print(f"DATA_ERROR: {e}", file=sys.stderr)
        return 2

    # 3. Compute factor values
    try:
        factor_values = compute_factor(ast, prices, sym_col=sym_col, date_col=date_col)
    except Exception as e:
        print(f"COMPUTE_ERROR: {e}", file=sys.stderr)
        return 2

    if factor_values.empty or factor_values["factor_value"].isna().all():
        print("COMPUTE_ERROR: factor produced all NaN values", file=sys.stderr)
        return 2

    # Apply direction: flip factor values for short factors
    if args.direction == "short":
        factor_values["factor_value"] = -factor_values["factor_value"]

    # 4. Walk-forward backtest
    try:
        bt = walk_forward_backtest(
            factor_values, fwd,
            sym_col=sym_col, date_col=date_col,
            oos_start=cfg["oos_start"],
            cost_per_trade=cfg["cost_per_trade"],
        )
    except Exception as e:
        print(f"BACKTEST_ERROR: {e}", file=sys.stderr)
        return 3

    # 5. Check gates (with correlation against promoted factors)
    promoted_series, promoted_names = load_promoted_factor_values(args.market, prices)

    # Filter to IS-only for gate checks (prevent OOS leakage)
    oos_start = cfg["oos_start"]
    if promoted_series:
        is_mask = factor_values[date_col] < oos_start
        candidate_values_is = factor_values.loc[is_mask, "factor_value"]
        candidate_dates_is = factor_values.loc[is_mask, date_col]
    else:
        candidate_values_is = None
        candidate_dates_is = None

    gate_result = check_gates(
        bt, args.market,
        existing_factors=promoted_series if promoted_series else None,
        candidate_values=candidate_values_is,
        candidate_dates=candidate_dates_is,
    )

    # Update correlation gate with actual factor names
    if promoted_names and "correlation" in gate_result.details:
        corr_detail = gate_result.details["correlation"]
        mc = corr_detail.get("most_correlated", "")
        if mc and mc.startswith("factor_"):
            idx = int(mc.split("_")[1])
            if idx < len(promoted_names):
                corr_detail["most_correlated"] = promoted_names[idx]

    elapsed = time.time() - t0

    # 6. Print IS results
    print_results(args.market, args.formula, bt, gate_result, elapsed)

    # 7. OOS check
    if args.oos_check:
        if not gate_result.passed:
            print(f"oos_result:       SKIP (gates failed)")
        else:
            oos_pass = run_oos_check(
                factor_values, fwd,
                sym_col=sym_col, date_col=date_col,
                oos_start=cfg["oos_start"],
                market=args.market,
                cost_per_trade=cfg["cost_per_trade"],
            )
            print(f"oos_result:       {'PASS' if oos_pass else 'FAIL'}")

    # 8. Promote (only if gates passed — defense in depth)
    if args.promote:
        if not args.name:
            print("ERROR: --name required for --promote", file=sys.stderr)
            return 1
        if not gate_result.passed:
            print("PROMOTE_SKIP:     gates did not pass", file=sys.stderr)
            return 1
        factor_id = promote_factor(
            args.market, args.formula, args.name,
            args.hypothesis, args.direction, bt,
        )
        print(f"promoted_id:      {factor_id}")

    # 9. Append to experiments.jsonl (auto-log after every evaluation)
    log_entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "market": args.market,
        "formula": args.formula,
        "name": args.name or "",
        "is_ic": round(bt.avg_ic, 4),
        "is_ic_ir": round(bt.avg_ic_ir, 3),
        "gates": "PASS" if gate_result.passed else "FAIL",
        "status": "promoted" if args.promote and gate_result.passed else "evaluated",
        "eval_seconds": round(elapsed, 1),
    }
    log_path = Path(__file__).parent / "experiments.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
