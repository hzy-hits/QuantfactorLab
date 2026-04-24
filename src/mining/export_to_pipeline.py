#!/usr/bin/env python3
"""
Export promoted factors from Factor Lab → pipeline analytics tables.

Called by pipeline cron BEFORE filtering/rendering steps.
Reads factor_lab.duckdb → computes today's factor values → writes to pipeline DB.

Usage:
    python3 -m src.mining.export_to_pipeline --market cn --date 2026-03-18
    python3 -m src.mining.export_to_pipeline --market us --date 2026-03-17
"""
import sys
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dsl.parser import parse
from src.dsl.compute import compute_factor
from src.paths import FACTOR_LAB_DB, QUANT_CN_DB, QUANT_US_DB

PIPELINE_CONFIGS = {
    "cn": {
        "db_path": str(QUANT_CN_DB),
        "price_sql": """
            SELECT p.ts_code, p.trade_date, p.open, p.high, p.low, p.close,
                   p.vol as volume, p.amount,
                   db.turnover_rate, db.pe_ttm, db.pb, db.ps_ttm,
                   db.total_mv AS market_cap, db.circ_mv AS circ_market_cap
            FROM prices p
            LEFT JOIN daily_basic db
                ON p.ts_code = db.ts_code AND p.trade_date = db.trade_date
            WHERE p.close > 0
            ORDER BY p.ts_code, p.trade_date
        """,
        "sym_col": "ts_code",
        "date_col": "trade_date",
        "universe_top_n": 2000,
        "insert_sql": "INSERT OR REPLACE INTO analytics (ts_code, as_of, module, metric, value, detail) VALUES (?, ?, 'lab_factor', ?, ?, ?)",
    },
    "us": {
        "db_path": str(QUANT_US_DB),
        "price_sql": "SELECT symbol as ts_code, date as trade_date, open, high, low, adj_close as close, volume FROM prices_daily WHERE adj_close > 0 ORDER BY symbol, date",
        "sym_col": "ts_code",
        "date_col": "trade_date",
        "insert_sql": None,  # US uses analysis_daily, handled differently
    },
}


def _load_saved_weights(
    lab_con: duckdb.DuckDBPyConnection,
    market: str,
    factor_ids: list[str],
) -> dict[str, float]:
    if not factor_ids:
        return {}

    try:
        latest_as_of = lab_con.execute(
            "SELECT MAX(as_of) FROM factor_weights WHERE market=?",
            [market],
        ).fetchone()[0]
    except duckdb.Error:
        return {}
    if latest_as_of is None:
        return {}

    factor_id_set = set(factor_ids)
    rows = lab_con.execute("""
        SELECT factor_id, weight
        FROM factor_weights
        WHERE market=? AND as_of=?
    """, [market, latest_as_of]).fetchall()

    weights = {
        factor_id: float(weight)
        for factor_id, weight in rows
        if factor_id in factor_id_set and weight is not None and weight > 0
    }
    if len(weights) != len(factor_ids):
        return {}

    total = sum(weights.values())
    if total <= 0:
        return {}

    return {factor_id: weight / total for factor_id, weight in weights.items()}


def _resolve_direction(
    stored_direction: str | None,
    ic_7d: float | None,
    ic_14d: float | None,
    ic_30d: float | None,
) -> str:
    weighted_ic = 0.5 * float(ic_7d or 0.0) + 0.3 * float(ic_14d or 0.0) + 0.2 * float(ic_30d or 0.0)
    if abs(weighted_ic) > 1e-12:
        return "short" if weighted_ic < 0 else "long"

    direction = (stored_direction or "long").lower()
    return direction if direction in {"long", "short"} else "long"


def _orient_factor_values_for_direction(
    factor_df: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    """Return factor values where higher values always mean better long candidates."""
    if direction != "short":
        return factor_df
    out = factor_df.copy()
    out["factor_value"] = -out["factor_value"]
    return out


def _resolve_effective_trade_date(prices: pd.DataFrame, requested_as_of: str) -> pd.Timestamp | None:
    """Use the latest available trade date on or before the requested report date."""
    if prices.empty:
        return None

    requested_ts = pd.Timestamp(requested_as_of)
    trade_dates = pd.to_datetime(prices["trade_date"], errors="coerce").dropna()
    eligible = trade_dates[trade_dates <= requested_ts]
    if eligible.empty:
        return None
    return eligible.max()


def _load_prices_with_fallback(db_path: str, price_sql: str) -> pd.DataFrame:
    """Load prices from the primary DB, or a temp snapshot if the file is locked."""
    try:
        con = duckdb.connect(db_path, read_only=True)
        try:
            return con.execute(price_sql).fetchdf()
        finally:
            con.close()
    except Exception as exc:
        print(f"  Price DB locked, reading from temp snapshot: {exc}")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = Path(tmpdir) / Path(db_path).name
            shutil.copy2(db_path, tmp_db)
            con = duckdb.connect(str(tmp_db), read_only=True)
            try:
                return con.execute(price_sql).fetchdf()
            finally:
                con.close()


def _connect_for_write(db_path: str, retries: int = 12, delay_seconds: float = 5.0) -> duckdb.DuckDBPyConnection:
    """Open the target DB for writing, retrying briefly if another process holds the lock."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return duckdb.connect(db_path)
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            print(f"  Target DB locked for write, retrying ({attempt}/{retries})...")
            time.sleep(delay_seconds)
    raise RuntimeError(f"Could not open writable pipeline DB {db_path}: {last_exc}") from last_exc


def _recent_history_dates(
    prices: pd.DataFrame,
    date_col: str,
    effective_trade_date: pd.Timestamp,
    lookback_days: int = 120,
    holdout_days: int = 5,
) -> set[pd.Timestamp]:
    """Recent dates whose forward returns are fully known at effective_trade_date."""
    trade_dates = sorted(pd.to_datetime(prices[date_col], errors="coerce").dropna().unique())
    if not trade_dates:
        return set()

    current = pd.Timestamp(effective_trade_date).to_datetime64()
    try:
        current_idx = trade_dates.index(current)
    except ValueError:
        return set()

    end_idx = max(0, current_idx - holdout_days)
    start_idx = max(0, end_idx - lookback_days)
    return {pd.Timestamp(dt) for dt in trade_dates[start_idx:end_idx]}


def _select_best_quintile(
    factor_df: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    history_dates: set[pd.Timestamp],
    sym_col: str,
    date_col: str,
    min_history_dates: int = 20,
) -> int:
    """Choose between Q1 and Q5 based on recent realized forward returns."""
    if not history_dates:
        return 5

    history = factor_df[factor_df[date_col].isin(history_dates)].merge(
        fwd_returns, on=[sym_col, date_col], how="inner"
    )
    if history.empty:
        return 5

    quintile_means = []
    for _, group in history.groupby(date_col):
        valid = group.dropna(subset=["factor_value", "fwd_5d"])
        if len(valid) < 25:
            continue
        try:
            valid = valid.copy()
            valid["quintile"] = pd.qcut(
                valid["factor_value"], 5, labels=False, duplicates="drop"
            ) + 1
        except ValueError:
            continue
        if valid["quintile"].nunique() < 5:
            continue
        quintile_means.append(valid.groupby("quintile")["fwd_5d"].mean())

    if len(quintile_means) < min_history_dates:
        return 5

    means = pd.concat(quintile_means, axis=1).mean(axis=1)
    q1_mean = float(means.get(1, float("-inf")))
    q5_mean = float(means.get(5, float("-inf")))
    return 1 if q1_mean > q5_mean else 5


def export(market: str, as_of: str | None = None):
    """Export promoted factors to pipeline DB."""
    cfg = PIPELINE_CONFIGS[market]
    as_of = as_of or date.today().isoformat()

    # 1. Read promoted factors from factor_lab.duckdb
    if not FACTOR_LAB_DB.exists():
        print(f"  Factor Lab DB not found, skipping")
        return 0

    lab_con = duckdb.connect(str(FACTOR_LAB_DB), read_only=True)
    promoted = lab_con.execute("""
        SELECT factor_id, formula, name, composite_score, direction, ic_7d, ic_14d, ic_30d
        FROM factor_registry
        WHERE market = ? AND status = 'promoted'
        ORDER BY composite_score DESC
    """, [market]).fetchall()
    saved_weights = _load_saved_weights(lab_con, market, [row[0] for row in promoted])
    lab_con.close()

    if not promoted:
        print(f"  No promoted factors for {market}, skipping")
        return 0

    print(f"  {len(promoted)} promoted factors for {market}")

    # 2. Load prices (with lock-safe fallback)
    prices = _load_prices_with_fallback(cfg["db_path"], cfg["price_sql"])

    # Universe filter (CN only)
    top_n = cfg.get("universe_top_n")
    if top_n and "market_cap" in prices.columns:
        prices["_r"] = prices.groupby("trade_date")["market_cap"].rank(
            ascending=False, method="first", na_option="bottom"
        )
        prices = prices[prices["_r"] <= top_n].drop(columns=["_r"]).reset_index(drop=True)
        print(f"  Universe filter: top {top_n} by market_cap")

    effective_trade_date = _resolve_effective_trade_date(prices, as_of)
    if effective_trade_date is None:
        print(f"  No trade_date <= {as_of} in pipeline DB, skipping")
        return 0

    if effective_trade_date.date().isoformat() != as_of:
        print(
            f"  Using latest available trade_date {effective_trade_date.date().isoformat()} "
            f"for requested as_of {as_of}"
        )

    # 3. Compute each factor and combine into lab_composite
    sym_col = cfg["sym_col"]
    date_col = cfg["date_col"]
    prices_sorted = prices.sort_values([sym_col, date_col]).copy()
    prices_sorted["fwd_5d"] = (
        prices_sorted.groupby(sym_col)["close"].shift(-5) / prices_sorted["close"] - 1
    )
    fwd_returns = prices_sorted[[sym_col, date_col, "fwd_5d"]].dropna()
    history_dates = _recent_history_dates(prices_sorted, date_col, effective_trade_date)

    factor_values = {}
    factor_best_quintile = {}

    for factor_id, formula, name, score, direction, ic_7d, ic_14d, ic_30d in promoted:
        try:
            ast = parse(formula)
            fdf = compute_factor(ast, prices, sym_col=sym_col, date_col=date_col)
            fdf = fdf[[sym_col, date_col, "factor_value"]].copy()
            effective_direction = _resolve_direction(direction, ic_7d, ic_14d, ic_30d)
            fdf = _orient_factor_values_for_direction(fdf, effective_direction)

            # Filter to effective trade date, but write under requested report date.
            today_values = fdf[fdf[date_col] == effective_trade_date]
            if len(today_values) > 0:
                series = today_values.set_index(sym_col)["factor_value"]
                factor_values[factor_id] = series
                best_q = _select_best_quintile(
                    fdf, fwd_returns, history_dates, sym_col, date_col
                )
                factor_best_quintile[factor_id] = best_q
                print(
                    f"    {name}: {len(today_values)} values computed "
                    f"({effective_direction}, best_q=Q{best_q}, "
                    f"trade_date={effective_trade_date.date().isoformat()})"
                )
            else:
                print(
                    f"    {name}: no values for requested {as_of} "
                    f"(effective trade_date={effective_trade_date.date().isoformat()})"
                )
        except Exception as e:
            print(f"    {name}: compute error — {e}")

    if not factor_values:
        return 0

    # 4. Rolling best-factor selection.
    # Instead of combining all factors (voting/weighting), pick the SINGLE best factor
    # from the recent lookback window and rank stocks by that factor.
    from src.strategy.rolling_best import select_best_factor, StrategyConfig
    from scipy.stats import spearmanr

    strat_cfg = StrategyConfig(lookback=40, hold_max=5, n_picks=10)

    # Compute multi-horizon returns for factor evaluation
    prices_sorted[f"ret_{strat_cfg.hold_max}d"] = (
        prices_sorted.groupby(sym_col)["close"].shift(-strat_cfg.hold_max)
        / prices_sorted["close"] - 1
    )

    # Build factor DataFrames with returns for select_best_factor
    factor_dfs_for_select = {}
    for fid, series in factor_values.items():
        fdf = series.reset_index()
        fdf.columns = [sym_col, "factor_value"]
        fdf[date_col] = effective_trade_date
        # Merge with historical data for lookback evaluation
        full_fdf = None
        for factor_id2, formula2, name2, _score2, direction2, ic7_2, ic14_2, ic30_2 in promoted:
            if factor_id2 == fid:
                try:
                    ast2 = parse(formula2)
                    full_fdf = compute_factor(ast2, prices, sym_col=sym_col, date_col=date_col)
                    dir2 = _resolve_direction(direction2, ic7_2, ic14_2, ic30_2)
                    full_fdf = _orient_factor_values_for_direction(full_fdf, dir2)
                except Exception:
                    pass
                break
        if full_fdf is not None:
            merged = full_fdf.merge(
                prices_sorted[[sym_col, date_col, f"ret_{strat_cfg.hold_max}d"]],
                on=[sym_col, date_col], how="left"
            )
            factor_dfs_for_select[fid] = merged

    # Get lookback dates
    all_dates = sorted(prices_sorted[date_col].unique())
    try:
        eff_idx = list(all_dates).index(effective_trade_date)
    except ValueError:
        eff_idx = len(all_dates) - 1
    lookback_dates = all_dates[max(0, eff_idx - strat_cfg.lookback):eff_idx]

    # Select best factor
    best_factor, best_side, best_sharpe = select_best_factor(
        factor_dfs_for_select, lookback_dates, strat_cfg.hold_max, date_col, strat_cfg.n_picks
    )

    selected_factor_name = None
    selected_side = None
    selected_sharpe = None
    if best_factor is None:
        print("  No factor qualified in rolling selection, exporting zeros")
        composite = {sym: 0.0 for fv in factor_values.values() for sym in fv.index}
    else:
        # Get the factor name for logging
        factor_name_map = {fid: name for fid, formula, name, *_ in promoted}
        sel_name = factor_name_map.get(best_factor, best_factor)
        selected_factor_name = sel_name
        selected_side = best_side
        selected_sharpe = round(float(best_sharpe), 4)
        print(f"  Selected: {sel_name} ({best_side}), lookback_sharpe={best_sharpe:.2f}")

        # Rank all stocks by this factor
        best_fv = factor_values[best_factor].dropna()
        ranked = best_fv.rank(pct=True, ascending=(best_side == "top"))
        # Top N_PICKS get score in (0, 1], rest get 0
        threshold = 1.0 - strat_cfg.n_picks / len(ranked)
        composite = {}
        for sym, pct in ranked.items():
            if pct >= threshold:
                composite[sym] = round(float(pct), 4)
            else:
                composite[sym] = 0.0

    n_active = sum(1 for v in composite.values() if v > 0.1)
    print(f"  Rolling composite: {len(composite)} symbols, {n_active} selected (score>0.1)")

    n_factors = len(factor_values)

    # 5. Compute trade parameters (ATR, entry/stop/target) for actionable signals
    trade_params = {}
    today_prices = prices[prices[date_col] == effective_trade_date].set_index(sym_col)

    if "high" in prices.columns and "low" in prices.columns and "close" in prices.columns:
        # Compute 14-day ATR per stock
        prices_sorted["tr"] = np.maximum(
            prices_sorted["high"] - prices_sorted["low"],
            np.maximum(
                abs(prices_sorted["high"] - prices_sorted.groupby(sym_col)["close"].shift(1)),
                abs(prices_sorted["low"] - prices_sorted.groupby(sym_col)["close"].shift(1)),
            ),
        )
        atr_14 = prices_sorted.groupby(sym_col)["tr"].transform(
            lambda s: s.rolling(14, min_periods=5).mean()
        )
        prices_sorted["atr_14"] = atr_14
        atr_today = prices_sorted[prices_sorted[date_col] == effective_trade_date].set_index(sym_col)["atr_14"]

        for sym, score in composite.items():
            if sym not in today_prices.index or sym not in atr_today.index:
                continue
            close_price = today_prices.loc[sym, "close"]
            atr = atr_today.loc[sym]
            if pd.isna(close_price) or pd.isna(atr) or atr <= 0:
                continue

            direction = "long" if score > 0 else "short"

            if direction == "long":
                entry = close_price  # buy at close or next open
                stop = round(entry - 2 * atr, 2)
                target = round(entry + 3 * atr, 2)
            else:
                entry = close_price
                stop = round(entry + 2 * atr, 2)
                target = round(entry - 3 * atr, 2)

            # A-share: clamp stop to ±10% limit
            if market == "cn":
                stop = max(stop, round(entry * 0.90, 2)) if direction == "long" else min(stop, round(entry * 1.10, 2))
                target = min(target, round(entry * 1.10, 2)) if direction == "short" else target

            rr = abs(target - entry) / abs(entry - stop) if abs(entry - stop) > 0.01 else 0
            trade_params[sym] = {
                "entry": round(entry, 2),
                "stop": stop,
                "target": target,
                "atr": round(atr, 2),
                "rr": round(rr, 2),
                "direction": direction,
                "hold_days": 5,
            }

        print(f"  Trade params computed for {len(trade_params)} stocks")

    # 6. Write to pipeline DB
    pipeline_con = _connect_for_write(cfg["db_path"])
    pipeline_con.execute("BEGIN")
    try:
        if market == "cn":
            pipeline_con.execute("""
                DELETE FROM analytics
                WHERE as_of = ? AND module = 'lab_factor' AND metric = 'lab_composite'
            """, [as_of])
            for sym, val in composite.items():
                detail = json.dumps({
                    "method": "rolling_best",
                    "factors": n_factors,
                    "selected_factor": selected_factor_name,
                    "selected_side": selected_side,
                    "lookback_sharpe": selected_sharpe,
                    "trade_date": effective_trade_date.date().isoformat(),
                    **trade_params.get(sym, {}),
                })
                pipeline_con.execute(cfg["insert_sql"], [sym, as_of, "lab_composite", val, detail])

        elif market == "us":
            pipeline_con.execute("""
                DELETE FROM analysis_daily
                WHERE date = ? AND module_name = 'lab_factor'
            """, [as_of])
            for sym, val in composite.items():
                detail_json = json.dumps({
                    "method": "rolling_best",
                    "factors": n_factors,
                    "selected_factor": selected_factor_name,
                    "selected_side": selected_side,
                    "lookback_sharpe": selected_sharpe,
                    "trade_date": effective_trade_date.date().isoformat(),
                    **trade_params.get(sym, {}),
                })
                pipeline_con.execute("""
                    INSERT OR REPLACE INTO analysis_daily
                        (symbol, date, module_name, trend_prob, z_score, details)
                    VALUES (?, ?, 'lab_factor', ?, 0, ?)
                """, [sym, as_of, val, detail_json])
        pipeline_con.execute("COMMIT")
    except Exception:
        pipeline_con.execute("ROLLBACK")
        raise
    finally:
        pipeline_con.close()

    print(f"  Exported {len(composite)} lab_composite values to {market} pipeline DB")
    return len(composite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["cn", "us"], required=True)
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    export(args.market, args.date)


if __name__ == "__main__":
    main()
