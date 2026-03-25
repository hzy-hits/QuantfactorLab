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
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dsl.parser import parse
from src.dsl.compute import compute_factor

FACTOR_LAB_DB = "/home/ivena/coding/python/factor-lab/data/factor_lab.duckdb"

PIPELINE_CONFIGS = {
    "cn": {
        "db_path": "/home/ivena/coding/rust/quant-research-cn/data/quant_cn.duckdb",
        "price_sql": "SELECT ts_code, trade_date, open, high, low, close, vol as volume, amount FROM prices WHERE close > 0 ORDER BY ts_code, trade_date",
        "sym_col": "ts_code",
        "date_col": "trade_date",
        "insert_sql": "INSERT OR REPLACE INTO analytics (ts_code, as_of, module, metric, value, detail) VALUES (?, ?, 'lab_factor', ?, ?, ?)",
    },
    "us": {
        "db_path": "/home/ivena/coding/python/quant-research-v1/data/quant.duckdb",
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


def export(market: str, as_of: str | None = None):
    """Export promoted factors to pipeline DB."""
    cfg = PIPELINE_CONFIGS[market]
    as_of = as_of or date.today().isoformat()

    # 1. Read promoted factors from factor_lab.duckdb
    if not Path(FACTOR_LAB_DB).exists():
        print(f"  Factor Lab DB not found, skipping")
        return 0

    lab_con = duckdb.connect(FACTOR_LAB_DB, read_only=True)
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

    # 2. Load prices
    pipeline_con = duckdb.connect(cfg["db_path"])
    prices = pipeline_con.execute(cfg["price_sql"]).fetchdf()
    effective_trade_date = _resolve_effective_trade_date(prices, as_of)
    if effective_trade_date is None:
        print(f"  No trade_date <= {as_of} in pipeline DB, skipping")
        pipeline_con.close()
        return 0

    if effective_trade_date.date().isoformat() != as_of:
        print(
            f"  Using latest available trade_date {effective_trade_date.date().isoformat()} "
            f"for requested as_of {as_of}"
        )

    # 3. Compute each factor and combine into lab_composite
    factor_values = {}
    weights = {}
    total_weight = 0
    weight_source = "agent_regime" if saved_weights else "composite_score"

    for factor_id, formula, name, score, direction, ic_7d, ic_14d, ic_30d in promoted:
        try:
            ast = parse(formula)
            fdf = compute_factor(ast, prices, sym_col="ts_code", date_col="trade_date")
            # Filter to effective trade date, but write under requested report date.
            today_values = fdf[fdf["trade_date"] == effective_trade_date]
            if len(today_values) > 0:
                series = today_values.set_index("ts_code")["factor_value"]
                effective_direction = _resolve_direction(direction, ic_7d, ic_14d, ic_30d)
                if effective_direction == "short":
                    series = -series

                factor_values[factor_id] = series
                fallback_w = max(score, 0.01) if score is not None else 1.0
                weights[factor_id] = saved_weights.get(factor_id, fallback_w)
                total_weight += weights[factor_id]
                print(
                    f"    {name}: {len(today_values)} values computed "
                    f"({effective_direction}, trade_date={effective_trade_date.date().isoformat()})"
                )
            else:
                print(
                    f"    {name}: no values for requested {as_of} "
                    f"(effective trade_date={effective_trade_date.date().isoformat()})"
                )
        except Exception as e:
            print(f"    {name}: compute error — {e}")

    if not factor_values:
        pipeline_con.close()
        return 0

    # 4. Voting-based composite (intersection strategy)
    # For each factor, determine best quintile and tag stocks in it.
    # Composite score = fraction of factors that "vote" for this stock.
    # This avoids Q1/Q5 direction conflicts that plague weighted-average composites.

    n_factors = len(factor_values)
    all_symbols = set()
    for fv in factor_values.values():
        all_symbols.update(fv.index)

    # Build per-factor quintile assignments
    factor_best_q = {}
    factor_quintiles = {}
    for fid, fv in factor_values.items():
        clean = fv.dropna()
        if len(clean) < 25:
            continue
        try:
            quintiles = pd.qcut(clean, 5, labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        factor_quintiles[fid] = quintiles

        # Best quintile = the one with historically highest factor values
        # Since direction is already resolved (short factors are negated),
        # Q5 (highest value) should be the "good" side after direction flip.
        # But empirically Q1 often has the alpha. Use factor registry direction
        # to decide: if factor was already negated, Q5 is correct.
        # Simplest: just pick Q5 (highest factor value) since we already
        # flipped short factors with -series above.
        factor_best_q[fid] = 5

    # Count votes: how many factors place each stock in their best quintile
    votes = {}
    for sym in all_symbols:
        count = 0
        total = 0
        for fid, quintiles in factor_quintiles.items():
            if sym in quintiles.index:
                total += 1
                if quintiles[sym] == factor_best_q[fid]:
                    count += 1
        if total > 0:
            # Score = vote fraction, scaled to [-1, 1]
            # 0 votes → -1 (short), all votes → +1 (strong long)
            votes[sym] = 2.0 * (count / total) - 1.0

    # Also compute worst quintile votes for short signal
    for sym in all_symbols:
        if sym in votes:
            continue
        votes[sym] = -1.0  # no data = no signal

    # 4b. Mahalanobis filter: penalize stocks that deviate from factor-space norm
    # Stocks the factor model can't explain well are less predictable.
    # Compute PCA on factor matrix, then Mahalanobis distance per stock.
    try:
        # Build factor matrix for common symbols
        common_syms = sorted(set.intersection(*[set(fv.index) for fv in factor_values.values()]))
        if len(common_syms) >= 100:
            mat = np.column_stack([factor_values[fid].loc[common_syms].values for fid in factor_quintiles])
            mask = ~np.any(np.isnan(mat), axis=1)
            clean_syms = [common_syms[i] for i in range(len(common_syms)) if mask[i]]
            mat_clean = mat[mask]

            if len(mat_clean) >= 100:
                mat_std = (mat_clean - mat_clean.mean(axis=0)) / (mat_clean.std(axis=0) + 1e-12)
                U, S, Vt = np.linalg.svd(mat_std, full_matrices=False)
                pc_scores = U * S
                eigenvalues = S ** 2 / len(mat_std)
                weighted = pc_scores / (np.sqrt(eigenvalues) + 1e-6)
                mahal = np.sqrt(np.sum(weighted ** 2, axis=1))

                # Penalize top 20% outliers: multiply their vote score by 0.5
                mahal_threshold = np.percentile(mahal, 80)
                outlier_set = {clean_syms[i] for i in range(len(clean_syms)) if mahal[i] > mahal_threshold}
                n_penalized = 0
                for sym in outlier_set:
                    if sym in votes:
                        votes[sym] *= 0.5
                        n_penalized += 1
                print(f"  Mahalanobis filter: penalized {n_penalized} outlier stocks (top 20% deviation)")
    except Exception as e:
        print(f"  Mahalanobis filter skipped: {e}")

    composite = votes
    n_active = sum(1 for v in composite.values() if abs(v) > 0.1)

    print(f"  Voting composite: {len(composite)} symbols, {n_active} active (|score|>0.1)")
    print(f"  Score distribution: [{min(composite.values()):.2f}, {max(composite.values()):.2f}]")
    vote_counts = {}
    for v in composite.values():
        bucket = round((v + 1) / 2 * n_factors)
        vote_counts[bucket] = vote_counts.get(bucket, 0) + 1
    for k in sorted(vote_counts.keys(), reverse=True)[:5]:
        print(f"    {k}/{n_factors} votes: {vote_counts[k]} stocks")

    # 5. Write to pipeline DB
    detail_str = (
        f"method=voting,factors={n_factors},"
        f"effective_trade_date={effective_trade_date.date().isoformat()}"
    )

    if market == "cn":
        # CN: write to analytics table
        for sym, val in composite.items():
            pipeline_con.execute(cfg["insert_sql"], [sym, as_of, "lab_composite", val, detail_str])

    elif market == "us":
        # US: write to analysis_daily table
        import json as json_mod
        detail_json = json_mod.dumps({
            "method": "voting",
            "factors": n_factors,
            "effective_trade_date": effective_trade_date.date().isoformat(),
        })
        for sym, val in composite.items():
            pipeline_con.execute("""
                INSERT OR REPLACE INTO analysis_daily
                    (symbol, date, module_name, trend_prob, z_score, details)
                VALUES (?, ?, 'lab_factor', ?, 0, ?)
            """, [sym, as_of, val, detail_json])

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
