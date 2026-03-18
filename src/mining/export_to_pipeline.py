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
        SELECT factor_id, formula, name, composite_score
        FROM factor_registry
        WHERE market = ? AND status = 'promoted'
        ORDER BY composite_score DESC
    """, [market]).fetchall()
    lab_con.close()

    if not promoted:
        print(f"  No promoted factors for {market}, skipping")
        return 0

    print(f"  {len(promoted)} promoted factors for {market}")

    # 2. Load prices
    pipeline_con = duckdb.connect(cfg["db_path"])
    prices = pipeline_con.execute(cfg["price_sql"]).fetchdf()

    # 3. Compute each factor and combine into lab_composite
    factor_values = {}
    weights = {}
    total_weight = 0

    for factor_id, formula, name, score in promoted:
        try:
            ast = parse(formula)
            fdf = compute_factor(ast, prices, sym_col="ts_code", date_col="trade_date")
            # Filter to as_of date
            today_values = fdf[fdf["trade_date"] == as_of]
            if len(today_values) > 0:
                factor_values[factor_id] = today_values.set_index("ts_code")["factor_value"]
                weights[factor_id] = max(score, 0.01)
                total_weight += weights[factor_id]
                print(f"    {name}: {len(today_values)} values computed")
            else:
                print(f"    {name}: no values for {as_of}")
        except Exception as e:
            print(f"    {name}: compute error — {e}")

    if not factor_values:
        pipeline_con.close()
        return 0

    # 4. Compute weighted composite
    # Normalize weights
    for k in weights:
        weights[k] /= total_weight

    # Combine: weighted average of factor values
    all_symbols = set()
    for fv in factor_values.values():
        all_symbols.update(fv.index)

    composite = {}
    for sym in all_symbols:
        val = 0.0
        w_sum = 0.0
        for fid, fv in factor_values.items():
            if sym in fv.index and not np.isnan(fv[sym]):
                val += weights[fid] * fv[sym]
                w_sum += weights[fid]
        if w_sum > 0:
            composite[sym] = val / w_sum

    print(f"  lab_composite: {len(composite)} symbols")

    # 5. Write to pipeline DB
    if market == "cn":
        # CN: write to analytics table
        detail = f"factors={len(factor_values)},weights={json.dumps({k:round(v,3) for k,v in weights.items()})}"
        for sym, val in composite.items():
            pipeline_con.execute(cfg["insert_sql"], [sym, as_of, "lab_composite", val, detail])

        # Also write individual factor scores
        for fid, fv in factor_values.items():
            for sym, val in fv.items():
                if not np.isnan(val):
                    pipeline_con.execute(cfg["insert_sql"],
                                         [sym, as_of, f"lab_{fid}", val, ""])

    elif market == "us":
        # US: write to analysis_daily table
        import json as json_mod
        for sym, val in composite.items():
            detail_json = json_mod.dumps({
                "factors": len(factor_values),
                "weights": {k: round(v, 3) for k, v in weights.items()},
            })
            pipeline_con.execute("""
                INSERT OR REPLACE INTO analysis_daily
                    (symbol, date, module_name, trend_prob, z_score, details)
                VALUES (?, ?, 'lab_factor', ?, 0, ?)
            """, [sym, as_of, val, detail_json])

    pipeline_con.close()
    print(f"  Exported {len(composite)} lab_composite values to {market} pipeline DB")
    return len(composite)


import json  # noqa: E402 — needed for detail string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["cn", "us"], required=True)
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    export(args.market, args.date)


if __name__ == "__main__":
    main()
