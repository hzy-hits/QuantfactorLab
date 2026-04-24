#!/usr/bin/env python3
"""SigReg factor diagnostics — run daily after health check.

Reports: diversity score, redundant factors, IC health, PCA structure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
import duckdb
from scipy.stats import spearmanr

from src.dsl.parser import parse
from src.dsl.compute import compute_factor
from src.market_data import load_forward_returns, load_market_prices
from src.evaluate.sigreg import (
    factor_diversity_score,
    multi_collinearity_check,
    ic_health_test,
)

FACTOR_LAB_DB = "data/factor_lab.duckdb"


def run_diagnostics(market: str):
    prices = load_market_prices(market)
    fwd = load_forward_returns(market, prices=prices)

    sym_col = "ts_code" if market == "cn" else "symbol"
    date_col = "trade_date" if market == "cn" else "date"
    latest = prices[date_col].max()

    con = duckdb.connect(FACTOR_LAB_DB, read_only=True)
    promoted = con.execute(
        "SELECT name, formula FROM factor_registry WHERE market=? AND status='promoted'",
        [market],
    ).fetchdf()
    con.close()

    if promoted.empty:
        print(f"  No promoted factors for {market}")
        return

    # Compute factor values
    factor_values = {}
    factor_ic_series = {}
    for _, row in promoted.iterrows():
        try:
            ast = parse(row["formula"])
            vals = compute_factor(ast, prices, sym_col=sym_col, date_col=date_col)
            today = vals[vals[date_col] == latest].set_index(sym_col)["factor_value"]
            factor_values[row["name"]] = today

            merged = vals.merge(fwd, on=[sym_col, date_col]).dropna(
                subset=["factor_value", "fwd_5d"]
            )
            ics = []
            for dt, g in merged.groupby(date_col):
                v = g.dropna(subset=["factor_value", "fwd_5d"])
                if len(v) >= 30:
                    rho, _ = spearmanr(v["factor_value"], v["fwd_5d"])
                    ics.append(rho)
            factor_ic_series[row["name"]] = ics
        except Exception:
            pass

    if not factor_values:
        print(f"  No computable factors for {market}")
        return

    print(f"\n  [{market.upper()}] {len(factor_values)} factors, {latest.date()}")

    # 1. Diversity
    div = factor_diversity_score(factor_values)
    status = "🟢" if div["diversity_score"] > 0.7 else "🟡" if div["diversity_score"] > 0.4 else "🔴"
    print(f"  {status} Diversity: {div['diversity_score']} (effective: {div['n_effective']}/{div['n_total']})")
    if div["cluster_warning"]:
        print(f"     ⚠️ Factor clustering detected — new factors should target unexplored dimensions")

    # 2. Multi-collinearity
    redundant = []
    for name, fv in factor_values.items():
        others = {k: v for k, v in factor_values.items() if k != name}
        mc = multi_collinearity_check(fv, others)
        if mc["is_redundant"]:
            redundant.append((name, mc["r_squared"], mc["top_contributors"]))

    if redundant:
        print(f"  🔴 Redundant factors ({len(redundant)}):")
        for name, r2, top in redundant:
            top_str = ", ".join(f"{n}" for n, _ in top[:2])
            print(f"     {name}: R²={r2:.2f} (explained by {top_str})")
    else:
        print(f"  🟢 No redundant factors (all R²<0.85)")

    # 3. Signal Analysis (Wavelet + FFT + Phase Space)
    from src.evaluate.signal_analysis import full_diagnosis

    print(f"\n  --- Signal Analysis (小波/FFT/相空间) ---")
    for name, ics in factor_ic_series.items():
        if len(ics) < 80:
            continue
        diag = full_diagnosis(np.array(ics))
        action_icon = {"EXIT": "🔴", "REDUCE": "🟡", "HOLD": "🟢"}.get(diag["action"], "⚪")
        print(f"  {action_icon} {name}: {diag['action']} — {diag['reason']}")
        print(f"       IC半衰期={diag['ic_halflife']}天, 主周期={diag['dominant_period']:.0f}天, 相维度={diag['phase_dimension']:.1f}, 建议持仓={diag['suggested_hold_days']}天")


def main():
    parser = argparse.ArgumentParser(description="Run SigReg diagnostics for one or both markets.")
    parser.add_argument(
        "--market",
        choices=["cn", "us", "all"],
        default="all",
        help="Which market to diagnose.",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  SigReg Factor Diagnostics")
    print("=" * 50)

    markets = ["cn", "us"] if args.market == "all" else [args.market]
    for market in markets:
        try:
            run_diagnostics(market)
        except Exception as e:
            print(f"  [{market.upper()}] Error: {e}")

    print()


if __name__ == "__main__":
    main()
