#!/usr/bin/env python3
"""
Factor evaluation MVP — run against CN or US pipeline data.

Usage:
    python src/evaluate/run_evaluation.py --market cn
    python src/evaluate/run_evaluation.py --market us
    python src/evaluate/run_evaluation.py --market cn --output reports/eval_cn.md
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluate.forward_returns import compute_forward_returns
from src.evaluate.factors import compute_all_factors
from src.evaluate.ic import compute_ic_series, ic_summary, ic_by_regime
from src.evaluate.quintile import compute_quintile_returns
from src.evaluate.correlation import factor_correlation_matrix, find_redundant_pairs
from src.evaluate.turnover import compute_turnover
from src.evaluate.rolling_ic import compute_rolling_ic, compute_ic_trend
from src.paths import QUANT_CN_DB, QUANT_US_DB


# Market configs
CONFIGS = {
    "cn": {
        "db_path": str(QUANT_CN_DB),
        "table": "prices",
        "sym_col": "ts_code",
        "date_col": "trade_date",
        "close_col": "close",
        "vol_col": "vol",
        "label": "A-Share",
    },
    "us": {
        "db_path": str(QUANT_US_DB),
        "table": "prices_daily",
        "sym_col": "symbol",
        "date_col": "date",
        "close_col": "adj_close",
        "vol_col": "volume",
        "label": "US Equity",
    },
}

FACTORS_TO_EVALUATE = [
    "rsi_14",
    "bb_position",
    "ma20_dist",
    "reversion_score",
    "volume_ratio",
    "squeeze_ratio",
    "ret_5d",
]


def run(market: str, output_path: str | None = None):
    cfg = CONFIGS[market]
    print(f"=== Factor Evaluation: {cfg['label']} ===\n")

    # 1. Load prices
    print("Loading prices...")
    prices_df = compute_forward_returns(
        cfg["db_path"], cfg["table"], cfg["date_col"], cfg["close_col"], cfg["sym_col"]
    )
    print(f"  {len(prices_df)} rows, {prices_df[cfg['sym_col']].nunique()} symbols")
    print(f"  Date range: {prices_df[cfg['date_col']].min()} ~ {prices_df[cfg['date_col']].max()}")

    # 2. Compute factors from raw prices
    print("\nComputing factors...")

    # Need raw prices for factor computation
    import duckdb
    con = duckdb.connect(cfg["db_path"], read_only=True)
    cols = f"{cfg['sym_col']}, {cfg['date_col']}, {cfg['close_col']}"
    if cfg["vol_col"]:
        cols += f", {cfg['vol_col']}"
    raw_prices = con.execute(
        f"SELECT {cols} FROM {cfg['table']} WHERE {cfg['close_col']} > 0 ORDER BY {cfg['sym_col']}, {cfg['date_col']}"
    ).fetchdf()
    con.close()

    factor_df = compute_all_factors(
        raw_prices, cfg["sym_col"], cfg["date_col"], cfg["close_col"], cfg["vol_col"]
    )
    print(f"  {len(factor_df)} factor rows, {factor_df[cfg['sym_col']].nunique()} symbols")

    # 3. Merge factors with forward returns
    merged = factor_df.merge(
        prices_df[[cfg["sym_col"], cfg["date_col"], "fwd_5d"]],
        on=[cfg["sym_col"], cfg["date_col"]],
        how="inner",
    )
    merged = merged.dropna(subset=["fwd_5d"])
    print(f"  {len(merged)} merged rows with fwd_5d returns")

    # 4. Evaluate each factor
    print("\n" + "=" * 60)
    report_lines = []
    report_lines.append(f"# Factor Evaluation Report — {cfg['label']}")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"Data: {prices_df[cfg['date_col']].min()} ~ {prices_df[cfg['date_col']].max()}")
    report_lines.append(f"Symbols: {merged[cfg['sym_col']].nunique()}")
    report_lines.append(f"Trading days with factors: {merged[cfg['date_col']].nunique()}")
    report_lines.append("")

    # Summary table
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append("| Factor | IC | IC_IR | IC>0% | Q5-Q1 | Mono | Turnover/d | IC Trend | Status |")
    report_lines.append("|--------|-----|-------|-------|-------|------|------------|----------|--------|")

    factor_results = {}
    for factor_name in FACTORS_TO_EVALUATE:
        if factor_name not in merged.columns:
            continue

        valid = merged.dropna(subset=[factor_name])
        if len(valid) < 100:
            print(f"  {factor_name}: insufficient data ({len(valid)} rows), skipping")
            continue

        print(f"\n--- {factor_name} ---")

        # IC
        ic_series = compute_ic_series(
            valid[factor_name], valid["fwd_5d"], valid[cfg["date_col"]]
        )
        ic_stats = ic_summary(ic_series)
        print(f"  IC={ic_stats['ic_mean']}, IC_IR={ic_stats['ic_ir']}, "
              f"IC>0={ic_stats['ic_positive_pct']}%, n_days={ic_stats['n_days']}")

        # IC by regime
        regime_ic = ic_by_regime(
            valid[factor_name], valid["fwd_5d"], valid[cfg["date_col"]], valid["regime"]
        )
        for r, stats in regime_ic.items():
            if stats["ic_mean"] is not None:
                print(f"    {r}: IC={stats['ic_mean']}, IC_IR={stats['ic_ir']}, n={stats['n_obs']}")

        # Quintiles
        q = compute_quintile_returns(
            valid[factor_name], valid["fwd_5d"], valid[cfg["date_col"]]
        )
        print(f"  Quintiles: {q['quintile_returns']}")
        print(f"  Long-Short: {q['long_short_pct']}%, Monotonicity: {q['monotonicity']}")

        # Turnover
        turnover = compute_turnover(
            valid[factor_name], valid[cfg["date_col"]], valid[cfg["sym_col"]]
        )
        print(f"  Turnover: daily={turnover['avg_daily']}, monthly={turnover['avg_monthly']}, "
              f"n_days={turnover['n_days']}")

        # Rolling IC & IC trend
        rolling_ic_df = compute_rolling_ic(ic_series)
        ic_trend = compute_ic_trend(ic_series)
        print(f"  IC trend: slope/yr={ic_trend['slope_per_year']}, R2={ic_trend['r_squared']}, "
              f"verdict={ic_trend['verdict']}")

        # Status
        ic_ok = ic_stats["ic_mean"] is not None and abs(ic_stats["ic_mean"]) > 0.02
        icir_ok = ic_stats["ic_ir"] is not None and abs(ic_stats["ic_ir"]) > 0.3
        mono_ok = abs(q["monotonicity"]) > 0.7
        status = "✅" if (ic_ok and icir_ok) else "⚠️" if (ic_ok or icir_ok or mono_ok) else "❌"

        report_lines.append(
            f"| {factor_name} | {ic_stats['ic_mean']} | {ic_stats['ic_ir']} | "
            f"{ic_stats['ic_positive_pct']}% | {q['long_short_pct']}% | "
            f"{q['monotonicity']} | {turnover['avg_daily']} | "
            f"{ic_trend['verdict']} | {status} |"
        )

        factor_results[factor_name] = {
            "ic": ic_stats,
            "regime_ic": regime_ic,
            "quintile": q,
            "turnover": turnover,
            "rolling_ic": rolling_ic_df,
            "ic_trend": ic_trend,
            "status": status,
        }

    # 5. Correlation matrix
    print("\n--- Factor Correlations ---")
    available_factors = [f for f in FACTORS_TO_EVALUATE if f in merged.columns]
    corr_df = factor_correlation_matrix(merged, available_factors, cfg["date_col"])
    print(corr_df.to_string())

    redundant = find_redundant_pairs(corr_df)
    if redundant:
        print(f"\n  Redundant pairs (corr > 0.7):")
        for a, b, c in redundant:
            print(f"    {a} ~ {b}: {c}")

    # 6. Write report
    report_lines.append("")
    report_lines.append("## Regime-Conditional IC")
    report_lines.append("")
    report_lines.append("| Factor | Trending | Mean-Rev | Noisy |")
    report_lines.append("|--------|----------|----------|-------|")
    for factor_name, res in factor_results.items():
        ric = res["regime_ic"]
        t = ric.get("trending", {}).get("ic_mean", "N/A")
        m = ric.get("mean_reverting", {}).get("ic_mean", "N/A")
        n = ric.get("noisy", {}).get("ic_mean", "N/A")
        report_lines.append(f"| {factor_name} | {t} | {m} | {n} |")

    report_lines.append("")
    report_lines.append("## Quintile Returns (5D fwd, %)")
    report_lines.append("")
    report_lines.append("| Factor | Q1 | Q2 | Q3 | Q4 | Q5 | Q5-Q1 | Mono |")
    report_lines.append("|--------|-----|-----|-----|-----|-----|-------|------|")
    for factor_name, res in factor_results.items():
        q = res["quintile"]
        qr = q["quintile_returns"]
        if len(qr) == 5:
            report_lines.append(
                f"| {factor_name} | {qr[0]} | {qr[1]} | {qr[2]} | {qr[3]} | {qr[4]} | "
                f"{q['long_short_pct']} | {q['monotonicity']} |"
            )

    report_lines.append("")
    report_lines.append("## Factor Correlation Matrix")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append(corr_df.to_string())
    report_lines.append("```")

    if redundant:
        report_lines.append("")
        report_lines.append("**Redundant pairs (corr > 0.7):**")
        for a, b, c in redundant:
            report_lines.append(f"- {a} ~ {b}: {c}")

    report_lines.append("")
    report_lines.append("## Turnover (Top 20%)")
    report_lines.append("")
    report_lines.append("| Factor | Avg Daily | Avg Monthly | Days |")
    report_lines.append("|--------|-----------|-------------|------|")
    for factor_name, res in factor_results.items():
        t = res["turnover"]
        report_lines.append(
            f"| {factor_name} | {t['avg_daily']} | {t['avg_monthly']} | {t['n_days']} |"
        )

    report_lines.append("")
    report_lines.append("## IC Trend")
    report_lines.append("")
    report_lines.append("| Factor | Slope/yr | R2 | Verdict |")
    report_lines.append("|--------|----------|----|---------|")
    for factor_name, res in factor_results.items():
        trend = res["ic_trend"]
        report_lines.append(
            f"| {factor_name} | {trend['slope_per_year']} | {trend['r_squared']} | "
            f"{trend['verdict']} |"
        )

    report_lines.append("")
    report_lines.append("---")
    report_lines.append(f"*Generated by Factor Lab evaluation framework.*")

    report_text = "\n".join(report_lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report_text)
        print(f"\nReport written to {output_path}")
    else:
        print("\n" + report_text)

    return factor_results


def main():
    parser = argparse.ArgumentParser(description="Factor evaluation")
    parser.add_argument("--market", choices=["cn", "us"], required=True)
    parser.add_argument("--output", default=None, help="Output markdown path")
    args = parser.parse_args()
    run(args.market, args.output)


if __name__ == "__main__":
    main()
