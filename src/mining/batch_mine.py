#!/usr/bin/env python3
"""
Batch factor mining — systematically generate and evaluate factor combinations.

Instead of hand-writing 500 formulas, programmatically combine:
  operators × features × windows → thousands of candidates → evaluate → rank

Usage:
    python -m src.mining.batch_mine --market cn --max-factors 500
    python -m src.mining.batch_mine --market us --max-factors 500 --output reports/batch_us.md
"""
import sys
import argparse
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dsl.parser import parse, DSLParseError
from src.dsl.compute import compute_factor
from src.evaluate.forward_returns import compute_forward_returns
from src.evaluate.ic import compute_ic_series, ic_summary
from src.evaluate.quintile import compute_quintile_returns
from src.paths import QUANT_CN_DB, QUANT_US_DB

# ── Factor formula templates ──────────────────────────────────────────────────

# Depth 1: rank(transform(feature, window))
DEPTH1_TEMPLATES = [
    ("rank({f})", "raw cross-sectional rank"),
    ("rank(-{f})", "inverted rank"),
    ("rank(delta({f}, {w}))", "N-day change"),
    ("rank(-delta({f}, {w}))", "N-day reversal"),
    ("rank(pct_change({f}, {w}))", "N-day pct change"),
    ("rank(-pct_change({f}, {w}))", "N-day pct reversal"),
    ("rank(ts_mean({f}, {w}))", "N-day moving average"),
    ("rank(ts_std({f}, {w}))", "N-day volatility"),
    ("rank(-ts_std({f}, {w}))", "N-day low volatility"),
    ("rank(ts_rank({f}, {w}))", "N-day time-series rank"),
    ("rank(ts_max({f}, {w}) / {f} - 1)", "distance from N-day high"),
    ("rank({f} / ts_min({f}, {w}) - 1)", "distance from N-day low"),
]

# Depth 2: rank(A) * rank(B) — interaction of two signals
DEPTH2_PAIRS = [
    # (template_A, template_B, hypothesis)
    ("rank(-ret_5d)", "rank(-volume / ts_mean(volume, {w}))", "reversal + shrinking vol"),
    ("rank(-ret_5d)", "rank(volume / ts_mean(volume, {w}))", "reversal + volume spike"),
    ("rank(ret_5d)", "rank(volume / ts_mean(volume, {w}))", "momentum + volume confirm"),
    ("rank(-ts_corr(close, volume, {w}))", "rank(-ret_5d)", "VP divergence + decline"),
    ("rank(-ts_corr(close, volume, {w}))", "rank(ret_5d)", "VP divergence + rally"),
    ("rank(-ts_std(volume, {w}) / ts_mean(volume, {w}))", "rank(volume / ts_mean(volume, 5))", "shrink then surge"),
    ("rank(-ts_std(ret_1d, {w}))", "rank(abs(ret_1d))", "quiet then big move"),
    ("rank((close - low) / (high - low + 0.01))", "rank(-ret_5d)", "hammer + decline"),
    ("rank(-(high - close) / (high - low + 0.01))", "rank(ret_5d)", "no upper wick + uptrend"),
    ("rank(-pct_change(close, {w}))", "rank(-pct_change(close, 20))", "double reversal"),
    ("rank(ts_min(low, {w}) / close - 1)", "rank(ret_1d)", "near low + today up"),
    ("rank(close / ts_max(high, {w}))", "rank(volume / ts_mean(volume, {w}))", "breakout + volume"),
    ("rank(delta(volume, 5))", "rank(-delta(close, 5))", "volume up + price down"),
    ("rank(ret_20d)", "rank(-ts_std(ret_1d, 20))", "smooth uptrend = quality"),
    ("rank(-ret_20d)", "rank(-ts_std(ret_1d, 20))", "smooth downtrend reversal"),
]

# Features and windows for template substitution
PRICE_FEATURES = ["close", "high", "low", "open"]
VOLUME_FEATURES = ["volume"]
RETURN_FEATURES = ["ret_1d", "ret_5d", "ret_20d"]
RANGE_FEATURES = ["high - low"]  # intraday range
WINDOWS_SHORT = [3, 5, 10]
WINDOWS_MEDIUM = [14, 20, 30]
WINDOWS_LONG = [40, 60, 120]
ALL_WINDOWS = [3, 5, 10, 20, 60]  # 5 windows instead of 4


def generate_factor_formulas(max_factors: int = 500) -> list[tuple[str, str, str]]:
    """Generate factor formulas systematically. Returns [(name, formula, hypothesis)]."""
    formulas = []

    # Depth 1: single transforms
    for tmpl, hyp in DEPTH1_TEMPLATES:
        for f in PRICE_FEATURES + VOLUME_FEATURES + RETURN_FEATURES:
            for w in ALL_WINDOWS:
                if "{w}" in tmpl and "{f}" in tmpl:
                    formula = tmpl.format(f=f, w=w)
                elif "{f}" in tmpl:
                    formula = tmpl.format(f=f)
                else:
                    continue
                name = f"d1_{f}_{w}_{len(formulas)}"
                formulas.append((name, formula, f"{hyp} ({f}, {w}d)"))

    # Depth 1 without window
    for f in PRICE_FEATURES + VOLUME_FEATURES + RETURN_FEATURES:
        formulas.append((f"raw_{f}", f"rank({f})", f"raw rank of {f}"))
        formulas.append((f"raw_neg_{f}", f"rank(-{f})", f"inverted rank of {f}"))

    # Depth 2: interactions
    for tmpl_a, tmpl_b, hyp in DEPTH2_PAIRS:
        for w in ALL_WINDOWS:
            formula_a = tmpl_a.format(w=w) if "{w}" in tmpl_a else tmpl_a
            formula_b = tmpl_b.format(w=w) if "{w}" in tmpl_b else tmpl_b
            formula = f"{formula_a} * {formula_b}"
            name = f"d2_{w}_{len(formulas)}"
            formulas.append((name, formula, f"{hyp} ({w}d)"))

    # Volume-price correlation variants
    for w in [3, 5, 10, 20, 40, 60]:
        formulas.append((f"vpcorr_{w}", f"rank(-ts_corr(close, volume, {w}))", f"VP divergence {w}d"))
        formulas.append((f"vpcorr_neg_{w}", f"rank(ts_corr(close, volume, {w}))", f"VP alignment {w}d"))

    # Volatility ratios
    for w1, w2 in [(5, 20), (5, 60), (10, 20), (10, 60), (20, 60)]:
        formulas.append((f"volratio_{w1}_{w2}", f"rank(-ts_std(ret_1d, {w1}) / (ts_std(ret_1d, {w2}) + 0.001))", f"vol compression {w1}v{w2}"))
        formulas.append((f"volratio_inv_{w1}_{w2}", f"rank(ts_std(ret_1d, {w1}) / (ts_std(ret_1d, {w2}) + 0.001))", f"vol expansion {w1}v{w2}"))

    # RSV variants
    for w in [5, 10, 20, 40, 60]:
        formulas.append((f"rsv_{w}", f"rank((close - ts_min(low, {w})) / (ts_max(high, {w}) - ts_min(low, {w}) + 0.01))", f"RSV {w}d"))
        formulas.append((f"rsv_inv_{w}", f"rank(-((close - ts_min(low, {w})) / (ts_max(high, {w}) - ts_min(low, {w}) + 0.01)))", f"inverse RSV {w}d"))

    # K-bar patterns with different contexts
    for w in [5, 10, 20]:
        formulas.append((f"hammer_{w}", f"rank((close - low) / (high - low + 0.01)) * rank(-pct_change(close, {w}))", f"hammer after {w}d decline"))
        formulas.append((f"shooting_{w}", f"rank((high - close) / (high - low + 0.01)) * rank(pct_change(close, {w}))", f"shooting star after {w}d rally"))

    # === Additional templates to reach 500+ ===

    # K-bar body/shadow standalone
    for w in ALL_WINDOWS:
        formulas.append((f"kbody_{w}", f"rank(ts_mean((close - open) / (high - low + 0.01), {w}))", f"avg K-bar body {w}d"))
        formulas.append((f"kshadow_{w}", f"rank(ts_mean((high - close) / (high - low + 0.01), {w}))", f"avg upper shadow {w}d"))

    # Higher lows / lower highs (trend structure)
    for w1, w2 in [(5, 20), (5, 60), (10, 20), (10, 60)]:
        formulas.append((f"higher_lows_{w1}_{w2}", f"rank(ts_min(low, {w1}) / ts_min(low, {w2}) - 1)", f"higher lows {w1}v{w2}"))
        formulas.append((f"lower_highs_{w1}_{w2}", f"rank(-(ts_max(high, {w1}) / ts_max(high, {w2}) - 1))", f"lower highs {w1}v{w2}"))

    # Range contraction/expansion
    for w1, w2 in [(3, 20), (5, 20), (5, 60), (10, 60)]:
        formulas.append((f"range_contract_{w1}_{w2}", f"rank(-(ts_max(high, {w1}) - ts_min(low, {w1})) / (ts_max(high, {w2}) - ts_min(low, {w2}) + 0.01))", f"range contraction {w1}v{w2}"))

    # Return momentum with different windows
    for w in [3, 10, 14, 30, 40]:
        formulas.append((f"mom_{w}", f"rank(pct_change(close, {w}))", f"momentum {w}d"))
        formulas.append((f"rev_{w}", f"rank(-pct_change(close, {w}))", f"reversal {w}d"))

    # Acceleration (momentum of momentum)
    for w in [5, 10, 20]:
        formulas.append((f"accel_{w}", f"rank(ret_5d - shift(ret_5d, {w}))", f"momentum accel vs {w}d ago"))

    # Volume trend ratios (additional windows)
    for w1, w2 in [(3, 10), (3, 20), (5, 20), (5, 60), (10, 60)]:
        formulas.append((f"voltrd_{w1}_{w2}", f"rank(ts_mean(volume, {w1}) / ts_mean(volume, {w2}))", f"vol trend {w1}v{w2}"))

    # Intraday range patterns
    for w in [5, 10, 20]:
        formulas.append((f"range_norm_{w}", f"rank(ts_mean(high - low, {w}) / close)", f"normalized range {w}d"))
        formulas.append((f"range_vol_{w}", f"rank(ts_std(high - low, {w}) / ts_mean(high - low, {w}))", f"range volatility {w}d"))

    # Distance from high/low with more windows
    for w in [5, 10, 14, 30, 40, 60]:
        formulas.append((f"dist_high_{w}", f"rank((close - ts_max(high, {w})) / ts_max(high, {w}))", f"dist from {w}d high"))
        formulas.append((f"dist_low_{w}", f"rank((close - ts_min(low, {w})) / ts_min(low, {w}))", f"dist from {w}d low"))

    # Triple interactions (depth 3)
    for w in [10, 20]:
        formulas.append((f"tri_rev_vol_squeeze_{w}", f"rank(-ret_5d) * rank(-volume / ts_mean(volume, {w})) * rank(-ts_std(ret_1d, {w}) / (ts_std(ret_1d, 60) + 0.001))", f"reversal+shrink+squeeze {w}d"))
        formulas.append((f"tri_mom_vol_break_{w}", f"rank(ret_5d) * rank(volume / ts_mean(volume, {w})) * rank(close / ts_max(high, {w}))", f"momentum+volume+breakout {w}d"))

    # Deduplicate and limit
    seen = set()
    unique = []
    for name, formula, hyp in formulas:
        if formula not in seen:
            seen.add(formula)
            unique.append((name, formula, hyp))

    # Shuffle and limit (random subset if too many)
    np.random.seed(42)
    if len(unique) > max_factors:
        indices = np.random.choice(len(unique), max_factors, replace=False)
        unique = [unique[i] for i in sorted(indices)]

    return unique


CONFIGS = {
    "cn": {
        "db_path": str(QUANT_CN_DB),
        "table": "prices", "sym_col": "ts_code", "date_col": "trade_date",
        "close_col": "close", "vol_col": "vol",
        "universe_top_n": 2000,
        "sql": """
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
    },
    "us": {
        "db_path": str(QUANT_US_DB),
        "table": "prices_daily", "sym_col": "symbol", "date_col": "date",
        "close_col": "adj_close", "vol_col": "volume",
        "sql": "SELECT symbol as ts_code, date as trade_date, open, high, low, adj_close as close, volume FROM prices_daily WHERE adj_close > 0 ORDER BY symbol, date",
    },
}


def run_batch(market: str, max_factors: int = 500, output_path: str | None = None):
    cfg = CONFIGS[market]
    print(f"=== Batch Factor Mining: {market.upper()} ({max_factors} factors) ===\n")

    # Load data
    import duckdb
    con = duckdb.connect(cfg["db_path"], read_only=True)
    prices = con.execute(cfg["sql"]).fetchdf()
    con.close()

    # Universe filter: keep only top N stocks by market_cap each day
    top_n = cfg.get("universe_top_n")
    if top_n and "market_cap" in prices.columns:
        prices["_mcap_rank"] = prices.groupby("trade_date")["market_cap"].rank(
            ascending=False, method="first", na_option="bottom"
        )
        prices = prices[prices["_mcap_rank"] <= top_n].drop(columns=["_mcap_rank"]).reset_index(drop=True)
        print(f"  Universe filter: top {top_n} by market_cap")

    fwd = compute_forward_returns(cfg["db_path"], cfg["table"], cfg["date_col"], cfg["close_col"],
                                  cfg["sym_col"] if market == "cn" else "symbol")
    if market == "us":
        fwd = fwd.rename(columns={"symbol": "ts_code", "date": "trade_date"})

    print(f"Data: {len(prices)} rows, {prices['ts_code'].nunique()} symbols\n")

    # Generate formulas
    formulas = generate_factor_formulas(max_factors)
    print(f"Generated {len(formulas)} unique factor formulas\n")

    # Evaluate each
    results = []
    errors = 0
    for i, (name, formula, hypothesis) in enumerate(formulas):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(formulas)} ({len(results)} valid, {errors} errors)")

        try:
            ast = parse(formula)
            factor_df = compute_factor(ast, prices, sym_col="ts_code", date_col="trade_date")
            merged = factor_df.merge(
                fwd[["ts_code", "trade_date", "fwd_5d"]],
                on=["ts_code", "trade_date"], how="inner"
            ).dropna(subset=["fwd_5d", "factor_value"])

            if len(merged) < 500:
                continue

            ic_stats = ic_summary(compute_ic_series(
                merged["factor_value"], merged["fwd_5d"], merged["trade_date"]
            ))
            q = compute_quintile_returns(
                merged["factor_value"], merged["fwd_5d"], merged["trade_date"]
            )

            results.append({
                "name": name,
                "formula": formula,
                "hypothesis": hypothesis,
                "ic": ic_stats["ic_mean"],
                "ic_ir": ic_stats["ic_ir"],
                "ic_pos_pct": ic_stats["ic_positive_pct"],
                "q5_q1": q["long_short_pct"],
                "mono": q["monotonicity"],
                "n_days": ic_stats["n_days"],
            })
        except (DSLParseError, Exception):
            errors += 1

    print(f"\nDone: {len(results)} valid factors, {errors} errors\n")

    # Sort by |IC| * sign(IC * mono) — reward factors where IC and monotonicity agree
    for r in results:
        ic_mono_agree = 1 if (r["ic"] * r["mono"] >= 0) else -1
        r["score"] = abs(r["ic"]) * abs(r["ic_ir"]) * ic_mono_agree

    results.sort(key=lambda r: r["score"], reverse=True)

    # Report
    lines = [
        f"# Batch Factor Mining — {market.upper()}",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Factors tested: {len(formulas)}",
        f"Valid results: {len(results)}",
        f"Errors: {errors}",
        "",
        "## Top 30 Factors",
        "",
        "| # | IC | IC_IR | Q5-Q1% | Mono | Formula |",
        "|---|-----|-------|--------|------|---------|",
    ]

    for i, r in enumerate(results[:30]):
        lines.append(
            f"| {i+1} | {r['ic']:.4f} | {r['ic_ir']:.3f} | {r['q5_q1']:.3f} | {r['mono']:.2f} | `{r['formula'][:60]}` |"
        )

    lines.append("")
    lines.append("## Gate Analysis (top 30)")
    lines.append("")

    gate_ic = 0.01 if market == "cn" else 0.02
    for i, r in enumerate(results[:30]):
        ic_ok = abs(r["ic"]) >= gate_ic
        ir_ok = abs(r["ic_ir"]) >= 0.2
        mono_ok = abs(r["mono"]) >= 0.6
        gates = f"{'✅' if ic_ok else '❌'}IC {'✅' if ir_ok else '❌'}IR {'✅' if mono_ok else '❌'}Mono"
        lines.append(f"  {i+1}. **{r['name']}**: {gates} — {r['hypothesis']}")

    lines.append("")
    lines.append("## Distribution")
    lines.append(f"  IC > 0.02: {sum(1 for r in results if abs(r['ic']) > 0.02)}")
    lines.append(f"  IC > 0.01: {sum(1 for r in results if abs(r['ic']) > 0.01)}")
    lines.append(f"  IC_IR > 0.3: {sum(1 for r in results if abs(r['ic_ir']) > 0.3)}")
    lines.append(f"  Monotonicity > 0.7: {sum(1 for r in results if abs(r['mono']) > 0.7)}")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report)
        print(f"Report: {output_path}")
    else:
        print(report)

    # Print top 10 to console
    print("\n=== TOP 10 ===")
    print(f"{'#':>3} {'IC':>8} {'IR':>8} {'Q5-Q1':>8} {'Mono':>6} Formula")
    for i, r in enumerate(results[:10]):
        print(f"{i+1:>3} {r['ic']:>8.4f} {r['ic_ir']:>8.3f} {r['q5_q1']:>8.3f} {r['mono']:>6.2f} {r['formula'][:70]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["cn", "us"], required=True)
    parser.add_argument("--max-factors", type=int, default=500)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_batch(args.market, args.max_factors, args.output)


if __name__ == "__main__":
    main()
