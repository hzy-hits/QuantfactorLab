#!/usr/bin/env python3
"""Run rolling best-factor strategy with SigReg exit.

Usage:
    python scripts/run_strategy.py --market cn                    # backtest CN
    python scripts/run_strategy.py --market us                    # backtest US
    python scripts/run_strategy.py --market cn --lookback 40 --hold 20
    python scripts/run_strategy.py --market cn --today            # today's picks
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pickle
import time

import duckdb
import numpy as np
import pandas as pd

from src.dsl.parser import parse
from src.dsl.compute import compute_factor
from src.strategy.rolling_best import StrategyConfig, backtest, select_best_factor


def load_data(market: str):
    cache = Path(f"data/.cache/{market}_prices.pkl")
    prices = pickle.load(open(cache, "rb"))

    sym_col = "ts_code" if market == "cn" else "symbol"
    date_col = "trade_date" if market == "cn" else "date"

    prices = prices.sort_values([sym_col, date_col])
    prices["ret_next"] = prices.groupby(sym_col)["close"].transform(
        lambda x: x.shift(-1) / x - 1
    )

    con = duckdb.connect("data/factor_lab.duckdb", read_only=True)
    promoted = con.execute(
        "SELECT name, formula FROM factor_registry WHERE market=? AND status='promoted'",
        [market],
    ).fetchdf()
    con.close()

    all_factors = {}
    for _, row in promoted.iterrows():
        try:
            ast = parse(row["formula"])
            vals = compute_factor(ast, prices, sym_col=sym_col, date_col=date_col)
            merged = vals.merge(
                prices[[sym_col, date_col, "ret_next"]],
                on=[sym_col, date_col],
            ).dropna(subset=["factor_value"])
            all_factors[row["name"]] = merged
        except Exception:
            pass

    return prices, all_factors, sym_col, date_col


def run_backtest(market: str, cfg: StrategyConfig):
    t0 = time.time()
    prices, all_factors, sym_col, date_col = load_data(market)
    print(f"Loaded {len(all_factors)} factors in {time.time()-t0:.0f}s")

    dates = sorted([d for d in prices[date_col].unique() if str(d) >= "2024-06-01"])

    # Benchmark
    if market == "us":
        spy = prices[prices["symbol"] == "SPY"][[date_col, "ret_next"]].dropna()
        bench_map = dict(zip(spy[date_col], spy["ret_next"]))
        bench_name = "SPY"
    else:
        daily_mkt = prices.groupby(date_col)["ret_next"].mean()
        bench_map = daily_mkt.to_dict()
        bench_name = "EqWgt"

    results = backtest(all_factors, prices, dates, sym_col, date_col, cfg, bench_map)

    if results.empty:
        print("No results")
        return

    arr = results["ret"].values
    bench = results["benchmark"].values
    excess = arr - bench

    ann = arr.mean() * 252
    vol = arr.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0
    eq = np.cumprod(1 + arr)
    dd = np.min(eq / np.maximum.accumulate(eq) - 1)
    win = (arr > 0).mean()
    ann_bench = bench.mean() * 252
    ann_excess = excess.mean() * 252

    # Early exits triggered
    early_exits = results[
        (results["days_held"] < cfg.hold_max - 1) &
        (results["days_held"] > 0) &
        (results.groupby((results["factor"] != results["factor"].shift()).cumsum()).cumcount() == 0)
    ]

    print(f"\n{'='*60}")
    print(f"  {market.upper()} Rolling Best-Factor + SigReg Exit")
    print(f"  Config: lookback={cfg.lookback}d, hold={cfg.hold_max}d, exit_thresh={cfg.ic_exit_threshold}")
    print(f"{'='*60}")
    print(f"  Ann Return:    {ann*100:+.1f}%")
    print(f"  {bench_name}:          {ann_bench*100:+.1f}%")
    print(f"  Excess:        {ann_excess*100:+.1f}%")
    print(f"  Sharpe:        {sharpe:.2f}")
    print(f"  Max DD:        {dd*100:.1f}%")
    print(f"  Win Rate:      {win*100:.0f}%")
    print(f"  Cum Return:    {(eq[-1]-1)*100:+.1f}%")
    print(f"  Trading Days:  {len(arr)}")
    print(f"  Early Exits:   ~{len(early_exits)} (SigReg triggered)")

    # Monthly breakdown
    results["date"] = pd.to_datetime(results["date"])
    results["month"] = results["date"].dt.to_period("M")
    monthly = results.groupby("month").agg(
        ret=("ret", "sum"), bench=("benchmark", "sum")
    )
    monthly["excess"] = monthly["ret"] - monthly["bench"]

    print(f"\n  Monthly:")
    for m, r in monthly.tail(8).iterrows():
        bar = "+" * max(0, int(r["excess"] * 20)) + "-" * max(0, int(-r["excess"] * 20))
        print(f"    {m}: {r['ret']*100:+5.1f}% {bench_name} {r['bench']*100:+5.1f}% ex {r['excess']*100:+5.1f}% {bar}")

    # Factor usage
    factor_usage = results["factor"].value_counts()
    print(f"\n  Factor rotation:")
    for fname, cnt in factor_usage.head(5).items():
        print(f"    {fname:<28s}: {cnt:4d} days ({cnt/len(results)*100:.0f}%)")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


def show_today(market: str, cfg: StrategyConfig):
    """Output actionable trading instructions."""
    prices, all_factors, sym_col, date_col = load_data(market)
    dates = sorted(prices[date_col].unique())
    latest = dates[-1]
    # Shift lookback back by hold_max to avoid using unrealized future returns
    lb_dates = dates[-(cfg.lookback + cfg.hold_max) - 1 : -cfg.hold_max - 1]

    factor_name, side, sharpe = select_best_factor(
        all_factors, lb_dates, cfg.hold_max, date_col, cfg.n_picks
    )

    if factor_name is None:
        print("No factor selected — stay cash")
        return

    fdata = all_factors[factor_name]
    today = fdata[fdata[date_col] == latest].dropna(subset=["factor_value"])

    if side == "top":
        picks = today.nlargest(cfg.n_picks, "factor_value")
    else:
        picks = today.nsmallest(cfg.n_picks, "factor_value")

    today_prices = prices[prices[date_col] == latest].set_index(sym_col)

    # Compute ATR for stop/target
    prices_sorted = prices.sort_values([sym_col, date_col])
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
    prices_sorted["atr"] = atr_14
    atr_today = prices_sorted[prices_sorted[date_col] == latest].set_index(sym_col)["atr"]

    # SigReg IC health check
    from src.evaluate.sigreg import ic_health_test
    from scipy.stats import spearmanr as _sp
    ics = []
    for dt in lb_dates[-20:]:
        day = fdata[fdata[date_col] == dt].dropna(subset=["factor_value", "ret_next"])
        if len(day) >= 30:
            rho, _ = _sp(day["factor_value"], day["ret_next"])
            if not np.isnan(rho):
                ics.append(rho)
    health = ic_health_test(ics) if ics else {"health_score": 0.5, "ic_mean_recent": 0}
    health_icon = "🟢" if health["health_score"] >= 0.7 else "🟡" if health["health_score"] >= 0.4 else "🔴"

    label = "A股" if market == "cn" else "美股"
    print(f"\n{'═'*60}")
    print(f"  {label} 交易指令 — {latest.date()}")
    print(f"  因子: {factor_name} | 持仓: {cfg.hold_max}天 | 健康: {health_icon} {health['health_score']}")
    print(f"{'═'*60}")

    if health["health_score"] < 0.4:
        print(f"  ⚠️ 因子健康度过低, 建议观望不操作")
        return

    # Position sizing: rank-based (top pick 2x weight of bottom pick)
    # Rank 1 (strongest signal) gets weight proportional to N
    # Rank N (weakest signal) gets weight proportional to 1
    # This creates meaningful differentiation without volatility distortion
    sizing_data = []
    n_valid = 0
    for rank_i, (_, row) in enumerate(picks.iterrows()):
        sym = row[sym_col]
        if sym not in today_prices.index:
            continue
        close = today_prices.loc[sym, "close"]
        atr = atr_today.loc[sym] if sym in atr_today.index else close * 0.03
        if pd.isna(atr) or atr <= 0:
            atr = close * 0.03

        stop = round(close - 2 * atr, 2)
        target = round(close + 3 * atr, 2)
        if market == "cn":
            stop = max(stop, round(close * 0.90, 2))

        rank_weight = cfg.n_picks - rank_i  # top pick = N, bottom = 1
        sizing_data.append({
            "sym": sym, "close": close, "stop": stop, "target": target,
            "atr_pct": atr / close,
            "raw_weight": rank_weight,
        })

    if not sizing_data:
        print("  No valid picks")
        return

    total_w = sum(d["raw_weight"] for d in sizing_data)
    for d in sizing_data:
        d["weight"] = d["raw_weight"] / total_w * 100

    print(f"  操作: 信号强度加权买入以下 {len(sizing_data)} 只")
    print(f"  加权: #1 信号最强 → {sizing_data[0]['weight']:.0f}%, #10 最弱 → {sizing_data[-1]['weight']:.0f}%")
    print(f"  持有: {cfg.hold_max} 个交易日后平仓 (或止损触发时提前平)")
    print()
    print(f"  {'#':>3s} {'代码':<12s} {'买入价':>8s} {'止损':>8s} {'止盈':>8s} {'仓位':>6s}")
    print(f"  {'─'*46}")

    for i, d in enumerate(sizing_data):
        print(f"  {i+1:>3d} {d['sym']:<12s} {d['close']:>8.2f} {d['stop']:>8.2f} {d['target']:>8.2f} {d['weight']:>5.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["cn", "us"], required=True)
    parser.add_argument("--lookback", type=int, default=40)
    parser.add_argument("--hold", type=int, default=20)
    parser.add_argument("--n-picks", type=int, default=10)
    parser.add_argument("--ic-exit", type=float, default=-0.02)
    parser.add_argument("--today", action="store_true", help="Show today's picks")
    args = parser.parse_args()

    cfg = StrategyConfig(
        lookback=args.lookback,
        hold_max=args.hold,
        rebalance=args.hold,
        n_picks=args.n_picks,
        ic_exit_threshold=args.ic_exit,
    )

    if args.today:
        show_today(args.market, cfg)
    else:
        run_backtest(args.market, cfg)


if __name__ == "__main__":
    main()
