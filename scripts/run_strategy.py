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
import re
import time

import duckdb
import numpy as np
import pandas as pd

from src.dsl.parser import parse
from src.dsl.compute import compute_factor
from src.market_data import load_market_prices
from src.paths import FACTOR_LAB_DB, QUANT_CN_DB, QUANT_US_DB
from src.strategy.rolling_best import StrategyConfig, backtest, select_best_factor


US_ALLOWED_SECURITY_TYPES = {
    "Common Stock",
    "ADR",
    "REIT",
    "PUBLIC",
    "Foreign Sh.",
    "NVDR",
    "GDR",
    "MLP",
    "Ltd Part",
    "CDI",
    "NY Reg Shrs",
}
US_RECENT_WINDOW = 60
US_HISTORY_WINDOW = 252
US_MAX_ATR_PCT = 0.45
US_MAX_CLOSE_JUMP_RATIO = 3.0
US_MEDIAN_RATIO_LOW = 0.33
US_MEDIAN_RATIO_HIGH = 3.0
US_STOP_FLOOR_PCT = 0.85
US_BLOCKED_SYMBOL_RE = re.compile(r"[=\^]")
CN_BLOCKED_PREFIXES = ("688", "920", "830", "870")

FILTER_REASON_LABELS = {
    "missing_metadata": "缺少证券元数据",
    "non_equity_symbol": "代码不是股票/ADR",
    "unsupported_type": "证券类型不支持",
    "invalid_close": "收盘价无效",
    "invalid_atr": "ATR 无效",
    "atr_too_wide": "ATR/价格过大",
    "price_regime_break": "价格中枢断层",
    "price_gap_break": "价格跳变异常",
    "cn_untradable_board": "当前账户不可交易板块",
}


def is_cn_tradable_symbol(sym: str) -> bool:
    text = str(sym).upper()
    code = text.split(".")[0]
    if text.endswith(".BJ"):
        return False
    return not code.startswith(CN_BLOCKED_PREFIXES)


def load_data(market: str):
    prices = load_market_prices(market)

    sym_col = "ts_code" if market == "cn" else "symbol"
    date_col = "trade_date" if market == "cn" else "date"

    if market == "cn":
        prices = prices[prices[sym_col].map(is_cn_tradable_symbol)].copy()

    prices = prices.sort_values([sym_col, date_col])
    prices["ret_next"] = prices.groupby(sym_col)["close"].transform(
        lambda x: x.shift(-1) / x - 1
    )

    con = duckdb.connect(str(FACTOR_LAB_DB), read_only=True)
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


def _resolve_effective_trade_date(
    prices: pd.DataFrame,
    date_col: str,
    as_of: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    trade_dates = pd.Index(pd.to_datetime(prices[date_col], errors="coerce").dropna().unique()).sort_values()
    if trade_dates.empty:
        raise ValueError("No trade dates available")

    requested_ts = pd.Timestamp(as_of) if as_of else None
    if requested_ts is None:
        return pd.Timestamp(trade_dates[-1]), None

    eligible = trade_dates[trade_dates <= requested_ts]
    if eligible.empty:
        raise ValueError(f"No trade_date <= {as_of}")
    return pd.Timestamp(eligible[-1]), requested_ts


def _load_us_symbol_metadata() -> dict[str, dict[str, str]]:
    try:
        con = duckdb.connect(str(QUANT_US_DB), read_only=True)
        try:
            df = con.execute(
                """
                SELECT symbol,
                       COALESCE(name, symbol) AS name,
                       COALESCE(type, '') AS type,
                       COALESCE(exchange, '') AS exchange
                FROM us_symbols
                """
            ).fetchdf()
        finally:
            con.close()
    except Exception:
        return {}

    return {
        str(row["symbol"]): {
            "name": str(row["name"] or row["symbol"]),
            "type": str(row["type"] or ""),
            "exchange": str(row["exchange"] or ""),
        }
        for _, row in df.iterrows()
    }


def _build_us_quality_gate(
    prices: pd.DataFrame,
    sym_col: str,
    date_col: str,
    latest: pd.Timestamp,
) -> dict[str, dict[str, object]]:
    df = prices.sort_values([sym_col, date_col]).copy()
    df["prev_close"] = df.groupby(sym_col)["close"].shift(1)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["prev_close"]),
            abs(df["low"] - df["prev_close"]),
        ),
    )
    df["atr"] = df.groupby(sym_col)["tr"].transform(
        lambda s: s.rolling(14, min_periods=5).mean()
    )

    quality: dict[str, dict[str, object]] = {}
    for sym, hist in df.groupby(sym_col, sort=False):
        hist = hist[hist[date_col] <= latest].tail(US_HISTORY_WINDOW).copy()
        if hist.empty:
            continue

        close = float(hist.iloc[-1]["close"])
        atr = hist.iloc[-1]["atr"]
        if pd.isna(atr) or atr <= 0:
            atr = close * 0.03 if close > 0 else np.nan
        atr = float(atr) if pd.notna(atr) else np.nan
        atr_pct = atr / close if close > 0 and np.isfinite(atr) else np.nan

        recent = hist.tail(min(len(hist), US_RECENT_WINDOW))
        hist_median = float(hist["close"].median()) if not hist.empty else np.nan
        recent_median = float(recent["close"].median()) if not recent.empty else np.nan
        median_ratio = (
            recent_median / hist_median
            if np.isfinite(hist_median) and hist_median > 0
            else np.nan
        )

        jumps = (
            hist["close"] / hist["prev_close"]
        ).replace([np.inf, -np.inf], np.nan).dropna()
        max_jump_ratio = 1.0
        if not jumps.empty:
            up_jump = float(jumps.max())
            down_jump = float(jumps.min())
            if down_jump > 0:
                max_jump_ratio = max(up_jump, 1.0 / down_jump)
            else:
                max_jump_ratio = up_jump

        reasons: list[str] = []
        if not np.isfinite(close) or close <= 0:
            reasons.append("invalid_close")
        if not np.isfinite(atr) or atr <= 0:
            reasons.append("invalid_atr")
        if np.isfinite(atr_pct) and atr_pct > US_MAX_ATR_PCT:
            reasons.append("atr_too_wide")
        if np.isfinite(median_ratio) and (
            median_ratio < US_MEDIAN_RATIO_LOW or median_ratio > US_MEDIAN_RATIO_HIGH
        ):
            reasons.append("price_regime_break")
        if np.isfinite(max_jump_ratio) and max_jump_ratio > US_MAX_CLOSE_JUMP_RATIO:
            reasons.append("price_gap_break")

        quality[str(sym)] = {
            "close": close,
            "atr": atr,
            "atr_pct": float(atr_pct) if np.isfinite(atr_pct) else np.nan,
            "median_ratio": float(median_ratio) if np.isfinite(median_ratio) else np.nan,
            "max_jump_ratio": float(max_jump_ratio) if np.isfinite(max_jump_ratio) else np.nan,
            "reasons": reasons,
        }

    return quality


def _format_filter_reason(code: str) -> str:
    return FILTER_REASON_LABELS.get(code, code)


def _us_rejection_reason(
    sym: str,
    metadata: dict[str, str] | None,
    quality: dict[str, object] | None,
) -> str | None:
    if US_BLOCKED_SYMBOL_RE.search(sym):
        return _format_filter_reason("non_equity_symbol")
    if metadata is None:
        return _format_filter_reason("missing_metadata")
    if metadata.get("type", "") not in US_ALLOWED_SECURITY_TYPES:
        return f"{_format_filter_reason('unsupported_type')}: {metadata.get('type') or 'unknown'}"
    if quality:
        reasons = quality.get("reasons") or []
        if reasons:
            return " / ".join(_format_filter_reason(str(reason)) for reason in reasons)
    return None


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


def show_today(market: str, cfg: StrategyConfig, as_of: str | None = None):
    """Output research candidates for the report pipeline, not standalone orders."""
    prices, all_factors, sym_col, date_col = load_data(market)
    trade_dates = pd.Index(pd.to_datetime(prices[date_col], errors="coerce").dropna().unique()).sort_values()
    latest, requested_ts = _resolve_effective_trade_date(prices, date_col, as_of)
    effective_idx = trade_dates.get_loc(latest)

    lb_end = max(0, effective_idx - cfg.hold_max)
    lb_start = max(0, lb_end - cfg.lookback)
    lb_dates = list(trade_dates[lb_start:lb_end])
    if not lb_dates:
        print("Insufficient history for rolling selection")
        return

    factor_name, side, sharpe = select_best_factor(
        all_factors, lb_dates, cfg.hold_max, date_col, cfg.n_picks
    )

    if factor_name is None:
        print("No factor selected — stay cash")
        return

    fdata = all_factors[factor_name]
    today = fdata[fdata[date_col] == latest].dropna(subset=["factor_value"])
    candidate_limit = cfg.n_picks
    if market == "us":
        candidate_limit = min(len(today), max(cfg.n_picks * 5, 25))

    if side == "top":
        picks = today.nlargest(candidate_limit, "factor_value")
    else:
        picks = today.nsmallest(candidate_limit, "factor_value")

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

    # Load stock names for CN
    name_map = {}
    if market == "cn":
        try:
            import duckdb as _ddb2
            _con = _ddb2.connect(str(QUANT_CN_DB), read_only=True)
            _names = _con.execute("SELECT ts_code, name FROM stock_basic").fetchdf()
            name_map = dict(zip(_names["ts_code"], _names["name"]))
            _con.close()
        except Exception:
            pass
        us_metadata = {}
        us_quality = {}
    else:
        us_metadata = _load_us_symbol_metadata()
        us_quality = _build_us_quality_gate(prices, sym_col, date_col, latest)

    label = "A股" if market == "cn" else "美股"

    # Compute hold end date
    import datetime as _dt
    entry_date = latest.date() if hasattr(latest, 'date') else latest
    exit_date = entry_date + _dt.timedelta(days=int(cfg.hold_max * 1.5))  # approximate

    # Run wavelet health on this specific factor
    from src.evaluate.signal_analysis import full_diagnosis
    from scipy.stats import spearmanr as _sp2
    factor_ics = []
    for dt in lb_dates:
        day = fdata[fdata[date_col] == dt].dropna(subset=["factor_value", "ret_next"])
        if len(day) >= 30:
            rho, _ = _sp2(day["factor_value"], day["ret_next"])
            if not np.isnan(rho):
                factor_ics.append(rho)

    wavelet_diag = full_diagnosis(np.array(factor_ics)) if len(factor_ics) >= 30 else None

    if wavelet_diag:
        w_action = wavelet_diag["action"]
        w_icon = {"EXIT": "🔴", "REDUCE": "🟡", "HOLD": "🟢"}.get(w_action, "⚪")
        suggested_hold = wavelet_diag["suggested_hold_days"]
        actual_hold = min(cfg.hold_max, max(3, suggested_hold))
    else:
        w_action = "HOLD"
        w_icon = "⚪"
        actual_hold = cfg.hold_max

    import datetime as _dt
    exit_date = entry_date + _dt.timedelta(days=int(actual_hold * 1.5))

    print(f"")
    print(f"{'═'*70}")
    header_date = requested_ts.date().isoformat() if requested_ts is not None else entry_date.isoformat()
    print(f"  {label} — {header_date}")
    print(f"{'═'*70}")
    print(f"")

    if requested_ts is not None and requested_ts.date() != entry_date:
        print(f"  数据截止: {entry_date} (请求日期 {requested_ts.date().isoformat()} 无更新交易数据)")
        print(f"")

    if w_action == "EXIT":
        print(f"  {w_icon} 因子正在衰减，建议观望不操作。")
        print(f"  原因: {wavelet_diag['reason']}")
        print(f"  等下一次健康因子出现再建仓。")
        print(f"{'═'*70}")
        return

    # Keep Factor Lab subordinate to the main report/execution gate.
    side_note = "因子值最高" if side == "top" else "因子值最低"
    print(f"  类型: Factor Lab 研究候选，不是独立交易指令")
    print(f"  筛选: {side_note} 的 {cfg.n_picks} 只股票")
    print(f"  因子: {factor_name}")
    print(f"  健康: {w_icon} {'正常' if w_action == 'HOLD' else '注意衰减信号'}")
    print(f"")
    print(f"  使用方式:")
    print(f"    1. 只作为主系统的召回/对照候选")
    print(f"    2. 必须再通过主报告方向、执行 gate、流动性和追价过滤")
    print(f"    3. 未通过主系统时，最多进入观察/轮动池")
    print(f"    4. 参考持有窗口 {actual_hold} 个交易日，到 ~{exit_date} 复核")
    if market == "cn":
        print(f"    5. 已剔除科创板和北交所等当前不可交易板块")
    else:
        print(f"    5. 自动剔除非股票代码、价格断层异常和失效止损标的")
    print(f"")

    # Position sizing
    sizing_data = []
    filtered_out = []
    for rank_i, (_, row) in enumerate(picks.iterrows()):
        if len(sizing_data) >= cfg.n_picks:
            break
        sym = row[sym_col]
        if sym not in today_prices.index:
            continue

        if market == "us":
            reject_reason = _us_rejection_reason(sym, us_metadata.get(sym), us_quality.get(sym))
            if reject_reason is not None:
                filtered_out.append((sym, reject_reason))
                continue
        elif not is_cn_tradable_symbol(sym):
            filtered_out.append((sym, _format_filter_reason("cn_untradable_board")))
            continue

        close = today_prices.loc[sym, "close"]
        atr = atr_today.loc[sym] if sym in atr_today.index else close * 0.03
        if pd.isna(atr) or atr <= 0:
            atr = close * 0.03

        stop = round(close - 2 * atr, 2)
        target = round(close + 3 * atr, 2)
        if market == "cn":
            stop = max(stop, round(close * 0.90, 2))
        else:
            stop = max(stop, round(close * US_STOP_FLOOR_PCT, 2), 0.01)

        stock_name = (
            name_map.get(sym, sym)
            if market == "cn"
            else us_metadata.get(sym, {}).get("name", sym)
        )
        rank_weight = cfg.n_picks - len(sizing_data)
        sizing_data.append({
            "sym": sym, "name": stock_name,
            "close": close, "stop": stop, "target": target,
            "atr_pct": atr / close,
            "raw_weight": rank_weight,
        })

    if not sizing_data:
        print("  No valid picks after data quality filters")
        if filtered_out:
            print("")
            print(f"  数据清洗剔除了 {len(filtered_out)} 个候选标的:")
            for sym, reason in filtered_out[:8]:
                print(f"    - {sym}: {reason}")
        return

    total_w = sum(d["raw_weight"] for d in sizing_data)
    for d in sizing_data:
        d["weight"] = d["raw_weight"] / total_w * 100

    print(f"  {'#':>3s} {'代码':<12s} {'名称':<10s} {'参考价':>8s} {'风控线':>8s} {'观察上沿':>8s} {'权重':>6s}")
    print(f"  {'─'*62}")

    for i, d in enumerate(sizing_data):
        name_display = d["name"][:8]  # truncate long names
        print(f"  {i+1:>3d} {d['sym']:<12s} {name_display:<10s} {d['close']:>8.2f} {d['stop']:>8.2f} {d['target']:>8.2f} {d['weight']:>5.1f}%")

    print(f"\n  信号强度: #1 最强 → {sizing_data[0]['weight']:.0f}% 研究权重, #{len(sizing_data)} 最弱 → {sizing_data[-1]['weight']:.0f}% 研究权重")
    if filtered_out:
        print(f"  数据清洗: 已剔除 {len(filtered_out)} 个异常候选")
        for sym, reason in filtered_out[:8]:
            print(f"    - {sym}: {reason}")
        if len(filtered_out) > 8:
            print(f"    - 其余 {len(filtered_out) - 8} 个已省略")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["cn", "us"], required=True)
    parser.add_argument("--lookback", type=int, default=40)
    parser.add_argument("--hold", type=int, default=5)
    parser.add_argument("--n-picks", type=int, default=10)
    parser.add_argument("--ic-exit", type=float, default=-0.02)
    parser.add_argument("--today", action="store_true", help="Show today's picks")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD")
    args = parser.parse_args()

    cfg = StrategyConfig(
        lookback=args.lookback,
        hold_max=args.hold,
        rebalance=args.hold,
        n_picks=args.n_picks,
        ic_exit_threshold=args.ic_exit,
    )

    if args.today:
        show_today(args.market, cfg, as_of=args.date)
    else:
        run_backtest(args.market, cfg)


if __name__ == "__main__":
    main()
