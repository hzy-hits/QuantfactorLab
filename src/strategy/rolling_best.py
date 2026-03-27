"""Rolling Best-Factor Strategy with SigReg dynamic exit.

Core logic:
1. Every REBALANCE days, look back LOOKBACK days
2. For each promoted factor, compute rolling Sharpe of top/bottom 20
3. Pick the factor with best recent Sharpe + determine direction
4. Hold for up to HOLD_MAX days, but exit early if SigReg IC health degrades

This replaces voting composite, IC_IR weighting, and all other combination methods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    lookback: int = 40        # days to look back for factor selection
    hold_max: int = 5         # max holding period (days)
    hold_min: int = 2         # min holding period (T+1 constraint)
    rebalance: int = 5        # rebalance frequency (= hold_max for non-overlapping)
    n_picks: int = 10         # stocks to hold
    ic_exit_window: int = 10  # rolling window for IC health check
    ic_exit_threshold: float = -0.02  # IC below this → exit early
    adaptive_hold: bool = True  # if True, exit early when IC degrades


@dataclass
class Position:
    factor_name: str
    side: str  # 'top' or 'bot'
    symbols: list[str]
    entry_date: object
    entry_prices: dict  # {symbol: price}
    days_held: int = 0


def select_best_factor(
    all_factors: dict[str, pd.DataFrame],
    lookback_dates: list,
    hold: int,
    date_col: str,
    n_picks: int = 20,
) -> tuple[str | None, str | None, float]:
    """Select the factor with best recent risk-adjusted return.

    Returns (factor_name, side, sharpe) or (None, None, 0).
    """
    ret_col = f"ret_{hold}d" if f"ret_{hold}d" in next(iter(all_factors.values())).columns else "ret_next"

    best_sh = -999
    best_factor = None
    best_side = None

    for fname, fmerged in all_factors.items():
        lb = fmerged[fmerged[date_col].isin(lookback_dates)]
        if len(lb) < len(lookback_dates) * 3:
            continue

        top_r, bot_r = [], []
        for dt in lookback_dates:
            day = lb[lb[date_col] == dt].dropna(subset=["factor_value", ret_col])
            if len(day) < 30:
                continue
            top_r.append(day.nlargest(n_picks, "factor_value")[ret_col].mean())
            bot_r.append(day.nsmallest(n_picks, "factor_value")[ret_col].mean())

        if len(top_r) < 5:
            continue

        ta, ba = np.array(top_r), np.array(bot_r)
        ts = ta.mean() / (ta.std() + 1e-9)
        bs = ba.mean() / (ba.std() + 1e-9)

        if max(ts, bs) > best_sh:
            best_sh = max(ts, bs)
            best_factor = fname
            best_side = "top" if ts > bs else "bot"

    return best_factor, best_side, best_sh


def check_ic_health(
    factor_data: pd.DataFrame,
    recent_dates: list,
    date_col: str,
    threshold: float = -0.02,
) -> tuple[float, bool]:
    """Check if factor IC is still healthy on recent days.

    Returns (recent_ic, should_exit).
    """
    ics = []
    for dt in recent_dates:
        day = factor_data[factor_data[date_col] == dt].dropna(subset=["factor_value", "ret_next"])
        if len(day) >= 30:
            rho, _ = spearmanr(day["factor_value"], day["ret_next"])
            if not np.isnan(rho):
                ics.append(rho)

    if not ics:
        return 0.0, False

    recent_ic = np.mean(ics)
    should_exit = recent_ic < threshold
    return recent_ic, should_exit


def backtest(
    all_factors: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    dates: list,
    sym_col: str,
    date_col: str,
    cfg: StrategyConfig | None = None,
    benchmark_map: dict | None = None,
) -> pd.DataFrame:
    """Run full rolling best-factor backtest.

    Returns DataFrame with columns: date, ret, benchmark, factor, side, ic_health, n_stocks.
    """
    if cfg is None:
        cfg = StrategyConfig()

    # Precompute multi-horizon returns
    for h in [cfg.hold_max, 5, 10, 20]:
        col = f"ret_{h}d"
        if col not in prices.columns:
            prices[col] = prices.groupby(sym_col)["close"].transform(lambda x: x.shift(-h) / x - 1)

    if "ret_next" not in prices.columns:
        prices["ret_next"] = prices.groupby(sym_col)["close"].transform(lambda x: x.shift(-1) / x - 1)

    # Merge returns into factor data
    ret_cols = [c for c in prices.columns if c.startswith("ret_")]
    for fname in all_factors:
        all_factors[fname] = all_factors[fname].merge(
            prices[[sym_col, date_col] + ret_cols],
            on=[sym_col, date_col],
            how="left",
            suffixes=("", "_dup"),
        )
        # Drop duplicate columns
        all_factors[fname] = all_factors[fname][[c for c in all_factors[fname].columns if not c.endswith("_dup")]]

    ret_col = f"ret_{cfg.hold_max}d"
    results = []
    position: Position | None = None

    i = cfg.lookback + cfg.hold_max  # need extra hold_max for no-leakage lookback
    while i < len(dates) - 1:
        dt = dates[i]

        # Check if we need to rebalance or exit
        need_rebalance = position is None or position.days_held >= cfg.hold_max

        # Adaptive hold: IC-based early exit (only after hold_min days)
        if (cfg.adaptive_hold and position is not None
                and not need_rebalance
                and position.days_held >= cfg.hold_min):
            recent = dates[max(0, i - cfg.ic_exit_window) : i]
            fdata = all_factors.get(position.factor_name)
            if fdata is not None:
                ic, should_exit = check_ic_health(
                    fdata, recent, date_col, cfg.ic_exit_threshold
                )
                if should_exit:
                    need_rebalance = True

        if need_rebalance:
            # Select best factor using FULLY REALIZED returns only.
            # ret_{hold}d on day T needs price at T+hold, so to avoid
            # look-ahead we shift the lookback window back by hold_max days.
            lb_start = max(0, i - cfg.lookback - cfg.hold_max)
            lb_end = max(0, i - cfg.hold_max)
            lb_dates = dates[lb_start:lb_end]
            factor_name, side, _ = select_best_factor(
                all_factors, lb_dates, cfg.hold_max, date_col, cfg.n_picks
            )

            if factor_name is None:
                i += 1
                continue

            # Pick stocks
            fdata = all_factors[factor_name]
            today = fdata[fdata[date_col] == dt].dropna(subset=["factor_value"])
            if len(today) < 30:
                i += 1
                continue

            if side == "top":
                picks = today.nlargest(cfg.n_picks, "factor_value")
            else:
                picks = today.nsmallest(cfg.n_picks, "factor_value")

            position = Position(
                factor_name=factor_name,
                side=side,
                symbols=picks[sym_col].tolist(),
                entry_date=dt,
                entry_prices={},
                days_held=0,
            )

        # Compute daily return of current position
        if position is not None:
            day_prices = prices[prices[date_col] == dt].set_index(sym_col)
            rets = []
            for sym in position.symbols:
                if sym in day_prices.index:
                    r = day_prices.loc[sym, "ret_next"]
                    if not np.isnan(r):
                        rets.append(r)

            if rets:
                port_ret = np.mean(rets)
            else:
                port_ret = 0.0

            bench = benchmark_map.get(dt, 0.0) if benchmark_map else 0.0

            # IC health for logging
            recent = dates[max(0, i - cfg.ic_exit_window) : i]
            fdata = all_factors.get(position.factor_name)
            ic_val = 0.0
            if fdata is not None and len(recent) > 3:
                ic_val, _ = check_ic_health(fdata, recent, date_col)

            results.append({
                "date": dt,
                "ret": port_ret,
                "benchmark": bench,
                "factor": position.factor_name,
                "side": position.side,
                "ic_health": ic_val,
                "n_stocks": len(rets),
                "days_held": position.days_held,
            })

            position.days_held += 1

        i += 1

    return pd.DataFrame(results)
