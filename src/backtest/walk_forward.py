"""
Walk-forward backtest engine with expanding windows.

Anchored expanding window: train always starts from the beginning of the IS period.
OOS check is a separate method that returns bool only (agent never sees OOS metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class FoldMetrics:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    ic: float
    ic_ir: float
    sharpe: float  # long-short daily sharpe
    turnover: float
    monotonicity: float
    n_test_days: int


@dataclass
class BacktestResult:
    fold_metrics: list[FoldMetrics]
    avg_ic: float
    avg_ic_ir: float
    avg_sharpe: float
    avg_turnover: float
    avg_monotonicity: float
    oos_pass: bool | None  # None = not yet checked


def _daily_ic(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily rank IC between factor_value and fwd_5d.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns [date, factor_value, fwd_5d].

    Returns
    -------
    pd.DataFrame with columns [date, ic].
    """
    results = []
    for dt, group in df.groupby("date"):
        valid = group.dropna(subset=["factor_value", "fwd_5d"])
        if len(valid) < 10:
            continue
        rho, _ = spearmanr(valid["factor_value"], valid["fwd_5d"])
        if not np.isnan(rho):
            results.append({"date": dt, "ic": rho})
    return pd.DataFrame(results)


def _quintile_long_short(df: pd.DataFrame, n_groups: int = 5) -> pd.DataFrame:
    """Daily long-short return: mean(Q5) - mean(Q1).

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns [date, factor_value, fwd_5d].

    Returns
    -------
    pd.DataFrame with columns [date, ls_return, turnover].
    """
    results = []
    prev_long: set | None = None
    prev_short: set | None = None

    for dt, group in sorted(df.groupby("date"), key=lambda x: x[0]):
        valid = group.dropna(subset=["factor_value", "fwd_5d"])
        if len(valid) < n_groups * 5:
            prev_long = None
            prev_short = None
            continue

        valid = valid.copy()
        valid["quintile"] = pd.qcut(
            valid["factor_value"], n_groups, labels=False, duplicates="drop"
        ) + 1

        if valid["quintile"].nunique() < n_groups:
            prev_long = None
            prev_short = None
            continue

        long_mask = valid["quintile"] == n_groups
        short_mask = valid["quintile"] == 1
        ls_ret = valid.loc[long_mask, "fwd_5d"].mean() - valid.loc[short_mask, "fwd_5d"].mean()

        # Turnover: fraction of top/bottom group that changed from prior day
        cur_long = set(valid.loc[long_mask, "symbol"].values) if "symbol" in valid.columns else set()
        cur_short = set(valid.loc[short_mask, "symbol"].values) if "symbol" in valid.columns else set()

        if prev_long is not None and len(cur_long) > 0:
            long_to = 1.0 - len(cur_long & prev_long) / max(len(cur_long), 1)
            short_to = 1.0 - len(cur_short & prev_short) / max(len(cur_short), 1)
            turnover = (long_to + short_to) / 2.0
        else:
            turnover = np.nan

        prev_long = cur_long
        prev_short = cur_short

        results.append({"date": dt, "ls_return": ls_ret, "turnover": turnover})

    return pd.DataFrame(results)


def _quintile_monotonicity(df: pd.DataFrame, n_groups: int = 5) -> float:
    """Average monotonicity: Spearman(quintile_rank, mean_return) across days."""
    daily_qmeans: dict[int, list[float]] = {q: [] for q in range(1, n_groups + 1)}

    for dt, group in df.groupby("date"):
        valid = group.dropna(subset=["factor_value", "fwd_5d"])
        if len(valid) < n_groups * 5:
            continue
        valid = valid.copy()
        valid["quintile"] = pd.qcut(
            valid["factor_value"], n_groups, labels=False, duplicates="drop"
        ) + 1
        if valid["quintile"].nunique() < n_groups:
            continue
        for q in range(1, n_groups + 1):
            qr = valid.loc[valid["quintile"] == q, "fwd_5d"].mean()
            if not np.isnan(qr):
                daily_qmeans[q].append(qr)

    q_means = []
    for q in range(1, n_groups + 1):
        vals = daily_qmeans[q]
        q_means.append(float(np.mean(vals)) if vals else 0.0)

    if len(q_means) < 3:
        return 0.0
    mono, _ = spearmanr(list(range(1, n_groups + 1)), q_means)
    return float(mono) if not np.isnan(mono) else 0.0


def _compute_fold(
    factor_df: pd.DataFrame,
    fwd_df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    sym_col: str,
    date_col: str,
    cost_per_trade: float,
) -> FoldMetrics | None:
    """Evaluate one walk-forward fold.

    Train period is used only to confirm the factor has signal (not for fitting
    any parameters, since the DSL factors are parameter-free).  Test period
    metrics are what matter.
    """
    # Merge factor values with forward returns for the test window
    merged = factor_df.merge(fwd_df, on=[sym_col, date_col], how="inner")
    test = merged[
        (merged[date_col] >= test_start) & (merged[date_col] <= test_end)
    ].copy()

    if len(test) == 0:
        return None

    # Normalise column names for helper functions
    test = test.rename(columns={sym_col: "symbol", date_col: "date"})

    # --- IC ---
    ic_df = _daily_ic(test)
    if len(ic_df) == 0:
        return None
    ic_mean = float(ic_df["ic"].mean())
    ic_std = float(ic_df["ic"].std())
    ic_ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0

    # --- Long-short return + turnover ---
    ls_df = _quintile_long_short(test)
    if len(ls_df) == 0:
        return None

    avg_turnover = float(ls_df["turnover"].dropna().mean()) if len(ls_df["turnover"].dropna()) > 0 else 0.0

    # Cost-adjusted daily returns (5-day fwd divided by 5 to get ~daily)
    # Always use raw Q5-Q1 direction — no auto-flip, which would be IS overfitting.
    daily_ls = ls_df["ls_return"].values / 5.0
    cost_adj = daily_ls - avg_turnover * cost_per_trade
    sharpe = float(np.mean(cost_adj) / np.std(cost_adj) * np.sqrt(252)) if np.std(cost_adj) > 1e-10 else 0.0

    # --- Monotonicity ---
    mono = _quintile_monotonicity(test)

    n_test_days = int(test["date"].nunique())

    return FoldMetrics(
        train_start=str(train_start),
        train_end=str(train_end),
        test_start=str(test_start),
        test_end=str(test_end),
        ic=round(ic_mean, 4),
        ic_ir=round(ic_ir, 3),
        sharpe=round(sharpe, 3),
        turnover=round(avg_turnover, 4),
        monotonicity=round(mono, 3),
        n_test_days=n_test_days,
    )


def walk_forward_backtest(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    sym_col: str = "ts_code",
    date_col: str = "trade_date",
    oos_start: str = "2025-10-01",
    n_folds: int = 2,
    min_train_days: int = 120,
    min_test_days: int = 60,
    cost_per_trade: float = 0.003,
) -> BacktestResult:
    """Run anchored expanding-window walk-forward backtest (IS only).

    Parameters
    ----------
    factor_values : pd.DataFrame
        Must have columns [sym_col, date_col, "factor_value"].
    forward_returns : pd.DataFrame
        Must have columns [sym_col, date_col, "fwd_5d"].
    sym_col : str
        Symbol column name.
    date_col : str
        Date column name.
    oos_start : str
        Start of OOS period. IS period ends the day before this.
    n_folds : int
        Number of walk-forward folds within the IS period.
    min_train_days : int
        Minimum trading days in the training window.
    min_test_days : int
        Minimum trading days in the test window.
    cost_per_trade : float
        Round-trip transaction cost (applied per unit turnover).

    Returns
    -------
    BacktestResult with fold_metrics, averages, and oos_pass=None.
    """
    # Ensure consistent date types (string comparison friendly)
    factor_values = factor_values.copy()
    forward_returns = forward_returns.copy()
    factor_values[date_col] = pd.to_datetime(factor_values[date_col]).dt.strftime("%Y-%m-%d")
    forward_returns[date_col] = pd.to_datetime(forward_returns[date_col]).dt.strftime("%Y-%m-%d")

    # IS period: exclude the last fwd_horizon TRADING days before oos_start
    # to prevent forward return leakage (fwd_5d at boundary peeks into OOS).
    # Using trading days (not calendar days) is holiday-safe.
    fwd_horizon = 5  # must match the forward return horizon
    all_dates_before_oos = sorted(
        d for d in factor_values[date_col].unique() if d < oos_start
    )
    if len(all_dates_before_oos) > fwd_horizon:
        is_cutoff = all_dates_before_oos[-(fwd_horizon + 1)]  # exclude last 5 trading days
    else:
        is_cutoff = oos_start
    is_factor = factor_values[factor_values[date_col] <= is_cutoff]
    is_fwd = forward_returns[forward_returns[date_col] <= is_cutoff]

    is_dates = sorted(is_factor[date_col].unique())
    if len(is_dates) < min_train_days + min_test_days:
        return BacktestResult(
            fold_metrics=[],
            avg_ic=0.0,
            avg_ic_ir=0.0,
            avg_sharpe=0.0,
            avg_turnover=0.0,
            avg_monotonicity=0.0,
            oos_pass=None,
        )

    # Anchored expanding window: train always starts at is_dates[0].
    # Split the IS dates into n_folds test windows.
    # Reserve at least min_train_days for the first fold's training.
    available_test_dates = is_dates[min_train_days:]
    if len(available_test_dates) < min_test_days:
        return BacktestResult(
            fold_metrics=[],
            avg_ic=0.0,
            avg_ic_ir=0.0,
            avg_sharpe=0.0,
            avg_turnover=0.0,
            avg_monotonicity=0.0,
            oos_pass=None,
        )

    # Divide available test dates into n_folds roughly equal chunks
    chunk_size = max(min_test_days, len(available_test_dates) // n_folds)

    fold_metrics: list[FoldMetrics] = []
    train_start = is_dates[0]

    for fold_i in range(n_folds):
        test_start_idx = fold_i * chunk_size
        test_end_idx = min((fold_i + 1) * chunk_size, len(available_test_dates))
        if test_start_idx >= len(available_test_dates):
            break

        test_dates_slice = available_test_dates[test_start_idx:test_end_idx]
        if len(test_dates_slice) < min_test_days // 2:
            # Skip fold if too few test days
            break

        test_start = test_dates_slice[0]
        test_end = test_dates_slice[-1]

        # Training ends the day before test starts
        train_end_idx = min_train_days + test_start_idx - 1
        train_end = is_dates[max(0, train_end_idx)]

        fm = _compute_fold(
            factor_df=is_factor,
            fwd_df=is_fwd,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            sym_col=sym_col,
            date_col=date_col,
            cost_per_trade=cost_per_trade,
        )
        if fm is not None:
            fold_metrics.append(fm)

    if not fold_metrics:
        return BacktestResult(
            fold_metrics=[],
            avg_ic=0.0,
            avg_ic_ir=0.0,
            avg_sharpe=0.0,
            avg_turnover=0.0,
            avg_monotonicity=0.0,
            oos_pass=None,
        )

    return BacktestResult(
        fold_metrics=fold_metrics,
        avg_ic=round(float(np.mean([f.ic for f in fold_metrics])), 4),
        avg_ic_ir=round(float(np.mean([f.ic_ir for f in fold_metrics])), 3),
        avg_sharpe=round(float(np.mean([f.sharpe for f in fold_metrics])), 3),
        avg_turnover=round(float(np.mean([f.turnover for f in fold_metrics])), 4),
        avg_monotonicity=round(float(np.mean([f.monotonicity for f in fold_metrics])), 3),
        oos_pass=None,
    )


def run_oos_check(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    sym_col: str = "ts_code",
    date_col: str = "trade_date",
    oos_start: str = "2025-10-01",
    market: str = "cn",
    cost_per_trade: float = 0.003,
) -> bool:
    """Run OOS check. Returns PASS/FAIL only. Agent never sees OOS metric values.

    OOS passes if IC > gate minimum for the market (lower bar than IS since
    we only require the factor to have *some* predictive power OOS).
    """
    from src.backtest.gates import GATE_THRESHOLDS

    factor_values = factor_values.copy()
    forward_returns = forward_returns.copy()
    factor_values[date_col] = pd.to_datetime(factor_values[date_col]).dt.strftime("%Y-%m-%d")
    forward_returns[date_col] = pd.to_datetime(forward_returns[date_col]).dt.strftime("%Y-%m-%d")

    oos_factor = factor_values[factor_values[date_col] >= oos_start]
    oos_fwd = forward_returns[forward_returns[date_col] >= oos_start]

    merged = oos_factor.merge(oos_fwd, on=[sym_col, date_col], how="inner")
    if len(merged) == 0:
        return False

    merged = merged.rename(columns={sym_col: "symbol", date_col: "date"})

    ic_df = _daily_ic(merged)
    if len(ic_df) == 0:
        return False

    ic_mean = float(ic_df["ic"].mean())
    ic_std = float(ic_df["ic"].std())
    ic_ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0

    thresholds = GATE_THRESHOLDS.get(market, GATE_THRESHOLDS["cn"])

    # OOS gate: IC > threshold and IC_IR positive
    return abs(ic_mean) >= thresholds["ic_min"] and ic_ir > 0
