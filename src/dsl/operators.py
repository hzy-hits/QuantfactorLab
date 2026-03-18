"""
Implementation of all DSL operators for the Factor Lab.

Three categories:
  1. Time-series  — applied per-stock (groupby symbol, along the time axis)
  2. Cross-sectional — applied per-date (groupby date, across all stocks)
  3. Universal — element-wise or simple transforms
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ===========================================================================
# Time-series operators  (per-stock, along time axis)
# ===========================================================================

def ts_mean(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def ts_std(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).std(ddof=1)


def ts_max(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).max()


def ts_min(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).min()


def ts_sum(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).sum()


def ts_rank(x: pd.Series, n: int) -> pd.Series:
    """Percentile rank of the current value within the last *n* values."""
    def _pct_rank(arr: np.ndarray) -> float:
        if np.isnan(arr[-1]):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            return np.nan
        return (valid < arr[-1]).sum() / (len(valid) - 1)

    return x.rolling(n, min_periods=n).apply(_pct_rank, raw=True)


def ts_argmax(x: pd.Series, n: int) -> pd.Series:
    """Days since the max within the last *n* values (0 = today)."""
    def _argmax(arr: np.ndarray) -> float:
        if np.any(np.isnan(arr)):
            return np.nan
        return float(len(arr) - 1 - np.argmax(arr))

    return x.rolling(n, min_periods=n).apply(_argmax, raw=True)


def ts_argmin(x: pd.Series, n: int) -> pd.Series:
    """Days since the min within the last *n* values (0 = today)."""
    def _argmin(arr: np.ndarray) -> float:
        if np.any(np.isnan(arr)):
            return np.nan
        return float(len(arr) - 1 - np.argmin(arr))

    return x.rolling(n, min_periods=n).apply(_argmin, raw=True)


def ts_corr(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).corr(y)


def ts_cov(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).cov(y)


def ts_count(s: pd.Series, n: int) -> pd.Series:
    """Count of positive (> 0) values in the last *n* periods."""
    return (s > 0).astype(float).rolling(int(n), min_periods=1).sum()


def ts_skew(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).skew()


def ts_kurt(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).kurt()


def ts_product(x: pd.Series, n: int) -> pd.Series:
    """Rolling product over the last *n* values."""
    return x.rolling(n, min_periods=n).apply(np.prod, raw=True)


def delta(x: pd.Series, n: int) -> pd.Series:
    """x[t] - x[t - n]."""
    return x - x.shift(n)


def pct_change(x: pd.Series, n: int) -> pd.Series:
    """(x[t] / x[t - n]) - 1, guarded against division by zero."""
    prev = x.shift(n)
    return (x / prev.replace(0, np.nan)) - 1


def shift(x: pd.Series, n: int) -> pd.Series:
    """Lag by *n* periods."""
    return x.shift(n)


def decay_linear(x: pd.Series, n: int) -> pd.Series:
    """Linearly-weighted moving average: weight_i = i / sum(1..n)."""
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()

    def _wma(arr: np.ndarray) -> float:
        if np.any(np.isnan(arr)):
            return np.nan
        return float(np.dot(arr, weights))

    return x.rolling(n, min_periods=n).apply(_wma, raw=True)


def decay_exp(x: pd.Series, n: int) -> pd.Series:
    """Exponentially-weighted moving average with half-life = *n*."""
    return x.ewm(halflife=n, min_periods=n).mean()


# ===========================================================================
# Cross-sectional operators  (per-date, across all stocks)
# ===========================================================================

def rank(x: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank in [0, 1]."""
    return x.rank(pct=True)


def zscore(x: pd.Series) -> pd.Series:
    """Cross-sectional z-score."""
    mu = x.mean()
    sigma = x.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return x * 0.0  # all same → zero
    return (x - mu) / sigma


def demean(x: pd.Series) -> pd.Series:
    """Subtract cross-sectional mean."""
    return x - x.mean()


# ===========================================================================
# Universal operators  (element-wise)
# ===========================================================================

def op_abs(x: pd.Series) -> pd.Series:
    return x.abs()


def sign(x: pd.Series) -> pd.Series:
    return np.sign(x)


def log(x: pd.Series) -> pd.Series:
    """Natural log, guarded: log(x) where x <= 0 → NaN."""
    return np.log(x.where(x > 0, np.nan))


def sqrt(x: pd.Series) -> pd.Series:
    """Square root, guarded: negative values → NaN."""
    return np.sqrt(x.where(x >= 0, np.nan))


def power(x: pd.Series, p: float) -> pd.Series:
    return x ** p


def clamp(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return x.clip(lower=lo, upper=hi)


def op_max(x: pd.Series, y: pd.Series) -> pd.Series:
    """Element-wise max of two series."""
    return pd.concat([x, y], axis=1).max(axis=1)


def op_min(x: pd.Series, y: pd.Series) -> pd.Series:
    """Element-wise min of two series."""
    return pd.concat([x, y], axis=1).min(axis=1)


def if_then(cond: pd.Series, a: pd.Series, b: pd.Series) -> pd.Series:
    """Where cond > 0, return a; otherwise b."""
    return a.where(cond > 0, b)


# ===========================================================================
# Operator registry (name → callable)
# ===========================================================================

# Maps DSL function name to (callable, operator_type).
# operator_type: "ts" = time-series, "cs" = cross-sectional, "univ" = universal.

OPERATOR_REGISTRY: dict[str, tuple[callable, str]] = {
    # time-series
    "ts_mean":      (ts_mean, "ts"),
    "ts_std":       (ts_std, "ts"),
    "ts_max":       (ts_max, "ts"),
    "ts_min":       (ts_min, "ts"),
    "ts_sum":       (ts_sum, "ts"),
    "ts_rank":      (ts_rank, "ts"),
    "ts_argmax":    (ts_argmax, "ts"),
    "ts_argmin":    (ts_argmin, "ts"),
    "ts_corr":      (ts_corr, "ts"),
    "ts_cov":       (ts_cov, "ts"),
    "ts_count":     (ts_count, "ts"),
    "ts_skew":      (ts_skew, "ts"),
    "ts_kurt":      (ts_kurt, "ts"),
    "ts_product":   (ts_product, "ts"),
    "delta":        (delta, "ts"),
    "pct_change":   (pct_change, "ts"),
    "shift":        (shift, "ts"),
    "decay_linear": (decay_linear, "ts"),
    "decay_exp":    (decay_exp, "ts"),
    # cross-sectional
    "rank":         (rank, "cs"),
    "zscore":       (zscore, "cs"),
    "demean":       (demean, "cs"),
    # universal
    "abs":          (op_abs, "univ"),
    "sign":         (sign, "univ"),
    "log":          (log, "univ"),
    "sqrt":         (sqrt, "univ"),
    "power":        (power, "univ"),
    "clamp":        (clamp, "univ"),
    "max":          (op_max, "univ"),
    "min":          (op_min, "univ"),
    "if_then":      (if_then, "univ"),
}
