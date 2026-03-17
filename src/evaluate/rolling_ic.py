"""Rolling IC analysis — detect IC decay and time-varying predictability."""
import numpy as np
import pandas as pd


def compute_rolling_ic(ic_series: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Rolling window IC mean and cumulative IC.

    Parameters
    ----------
    ic_series : pd.DataFrame
        DataFrame with columns [date, ic] from ``ic.compute_ic_series()``.
    window : int
        Rolling window size in trading days (default 60).

    Returns
    -------
    pd.DataFrame
        Columns: [date, rolling_ic, cumulative_ic]
    """
    df = ic_series[["date", "ic"]].dropna().sort_values("date").reset_index(drop=True)

    if len(df) == 0:
        return pd.DataFrame(columns=["date", "rolling_ic", "cumulative_ic"])

    df["rolling_ic"] = df["ic"].rolling(window=window, min_periods=max(1, window // 2)).mean()
    df["cumulative_ic"] = df["ic"].expanding().mean()

    return df[["date", "rolling_ic", "cumulative_ic"]].copy()


def compute_ic_trend(ic_series: pd.DataFrame) -> dict:
    """
    Fit a linear regression on daily IC values to detect decay.

    Parameters
    ----------
    ic_series : pd.DataFrame
        DataFrame with columns [date, ic] from ``ic.compute_ic_series()``.

    Returns
    -------
    dict
        {
            "slope": float — IC per day (positive = improving, negative = decaying),
            "slope_per_year": float — annualized slope (* 252),
            "r_squared": float — goodness of fit,
            "n_days": int,
            "verdict": str — "stable" / "decaying" / "improving",
        }
    """
    df = ic_series[["date", "ic"]].dropna().sort_values("date").reset_index(drop=True)

    if len(df) < 10:
        return {
            "slope": 0.0,
            "slope_per_year": 0.0,
            "r_squared": 0.0,
            "n_days": len(df),
            "verdict": "insufficient_data",
        }

    x = np.arange(len(df), dtype=float)
    y = df["ic"].values.astype(float)

    # Least squares: y = a + b*x
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx < 1e-15:
        slope = 0.0
    else:
        slope = ss_xy / ss_xx

    y_pred = y_mean + slope * (x - x_mean)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    slope_per_year = slope * 252

    # Verdict: use annualized slope relative to mean IC
    if abs(slope_per_year) < 0.005:
        verdict = "stable"
    elif slope_per_year < 0:
        verdict = "decaying"
    else:
        verdict = "improving"

    return {
        "slope": round(float(slope), 6),
        "slope_per_year": round(float(slope_per_year), 4),
        "r_squared": round(float(r_squared), 4),
        "n_days": len(df),
        "verdict": verdict,
    }
