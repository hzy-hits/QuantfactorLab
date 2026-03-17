"""Quintile analysis — do higher factor values predict higher returns?"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_quintile_returns(factor_values: pd.Series, forward_returns: pd.Series,
                             dates: pd.Series, n_groups: int = 5) -> dict:
    """
    Split stocks into N groups by factor value each day, compute mean return per group.

    Returns:
        {
            "quintile_returns": [Q1_mean, Q2_mean, ..., Q5_mean],
            "long_short": Q5 - Q1,
            "monotonicity": Spearman correlation of quintile rank vs return,
            "n_days": number of days used,
        }
    """
    df = pd.DataFrame({
        "date": dates,
        "factor": factor_values,
        "fwd_ret": forward_returns,
    }).dropna()

    daily_quintile_returns = {i: [] for i in range(1, n_groups + 1)}

    for dt, group in df.groupby("date"):
        if len(group) < n_groups * 5:  # need enough stocks per group
            continue

        group = group.copy()
        group["quintile"] = pd.qcut(group["factor"], n_groups, labels=False, duplicates="drop") + 1

        # If duplicates caused fewer bins than requested, skip this day
        actual_bins = group["quintile"].nunique()
        if actual_bins < n_groups:
            continue

        for q in range(1, n_groups + 1):
            qr = group[group["quintile"] == q]["fwd_ret"].mean()
            if not np.isnan(qr):
                daily_quintile_returns[q].append(qr)

    # Average across all days
    q_means = []
    for q in range(1, n_groups + 1):
        vals = daily_quintile_returns[q]
        q_means.append(float(np.mean(vals)) if vals else 0.0)

    # Monotonicity: rank correlation of quintile index vs mean return
    if len(q_means) >= 3:
        mono, _ = spearmanr(list(range(1, n_groups + 1)), q_means)
    else:
        mono = 0.0

    n_days = min(len(v) for v in daily_quintile_returns.values()) if daily_quintile_returns[1] else 0

    return {
        "quintile_returns": [round(x * 100, 3) for x in q_means],  # as percentage
        "long_short_pct": round((q_means[-1] - q_means[0]) * 100, 3),
        "monotonicity": round(float(mono), 3) if not np.isnan(mono) else 0.0,
        "n_days": n_days,
    }
