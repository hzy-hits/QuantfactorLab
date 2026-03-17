"""Information Coefficient (IC) analysis."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_ic_series(factor_values: pd.Series, forward_returns: pd.Series,
                      dates: pd.Series) -> pd.DataFrame:
    """
    Compute daily IC (rank correlation of factor vs forward return).

    Returns DataFrame with columns: [date, ic, n_stocks]
    """
    df = pd.DataFrame({
        "date": dates,
        "factor": factor_values,
        "fwd_ret": forward_returns,
    }).dropna()

    results = []
    for dt, group in df.groupby("date"):
        if len(group) < 10:  # need enough stocks for meaningful correlation
            continue
        ic, _ = spearmanr(group["factor"], group["fwd_ret"])
        results.append({"date": dt, "ic": ic, "n_stocks": len(group)})

    return pd.DataFrame(results)


def ic_summary(ic_series: pd.DataFrame) -> dict:
    """Compute IC summary statistics."""
    ics = ic_series["ic"].dropna()
    if len(ics) == 0:
        return {"ic_mean": 0, "ic_std": 0, "ic_ir": 0, "ic_positive_pct": 0, "n_days": 0}

    return {
        "ic_mean": round(float(ics.mean()), 4),
        "ic_std": round(float(ics.std()), 4),
        "ic_ir": round(float(ics.mean() / ics.std()) if ics.std() > 1e-10 else 0, 3),
        "ic_positive_pct": round(float((ics > 0).mean()) * 100, 1),
        "n_days": len(ics),
    }


def ic_by_regime(factor_values: pd.Series, forward_returns: pd.Series,
                 dates: pd.Series, regimes: pd.Series) -> dict[str, dict]:
    """Compute IC grouped by regime (0=trending, 1=MR, 2=noisy)."""
    df = pd.DataFrame({
        "date": dates,
        "factor": factor_values,
        "fwd_ret": forward_returns,
        "regime": regimes,
    }).dropna()

    regime_labels = {0.0: "trending", 1.0: "mean_reverting", 2.0: "noisy"}
    results = {}

    for regime_val, label in regime_labels.items():
        sub = df[df["regime"] == regime_val]
        if len(sub) < 20:
            results[label] = {"ic_mean": None, "ic_ir": None, "n_obs": len(sub)}
            continue

        ic_vals = []
        for dt, group in sub.groupby("date"):
            if len(group) < 10:
                continue
            ic, _ = spearmanr(group["factor"], group["fwd_ret"])
            ic_vals.append(ic)

        if len(ic_vals) == 0:
            results[label] = {"ic_mean": None, "ic_ir": None, "n_obs": 0}
            continue

        arr = np.array(ic_vals)
        results[label] = {
            "ic_mean": round(float(arr.mean()), 4),
            "ic_ir": round(float(arr.mean() / arr.std()) if arr.std() > 1e-10 else 0, 3),
            "n_obs": len(ic_vals),
        }

    return results
