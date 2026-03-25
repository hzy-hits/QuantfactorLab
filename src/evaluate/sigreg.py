"""SigReg-inspired factor quality checks.

Applies ideas from Sketched Isotropic Gaussian Regularization (SigReg,
Balestriero & LeCun 2025) to quantitative factor evaluation:

1. factor_diversity_score: Are promoted factors spanning diverse signal space,
   or collapsing into a few correlated clusters? Uses random projections +
   Epps-Pulley normality test (Cramer-Wold theorem).

2. multi_collinearity_check: Does a new factor add genuine information beyond
   what existing factors already capture? Goes beyond pairwise correlation to
   detect redundancy from linear combinations of 2+ factors.

3. ic_health_test: Is a factor's IC time series still well-behaved, or showing
   signs of regime change / decay? Uses Epps-Pulley on rolling IC windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _epps_pulley_statistic(x: np.ndarray) -> float:
    """Epps-Pulley test statistic for normality.

    Tests H0: x ~ N(mu, sigma^2) using the characteristic function approach.
    Higher values → stronger departure from normality.

    Returns the test statistic T (not p-value). For standard normal,
    E[T] ≈ 0 under H0. Values > 1.0 suggest significant non-normality.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return 0.0

    # Standardize
    mu, sigma = x.mean(), x.std()
    if sigma < 1e-12:
        return 999.0  # degenerate = maximally non-normal
    z = (x - mu) / sigma

    # Epps-Pulley: compare empirical CF with standard normal CF
    # at a grid of frequency points
    t_grid = np.linspace(0.1, 3.0, 20)

    T = 0.0
    for t in t_grid:
        # Empirical characteristic function: (1/n) * sum(exp(i*t*z_j))
        cf_emp = np.mean(np.exp(1j * t * z))
        # Standard normal CF: exp(-t^2/2)
        cf_norm = np.exp(-t ** 2 / 2)
        # Squared difference, weighted by Gaussian kernel
        w = np.exp(-t ** 2 / 2)
        T += n * abs(cf_emp - cf_norm) ** 2 * w

    return float(T / len(t_grid))


def factor_diversity_score(
    factor_values: dict[str, pd.Series],
    n_projections: int = 200,
    seed: int = 42,
) -> dict:
    """Measure how diverse the promoted factor set is.

    Uses SigReg's approach: project the N-factor space onto random directions
    and test if projections are isotropic (well-spread) vs collapsed (clustered).

    Parameters
    ----------
    factor_values : dict of {factor_id: pd.Series}
        Factor values keyed by stock symbol, for a single date.
    n_projections : int
        Number of random 1D projections (SigReg default: 1024, we use fewer).

    Returns
    -------
    dict with:
        diversity_score: float in [0, 1]. 1 = perfectly diverse, 0 = all collapsed.
        n_effective: float. Effective number of independent factors.
        cluster_warning: bool. True if factors are clustered.
        details: dict with per-projection stats.
    """
    # Build factor matrix: rows = stocks, columns = factors
    all_syms = set()
    for fv in factor_values.values():
        all_syms.update(fv.index)

    factor_names = list(factor_values.keys())
    n_factors = len(factor_names)

    if n_factors < 3:
        return {"diversity_score": 1.0, "n_effective": n_factors,
                "cluster_warning": False, "details": {}}

    # Build matrix
    mat = np.full((len(all_syms), n_factors), np.nan)
    sym_list = sorted(all_syms)
    sym_idx = {s: i for i, s in enumerate(sym_list)}

    for j, fname in enumerate(factor_names):
        fv = factor_values[fname]
        for sym, val in fv.items():
            if sym in sym_idx and not np.isnan(val):
                mat[sym_idx[sym], j] = val

    # Drop rows with any NaN
    mask = ~np.any(np.isnan(mat), axis=1)
    mat = mat[mask]

    if len(mat) < 50:
        return {"diversity_score": 0.5, "n_effective": n_factors,
                "cluster_warning": False, "details": {"error": "too few common stocks"}}

    # Standardize each column
    mat = (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-12)

    # Random projections (Cramer-Wold: test isotropy via 1D slices)
    rng = np.random.RandomState(seed)
    directions = rng.randn(n_projections, n_factors)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    ep_stats = []
    for d in directions:
        projection = mat @ d  # 1D projection
        T = _epps_pulley_statistic(projection)
        ep_stats.append(T)

    ep_stats = np.array(ep_stats)

    # Diversity score: low EP stats = projections look Gaussian = isotropic = diverse
    # High EP stats = projections non-Gaussian = collapsed/clustered
    mean_ep = np.mean(ep_stats)

    # Empirical mapping: EP < 0.5 → diverse, EP > 2.0 → collapsed
    diversity_score = float(np.clip(1.0 - mean_ep / 2.0, 0.0, 1.0))

    # Effective number of independent factors via eigenvalue decomposition
    cov = np.corrcoef(mat.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0.01]
    # Shannon entropy of normalized eigenvalues → effective dimensionality
    p = eigenvalues / eigenvalues.sum()
    n_effective = float(np.exp(-np.sum(p * np.log(p + 1e-12))))

    cluster_warning = n_effective < n_factors * 0.5  # less than half are independent

    return {
        "diversity_score": round(diversity_score, 3),
        "n_effective": round(n_effective, 1),
        "n_total": n_factors,
        "cluster_warning": cluster_warning,
        "mean_ep_stat": round(mean_ep, 3),
        "details": {
            "ep_median": round(float(np.median(ep_stats)), 3),
            "ep_p95": round(float(np.percentile(ep_stats, 95)), 3),
        },
    }


def multi_collinearity_check(
    new_factor: pd.Series,
    existing_factors: dict[str, pd.Series],
    threshold: float = 0.85,
) -> dict:
    """Check if a new factor is a linear combination of existing ones.

    Goes beyond pairwise correlation: uses R² from regressing the new factor
    on ALL existing factors simultaneously.

    Parameters
    ----------
    new_factor : pd.Series
        Factor values keyed by symbol.
    existing_factors : dict of {name: pd.Series}
    threshold : float
        R² above this → reject (new factor is redundant).

    Returns
    -------
    dict with r_squared, is_redundant, top_contributors.
    """
    if not existing_factors:
        return {"r_squared": 0.0, "is_redundant": False, "top_contributors": []}

    # Build aligned matrix
    common = set(new_factor.index)
    for fv in existing_factors.values():
        common &= set(fv.index)
    common = sorted(common)

    if len(common) < 50:
        return {"r_squared": 0.0, "is_redundant": False,
                "top_contributors": [], "error": "too few common stocks"}

    y = np.array([new_factor[s] for s in common])
    X = np.column_stack([
        [existing_factors[name][s] for s in common]
        for name in existing_factors
    ])

    # Drop NaN rows
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y, X = y[mask], X[mask]

    if len(y) < 50:
        return {"r_squared": 0.0, "is_redundant": False, "top_contributors": []}

    # Standardize
    y = (y - y.mean()) / (y.std() + 1e-12)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    # OLS regression
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    except np.linalg.LinAlgError:
        return {"r_squared": 0.0, "is_redundant": False, "top_contributors": []}

    # Top contributors
    names = list(existing_factors.keys())
    contributions = sorted(zip(names, np.abs(beta)), key=lambda x: -x[1])
    top = [(name, round(float(b), 3)) for name, b in contributions[:3]]

    return {
        "r_squared": round(r_squared, 3),
        "is_redundant": r_squared > threshold,
        "top_contributors": top,
    }


def ic_health_test(
    ic_series: list[float] | np.ndarray,
    window: int = 60,
) -> dict:
    """Test if a factor's IC time series is healthy using Epps-Pulley.

    A healthy factor has IC that looks like draws from a stable distribution.
    Regime changes, alpha decay, or structural breaks cause the distribution
    to become non-normal (heavy tails, bimodal, skewed).

    Parameters
    ----------
    ic_series : array-like
        Daily IC values (most recent last).
    window : int
        Rolling window for the test.

    Returns
    -------
    dict with health_score, regime_change_detected, details.
    """
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[~np.isnan(ic)]

    if len(ic) < window:
        return {"health_score": 0.5, "regime_change_detected": False,
                "details": {"error": "insufficient data"}}

    # Test recent window vs full history
    recent = ic[-window:]
    full = ic

    ep_recent = _epps_pulley_statistic(recent)
    ep_full = _epps_pulley_statistic(full)

    # Compare distributions: if recent is very different from full, regime changed
    recent_mean = recent.mean()
    full_mean = full.mean()
    recent_std = recent.std()
    full_std = full.std()

    # Z-score of recent mean vs full distribution
    drift_z = abs(recent_mean - full_mean) / (full_std / np.sqrt(window) + 1e-12)

    # Health scoring
    # Good: low EP (normal), low drift, positive mean
    is_positive = recent_mean > 0
    is_stable = ep_recent < 1.0
    no_drift = drift_z < 2.0

    health_score = 0.0
    if is_positive:
        health_score += 0.4
    if is_stable:
        health_score += 0.3
    if no_drift:
        health_score += 0.3

    regime_change = drift_z > 3.0 or (ep_recent > 2.0 and ep_full < 1.0)

    return {
        "health_score": round(health_score, 2),
        "regime_change_detected": regime_change,
        "ic_mean_recent": round(float(recent_mean), 4),
        "ic_mean_full": round(float(full_mean), 4),
        "drift_z": round(float(drift_z), 2),
        "ep_recent": round(ep_recent, 3),
        "ep_full": round(ep_full, 3),
        "is_positive": is_positive,
        "is_stable": is_stable,
    }
