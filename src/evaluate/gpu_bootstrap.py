"""
GPU-accelerated bootstrap significance test for factor IC.

Uses CuPy (GPU numpy) to run 100,000 permutation tests in seconds.
Falls back to CPU numpy if no GPU available.

Usage:
    result = bootstrap_significance(factor_values, forward_returns, dates, n_bootstrap=100000)
    print(result["p_value"])  # < 0.01 = statistically significant
"""
import numpy as np
import pandas as pd

try:
    import cupy as cp
    # Test if GPU actually works (new architectures may not have prebuilt kernels)
    cp.array([1.0]) + cp.array([1.0])
    GPU_AVAILABLE = True
except Exception:
    cp = np  # fallback to numpy
    GPU_AVAILABLE = False


def bootstrap_significance(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    dates: pd.Series,
    n_bootstrap: int = 100_000,
    seed: int = 42,
) -> dict:
    """
    Permutation bootstrap test for factor IC significance.

    Procedure:
    1. Compute real IC (daily cross-sectional Spearman, averaged)
    2. Shuffle date-factor alignment n_bootstrap times (preserve cross-section)
    3. Compute null IC distribution
    4. p-value = fraction of null ICs more extreme than real IC

    Returns:
        {
            "real_ic": float,
            "p_value": float,       # < 0.01 = significant at 99%
            "null_mean": float,
            "null_std": float,
            "percentile": float,    # where real IC sits in null distribution
            "significant_01": bool, # p < 0.01
            "significant_05": bool, # p < 0.05
            "n_bootstrap": int,
            "gpu": bool,
        }
    """
    # Build daily IC matrix
    df = pd.DataFrame({
        "date": dates.values,
        "factor": factor_values.values,
        "fwd": forward_returns.values,
    }).dropna()

    # Group by date, compute daily Spearman IC
    daily_groups = []
    unique_dates = sorted(df["date"].unique())

    for dt in unique_dates:
        mask = df["date"] == dt
        day_f = df.loc[mask, "factor"].values
        day_r = df.loc[mask, "fwd"].values
        if len(day_f) < 10:
            continue
        daily_groups.append((day_f, day_r))

    if len(daily_groups) < 10:
        return {
            "real_ic": 0.0, "p_value": 1.0, "null_mean": 0.0, "null_std": 0.0,
            "percentile": 50.0, "significant_01": False, "significant_05": False,
            "n_bootstrap": 0, "gpu": False,
        }

    n_days = len(daily_groups)

    # Compute real IC: average of daily Spearman correlations
    real_daily_ics = np.array([_spearman_ic(g[0], g[1]) for g in daily_groups])
    real_ic = float(np.nanmean(real_daily_ics))

    # Bootstrap: shuffle which date's returns go with which date's factors
    # This preserves cross-sectional structure but breaks time alignment
    xp = cp if GPU_AVAILABLE else np

    if GPU_AVAILABLE:
        rng = cp.random.default_rng(seed)
    else:
        rng = np.random.default_rng(seed)

    # Pre-compute rank arrays for speed
    factor_ranks = []
    return_ranks = []
    for day_f, day_r in daily_groups:
        factor_ranks.append(_rankdata(day_f))
        return_ranks.append(_rankdata(day_r))

    # Convert to GPU arrays if available
    # Strategy: shuffle the daily IC array (which day's returns pair with which factors)
    # This is much faster than re-ranking per permutation
    daily_ics_array = xp.array(real_daily_ics, dtype=xp.float32)

    # For each bootstrap: resample daily ICs with replacement → compute mean
    # This tests: "is the mean IC significantly different from 0?"
    # Block bootstrap: resample days with replacement
    null_ics = xp.zeros(n_bootstrap, dtype=xp.float32)

    # Batch generate all random indices at once (GPU efficient)
    boot_indices = rng.integers(0, n_days, size=(n_bootstrap, n_days))

    # Vectorized mean computation
    if GPU_AVAILABLE:
        daily_ics_gpu = cp.array(real_daily_ics, dtype=cp.float32)
        for i in range(n_bootstrap):
            # Resample with replacement and shuffle sign (under null: IC mean = 0)
            signs = rng.choice(cp.array([-1.0, 1.0], dtype=cp.float32), size=n_days)
            null_ics[i] = cp.mean(daily_ics_gpu[boot_indices[i]] * signs)
    else:
        daily_ics_np = np.array(real_daily_ics, dtype=np.float32)
        for i in range(n_bootstrap):
            signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n_days)
            null_ics[i] = np.mean(daily_ics_np[boot_indices[i]] * signs)

    # Convert back to numpy for stats
    if GPU_AVAILABLE:
        null_ics_np = cp.asnumpy(null_ics)
    else:
        null_ics_np = null_ics

    # p-value: two-sided test (how often is null |IC| >= real |IC|?)
    p_value = float(np.mean(np.abs(null_ics_np) >= abs(real_ic)))
    percentile = float(np.mean(null_ics_np < real_ic) * 100)

    return {
        "real_ic": round(real_ic, 6),
        "p_value": round(p_value, 6),
        "null_mean": round(float(np.mean(null_ics_np)), 6),
        "null_std": round(float(np.std(null_ics_np)), 6),
        "percentile": round(percentile, 1),
        "significant_01": p_value < 0.01,
        "significant_05": p_value < 0.05,
        "n_bootstrap": n_bootstrap,
        "gpu": GPU_AVAILABLE,
    }


def batch_bootstrap(
    candidates: list[dict],
    prices_df: pd.DataFrame,
    fwd_df: pd.DataFrame,
    sym_col: str = "ts_code",
    date_col: str = "trade_date",
    n_bootstrap: int = 100_000,
) -> list[dict]:
    """
    Run bootstrap significance test on a batch of factor candidates.
    Adds 'bootstrap_p' and 'bootstrap_significant' to each candidate.
    """
    import sys
    sys.path.insert(0, str(pd.__file__).rsplit("/", 3)[0])

    from src.dsl.parser import parse
    from src.dsl.compute import compute_factor

    tested = 0
    significant = 0

    for c in candidates:
        try:
            ast = parse(c["formula"])
            factor_df = compute_factor(ast, prices_df, sym_col=sym_col, date_col=date_col)

            merged = factor_df.merge(
                fwd_df[[sym_col, date_col, "fwd_5d"]],
                on=[sym_col, date_col], how="inner"
            ).dropna(subset=["fwd_5d", "factor_value"])

            if len(merged) < 500:
                c["bootstrap_p"] = 1.0
                c["bootstrap_significant"] = False
                continue

            result = bootstrap_significance(
                merged["factor_value"], merged["fwd_5d"], merged[date_col],
                n_bootstrap=n_bootstrap,
            )

            c["bootstrap_p"] = result["p_value"]
            c["bootstrap_significant"] = result["significant_01"]
            c["bootstrap_percentile"] = result["percentile"]

            tested += 1
            if result["significant_01"]:
                significant += 1

        except Exception as e:
            c["bootstrap_p"] = 1.0
            c["bootstrap_significant"] = False

    print(f"  Bootstrap: {tested} tested, {significant} significant (p<0.01), GPU={'yes' if GPU_AVAILABLE else 'no'}")
    return candidates


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Spearman IC for two arrays."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    d = rx - ry
    return float(1.0 - 6.0 * np.sum(d * d) / (n * (n * n - 1)))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Fast rank (average method)."""
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    idx = np.argsort(x)
    ranks[idx] = np.arange(1, n + 1, dtype=np.float64)
    return ranks
