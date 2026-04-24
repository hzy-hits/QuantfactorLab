"""Signal analysis toolkit: Wavelet, FFT, Phase Space for factor health monitoring.

Replaces SigReg IC health check with physics-based tools:
- Wavelet: multi-scale energy decomposition of IC time series
- FFT: dominant periods for optimal hold/rebalance
- Phase Space: Takens embedding for predictability assessment
"""
from __future__ import annotations

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import pdist

try:
    import pywt
except ModuleNotFoundError:  # pragma: no cover - depends on runtime image
    pywt = None


def wavelet_health(ic_series: np.ndarray, scales: list[int] | None = None,
                   recent_window: int = 60) -> dict:
    """Multi-scale wavelet energy analysis of factor IC.

    Returns energy at each time scale, plus recent vs historical comparison.
    A factor is "weakening" when recent energy drops significantly.
    """
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[~np.isnan(ic)]
    if len(ic) < recent_window + 20:
        return {"health": "insufficient_data"}

    if scales is None:
        scales = [3, 5, 10, 20, 40]

    if pywt is None:
        return {
            "overall": "unavailable",
            "reason": "PyWavelets missing; skipped wavelet energy check",
            "scales": {},
            "weakening_scales": 0,
            "total_scales": len(scales),
        }

    coeff, _ = pywt.cwt(ic, scales, "morl", sampling_period=1)
    energy = np.abs(coeff).mean(axis=1)
    recent_e = np.abs(coeff[:, -recent_window:]).mean(axis=1)
    hist_e = np.abs(coeff[:, :-recent_window]).mean(axis=1)

    scale_health = {}
    weakening_count = 0
    for s, e, r, h in zip(scales, energy, recent_e, hist_e):
        pct_change = (r / (h + 1e-12) - 1) * 100
        if pct_change < -30:
            status = "weakening"
            weakening_count += 1
        elif pct_change > 30:
            status = "strengthening"
        else:
            status = "stable"
        scale_health[s] = {
            "energy": round(float(e), 5),
            "recent": round(float(r), 5),
            "historical": round(float(h), 5),
            "change_pct": round(pct_change, 1),
            "status": status,
        }

    # Overall health: if most scales are weakening, factor is degrading
    if weakening_count >= len(scales) * 0.6:
        overall = "degrading"
    elif weakening_count >= len(scales) * 0.3:
        overall = "caution"
    else:
        overall = "healthy"

    return {
        "overall": overall,
        "scales": scale_health,
        "weakening_scales": weakening_count,
        "total_scales": len(scales),
    }


def fft_periods(ic_series: np.ndarray, min_period: int = 2, max_period: int = 120,
                top_n: int = 3) -> dict:
    """Find dominant periods in factor IC via FFT.

    Returns top N periods and suggested hold/rebalance intervals.
    """
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[~np.isnan(ic)]
    if len(ic) < 30:
        return {"periods": [], "suggested_hold": 5}

    N = len(ic)
    yf = fft(ic - ic.mean())
    xf = fftfreq(N, d=1)
    power = 2.0 / N * np.abs(yf[: N // 2])
    freqs = xf[: N // 2]

    mask = (freqs > 1 / max_period) & (freqs < 1 / min_period)
    if not mask.any():
        return {"periods": [], "suggested_hold": 5}

    masked_power = power[mask]
    masked_freqs = freqs[mask]
    peak_idx = np.argsort(masked_power)[-top_n:][::-1]

    periods = []
    for idx in peak_idx:
        p = 1.0 / masked_freqs[idx]
        periods.append({
            "period_days": round(p, 1),
            "power": round(float(masked_power[idx]), 6),
        })

    # Suggested hold = half of dominant period (Nyquist)
    dominant = periods[0]["period_days"] if periods else 10
    suggested_hold = max(3, min(20, int(dominant / 2)))

    return {
        "periods": periods,
        "dominant_period": round(dominant, 0),
        "suggested_hold": suggested_hold,
    }


def ic_autocorrelation(ic_series: np.ndarray, max_lag: int = 40) -> dict:
    """IC autocorrelation analysis for signal persistence.

    Returns autocorrelation at each lag and the half-life.
    """
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[~np.isnan(ic)]
    if len(ic) < max_lag + 10:
        return {"halflife": 1, "autocorr": {}}

    # Normalized autocorrelation
    ic_centered = ic - ic.mean()
    ac_full = np.correlate(ic_centered, ic_centered, mode="full")
    ac = ac_full[len(ic) - 1 :] / (ac_full[len(ic) - 1] + 1e-12)

    autocorr = {}
    for lag in [1, 2, 3, 5, 10, 20, 40]:
        if lag < len(ac):
            autocorr[lag] = round(float(ac[lag]), 4)

    # Half-life: first lag where autocorr drops below 50% of lag-1
    threshold = 0.5 * ac[1] if len(ac) > 1 and ac[1] > 0 else 0.05
    halflife = next(
        (lag for lag in range(1, min(max_lag, len(ac))) if ac[lag] < threshold),
        max_lag,
    )

    return {
        "halflife": halflife,
        "suggested_rebalance": max(3, halflife),
        "autocorr": autocorr,
    }


def phase_space_dimension(ic_series: np.ndarray, tau: int = 5, dim: int = 3,
                          max_points: int = 300) -> dict:
    """Phase space reconstruction via Takens embedding.

    Estimates correlation dimension to assess if IC is deterministic or random.
    dim < 2: periodic/predictable
    dim 2-2.5: low-dimensional chaos (some structure)
    dim > 2.5: high-dimensional / random
    """
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[~np.isnan(ic)]
    n = len(ic)
    if n < tau * dim + max_points:
        return {"dimension": 3.0, "verdict": "insufficient_data"}

    embedded = np.column_stack(
        [ic[i * tau : n - (dim - 1 - i) * tau] for i in range(dim)]
    )
    # Subsample for speed
    if len(embedded) > max_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(embedded), max_points, replace=False)
        embedded = embedded[idx]

    dists = pdist(embedded)
    if len(dists) == 0:
        return {"dimension": 3.0, "verdict": "degenerate"}

    eps_range = np.percentile(dists, [5, 10, 20, 30, 50])
    dim_estimates = []
    for i in range(len(eps_range) - 1):
        C1 = (dists < eps_range[i]).sum() / len(dists)
        C2 = (dists < eps_range[i + 1]).sum() / len(dists)
        if C1 > 0 and C2 > 0:
            d = np.log(C2 / C1) / np.log(eps_range[i + 1] / eps_range[i])
            dim_estimates.append(d)

    avg_dim = float(np.mean(dim_estimates)) if dim_estimates else 3.0

    if avg_dim < 1.5:
        verdict = "periodic"
    elif avg_dim < 2.0:
        verdict = "low_dim_chaos"
    elif avg_dim < 2.5:
        verdict = "moderate_complexity"
    else:
        verdict = "random"

    return {
        "dimension": round(avg_dim, 2),
        "verdict": verdict,
        "predictable": avg_dim < 2.5,
    }


def full_diagnosis(ic_series: np.ndarray) -> dict:
    """Run all analyses and produce a unified health report."""
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[~np.isnan(ic)]

    wavelet = wavelet_health(ic)
    periods = fft_periods(ic)
    autocorr = ic_autocorrelation(ic)
    phase = phase_space_dimension(ic)

    # Unified recommendation
    suggested_hold = min(
        periods.get("suggested_hold", 5),
        autocorr.get("suggested_rebalance", 5),
    )

    if wavelet.get("overall") == "degrading":
        action = "EXIT"
        reason = f"小波能量全面衰减 ({wavelet['weakening_scales']}/{wavelet['total_scales']} 尺度在走弱)"
    elif wavelet.get("overall") == "caution":
        action = "REDUCE"
        reason = "部分尺度能量下降"
    else:
        action = "HOLD"
        reason = "因子能量稳定"

    return {
        "action": action,
        "reason": reason,
        "suggested_hold_days": suggested_hold,
        "ic_halflife": autocorr.get("halflife", 5),
        "dominant_period": periods.get("dominant_period", 10),
        "phase_dimension": phase.get("dimension", 3.0),
        "predictable": phase.get("predictable", False),
        "wavelet": wavelet,
        "fft": periods,
        "autocorr": autocorr,
        "phase_space": phase,
    }
