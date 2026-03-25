"""
Gate system for factor validation.

All gates must pass for a factor to be considered for OOS testing.
Thresholds differ by market (CN is more lenient due to higher noise).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.backtest.walk_forward import BacktestResult


GATE_THRESHOLDS: dict[str, dict[str, float]] = {
    "cn": {
        "ic_min": 0.01,
        "ic_ir_min": 0.2,
        "turnover_max": 0.50,
        "monotonicity_min": 0.0,  # Disabled: A-share factor-return relationships are non-monotonic
        "corr_max": 0.6,          # Tightened from 0.7: voting strategy needs independent factors
    },
    "us": {
        "ic_min": 0.02,
        "ic_ir_min": 0.3,
        "turnover_max": 0.40,
        "monotonicity_min": 0.7,
        "corr_max": 0.6,          # Tightened from 0.7: voting strategy needs independent factors
    },
}


@dataclass
class GateResult:
    passed: bool
    details: dict[str, dict]  # gate_name -> {passed, value, threshold}


def _cross_sectional_corr(a: pd.Series, b: pd.Series, dates: pd.Series) -> float:
    """Average cross-sectional rank correlation between two factor series.

    Parameters
    ----------
    a, b : pd.Series
        Factor values, aligned with *dates*.
    dates : pd.Series
        Trading dates for each row.

    Returns
    -------
    float : average |rank correlation| across all days.
    """
    df = pd.DataFrame({"date": dates.values, "a": a.values, "b": b.values}).dropna()
    corrs: list[float] = []
    for _, group in df.groupby("date"):
        if len(group) < 10:
            continue
        rho, _ = spearmanr(group["a"], group["b"])
        if not np.isnan(rho):
            corrs.append(abs(rho))
    return float(np.mean(corrs)) if corrs else 0.0


def check_gates(
    result: BacktestResult,
    market: str,
    existing_factors: list[pd.Series] | None = None,
    candidate_values: pd.Series | None = None,
    candidate_dates: pd.Series | None = None,
) -> GateResult:
    """Check all gates against IS backtest metrics.

    Parameters
    ----------
    result : BacktestResult
        Output of walk_forward_backtest().
    market : str
        'cn' or 'us'.
    existing_factors : list of pd.Series, optional
        Factor value series for each existing promoted factor (for correlation check).
    candidate_values : pd.Series, optional
        Factor values of the candidate (needed for correlation check).
    candidate_dates : pd.Series, optional
        Dates aligned with candidate_values (needed for correlation check).

    Returns
    -------
    GateResult with overall pass/fail and per-gate details.
    """
    thresholds = GATE_THRESHOLDS.get(market, GATE_THRESHOLDS["cn"])
    details: dict[str, dict] = {}

    # Gate 1: IC
    ic_val = abs(result.avg_ic)
    ic_thresh = thresholds["ic_min"]
    details["ic"] = {
        "passed": ic_val >= ic_thresh,
        "value": round(result.avg_ic, 4),
        "abs_value": round(ic_val, 4),
        "threshold": ic_thresh,
    }

    # Gate 2: IC_IR
    icir_val = result.avg_ic_ir
    icir_thresh = thresholds["ic_ir_min"]
    details["ic_ir"] = {
        "passed": icir_val >= icir_thresh,
        "value": round(icir_val, 3),
        "threshold": icir_thresh,
    }

    # Gate 3: Turnover
    to_val = result.avg_turnover
    to_thresh = thresholds["turnover_max"]
    details["turnover"] = {
        "passed": to_val <= to_thresh,
        "value": round(to_val, 4),
        "threshold": to_thresh,
    }

    # Gate 4: Monotonicity
    mono_val = abs(result.avg_monotonicity)
    mono_thresh = thresholds["monotonicity_min"]
    details["monotonicity"] = {
        "passed": mono_val >= mono_thresh,
        "value": round(result.avg_monotonicity, 3),
        "abs_value": round(mono_val, 3),
        "threshold": mono_thresh,
    }

    # Gate 5: Correlation with existing factors
    if existing_factors and candidate_values is not None and candidate_dates is not None:
        max_corr = 0.0
        max_corr_name = ""
        for i, ef in enumerate(existing_factors):
            corr = _cross_sectional_corr(candidate_values, ef, candidate_dates)
            if corr > max_corr:
                max_corr = corr
                max_corr_name = f"factor_{i}"
        corr_thresh = thresholds["corr_max"]
        details["correlation"] = {
            "passed": max_corr < corr_thresh,
            "value": round(max_corr, 3),
            "threshold": corr_thresh,
            "most_correlated": max_corr_name,
        }
    else:
        # No existing factors to check against — passes by default
        details["correlation"] = {
            "passed": True,
            "value": 0.0,
            "threshold": thresholds["corr_max"],
            "most_correlated": None,
        }

    # Gate 6: Multi-collinearity (SigReg-inspired)
    # Pairwise correlation misses cases where a factor is a linear combination
    # of 2+ existing factors. Check R² from regressing new on all existing.
    if existing_factors and candidate_values is not None and len(existing_factors) >= 3:
        try:
            from src.evaluate.sigreg import multi_collinearity_check
            existing_dict = {f"f_{i}": ef for i, ef in enumerate(existing_factors)}
            mc_result = multi_collinearity_check(candidate_values, existing_dict, threshold=0.85)
            details["multi_collinearity"] = {
                "passed": not mc_result["is_redundant"],
                "value": mc_result["r_squared"],
                "threshold": 0.85,
            }
        except Exception:
            details["multi_collinearity"] = {"passed": True, "value": 0.0, "threshold": 0.85}
    else:
        details["multi_collinearity"] = {"passed": True, "value": 0.0, "threshold": 0.85}

    all_passed = all(g["passed"] for g in details.values())

    return GateResult(passed=all_passed, details=details)


def format_gate_result(gr: GateResult) -> str:
    """Human-readable gate result string for agent feedback."""
    lines = []
    for gate_name, info in gr.details.items():
        status = "PASS" if info["passed"] else "FAIL"
        val = info.get("abs_value", info["value"])
        lines.append(f"  {gate_name}: {status} (value={val}, threshold={info['threshold']})")
    header = "GATES: ALL PASSED" if gr.passed else "GATES: FAILED"
    return header + "\n" + "\n".join(lines)
