"""Portfolio turnover analysis for factor strategies."""
import numpy as np
import pandas as pd


def compute_turnover(factor_values: pd.Series, dates: pd.Series,
                     sym_col_values: pd.Series, top_pct: float = 0.2) -> dict:
    """
    Daily turnover of top-quintile portfolio.

    For each day, take the top ``top_pct`` fraction of stocks by factor value.
    Turnover = 1 - |intersection with yesterday's top group| / |top group|

    Parameters
    ----------
    factor_values : pd.Series
        Factor scores, aligned with *dates* and *sym_col_values*.
    dates : pd.Series
        Trading dates corresponding to each row.
    sym_col_values : pd.Series
        Symbol identifiers corresponding to each row.
    top_pct : float
        Fraction of stocks to include in the top group (default 0.2 = top quintile).

    Returns
    -------
    dict
        {
            "daily_turnover": list of daily turnover values,
            "avg_daily": float,
            "avg_monthly": float (avg_daily * 21),
            "n_days": int,
        }
    """
    df = pd.DataFrame({
        "date": dates.values,
        "factor": factor_values.values,
        "symbol": sym_col_values.values,
    }).dropna(subset=["factor"])

    sorted_dates = sorted(df["date"].unique())
    prev_top: set | None = None
    daily_turnover: list[float] = []

    for dt in sorted_dates:
        day = df[df["date"] == dt]
        if len(day) < 10:
            prev_top = None
            continue

        n_top = max(1, int(len(day) * top_pct))
        top_syms = set(day.nlargest(n_top, "factor")["symbol"])

        if prev_top is not None and len(top_syms) > 0:
            overlap = len(top_syms & prev_top)
            turnover = 1.0 - overlap / len(top_syms)
            daily_turnover.append(turnover)

        prev_top = top_syms

    if len(daily_turnover) == 0:
        return {
            "daily_turnover": [],
            "avg_daily": 0.0,
            "avg_monthly": 0.0,
            "n_days": 0,
        }

    avg_daily = float(np.mean(daily_turnover))
    return {
        "daily_turnover": [round(t, 4) for t in daily_turnover],
        "avg_daily": round(avg_daily, 4),
        "avg_monthly": round(avg_daily * 21, 4),
        "n_days": len(daily_turnover),
    }
