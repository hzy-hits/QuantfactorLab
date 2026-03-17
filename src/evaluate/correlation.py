"""Factor correlation analysis — detect redundant factors."""
import numpy as np
import pandas as pd


def factor_correlation_matrix(factor_df: pd.DataFrame,
                              factor_cols: list[str],
                              date_col: str = "date") -> pd.DataFrame:
    """
    Compute average cross-sectional rank correlation between all factor pairs.

    For each day, compute rank corr between every pair of factors.
    Average across all days.
    """
    corr_sums = np.zeros((len(factor_cols), len(factor_cols)))
    n_days = 0

    for dt, group in factor_df.groupby(date_col):
        if len(group) < 20:
            continue

        ranks = group[factor_cols].rank(method="average")
        day_corr = ranks.corr(method="spearman").values

        if not np.any(np.isnan(day_corr)):
            corr_sums += day_corr
            n_days += 1

    if n_days == 0:
        return pd.DataFrame(np.eye(len(factor_cols)),
                            index=factor_cols, columns=factor_cols)

    avg_corr = corr_sums / n_days
    return pd.DataFrame(avg_corr, index=factor_cols, columns=factor_cols).round(3)


def find_redundant_pairs(corr_matrix: pd.DataFrame,
                         threshold: float = 0.7) -> list[tuple[str, str, float]]:
    """Find factor pairs with correlation above threshold."""
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) >= threshold:
                pairs.append((cols[i], cols[j], round(float(corr), 3)))
    return sorted(pairs, key=lambda x: -abs(x[2]))
