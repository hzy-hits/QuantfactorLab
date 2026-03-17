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
    n_factors = len(factor_cols)
    corr_sums = np.zeros((n_factors, n_factors))
    pair_counts = np.zeros((n_factors, n_factors))

    for dt, group in factor_df.groupby(date_col):
        if len(group) < 20:
            continue

        for i in range(n_factors):
            for j in range(i, n_factors):
                if i == j:
                    corr_sums[i, j] += 1.0
                    pair_counts[i, j] += 1
                    continue
                pair = group[[factor_cols[i], factor_cols[j]]].dropna()
                if len(pair) < 10:
                    continue
                rho = pair.rank(method="average").corr(method="spearman").iloc[0, 1]
                if not np.isnan(rho):
                    corr_sums[i, j] += rho
                    corr_sums[j, i] += rho
                    pair_counts[i, j] += 1
                    pair_counts[j, i] += 1

    # Fill diagonal counts
    for i in range(n_factors):
        if pair_counts[i, i] == 0:
            pair_counts[i, i] = 1  # avoid division by zero on diagonal

    # Average: element-wise divide, default to 0 where no data
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_corr = np.where(pair_counts > 0, corr_sums / pair_counts, 0.0)

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
