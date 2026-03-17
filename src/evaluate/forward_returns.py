"""Compute forward returns for factor evaluation."""
import duckdb
import pandas as pd


def compute_forward_returns(db_path: str, table: str = "prices",
                            date_col: str = "trade_date", close_col: str = "close",
                            sym_col: str = "ts_code",
                            horizons: list[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Compute forward N-day returns for all symbols from a prices table.

    Returns DataFrame with columns: [sym_col, date_col, fwd_1d, fwd_5d, ...]
    """
    con = duckdb.connect(db_path, read_only=True)

    lead_cols = ", ".join(
        f"LEAD({close_col}, {h}) OVER w / {close_col} - 1 AS fwd_{h}d"
        for h in horizons
    )

    sql = f"""
        SELECT {sym_col}, {date_col}, {close_col}, {lead_cols}
        FROM {table}
        WHERE {close_col} > 0
        WINDOW w AS (PARTITION BY {sym_col} ORDER BY {date_col})
        ORDER BY {sym_col}, {date_col}
    """

    df = con.execute(sql).fetchdf()
    con.close()
    return df
