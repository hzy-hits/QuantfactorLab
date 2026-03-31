from __future__ import annotations

import pickle
import shutil
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from src.paths import FACTOR_LAB_ROOT, QUANT_CN_DB, QUANT_US_DB

CACHE_DIR = FACTOR_LAB_ROOT / "data" / ".cache"

MARKET_CONFIGS = {
    "cn": {
        "db_path": QUANT_CN_DB,
        "cache_path": CACHE_DIR / "cn_prices.pkl",
        "fwd_path": CACHE_DIR / "cn_fwd.pkl",
        "sym_col": "ts_code",
        "date_col": "trade_date",
        "price_sql": """
            SELECT p.ts_code, p.trade_date, p.open, p.high, p.low, p.close,
                   p.pre_close, p.change, p.pct_chg,
                   p.vol AS volume, p.amount, p.adj_factor,
                   db.turnover_rate, db.volume_ratio, db.pe_ttm, db.pb, db.ps_ttm,
                   db.total_mv AS market_cap, db.circ_mv AS circ_market_cap
            FROM prices p
            LEFT JOIN daily_basic db
                ON p.ts_code = db.ts_code AND p.trade_date = db.trade_date
            WHERE p.close > 0
            ORDER BY p.ts_code, p.trade_date
        """,
    },
    "us": {
        "db_path": QUANT_US_DB,
        "cache_path": CACHE_DIR / "us_prices.pkl",
        "fwd_path": CACHE_DIR / "us_fwd.pkl",
        "sym_col": "symbol",
        "date_col": "date",
        "price_sql": """
            SELECT symbol, date, open, high, low, adj_close AS close, volume
            FROM prices_daily
            WHERE adj_close > 0
            ORDER BY symbol, date
        """,
    },
}


def _load_prices_from_db(db_path: Path, price_sql: str) -> pd.DataFrame:
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            return con.execute(price_sql).fetchdf()
        finally:
            con.close()
    except Exception:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = Path(tmpdir) / db_path.name
            shutil.copy2(db_path, tmp_db)
            con = duckdb.connect(str(tmp_db), read_only=True)
            try:
                return con.execute(price_sql).fetchdf()
            finally:
                con.close()


def _is_stale(source_path: Path, cache_path: Path) -> bool:
    if not cache_path.exists():
        return True
    return source_path.stat().st_mtime > cache_path.stat().st_mtime


def _write_pickle_safely(path: Path, payload: object) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(payload, fh)
    except OSError:
        pass


def load_market_prices(
    market: str,
    *,
    refresh_if_stale: bool = True,
) -> pd.DataFrame:
    cfg = MARKET_CONFIGS[market]
    cache_path = cfg["cache_path"]
    db_path = cfg["db_path"]

    if cache_path.exists() and (not refresh_if_stale or not _is_stale(db_path, cache_path)):
        with cache_path.open("rb") as fh:
            return pickle.load(fh)

    prices = _load_prices_from_db(db_path, cfg["price_sql"])
    _write_pickle_safely(cache_path, prices)
    return prices


def load_forward_returns(
    market: str,
    *,
    prices: pd.DataFrame | None = None,
    refresh_if_stale: bool = True,
) -> pd.DataFrame:
    cfg = MARKET_CONFIGS[market]
    fwd_path = cfg["fwd_path"]
    db_path = cfg["db_path"]
    sym_col = cfg["sym_col"]
    date_col = cfg["date_col"]

    if fwd_path.exists() and (not refresh_if_stale or not _is_stale(db_path, fwd_path)):
        with fwd_path.open("rb") as fh:
            return pickle.load(fh)

    prices = prices if prices is not None else load_market_prices(market, refresh_if_stale=refresh_if_stale)
    prices_sorted = prices.sort_values([sym_col, date_col]).copy()
    prices_sorted["fwd_5d"] = (
        prices_sorted.groupby(sym_col)["close"].shift(-5) / prices_sorted["close"] - 1
    )
    fwd = prices_sorted[[sym_col, date_col, "fwd_5d"]].dropna().reset_index(drop=True)

    _write_pickle_safely(fwd_path, fwd)
    return fwd
