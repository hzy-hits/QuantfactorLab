"""Paper portfolio tracker — record daily picks, evaluate returns, report performance.

No broker connection needed. Just tracks which stocks the composite would select
and measures their actual returns the next day.
"""
from __future__ import annotations

import shutil
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.market_data import load_market_prices
from src.paths import FACTOR_LAB_DB, QUANT_US_DB

US_DB = str(QUANT_US_DB)


def _open_us_db_readonly() -> duckdb.DuckDBPyConnection:
    """Open US DB read-only. If locked, copy to temp file first.

    DuckDB is single-writer: even read_only=True fails when another process
    holds the write lock. Copying the file bypasses this entirely.
    Temp files are cleaned up at process exit.
    """
    try:
        return duckdb.connect(US_DB, read_only=True)
    except Exception:
        import atexit, os
        tmp = Path(tempfile.mkdtemp()) / "quant_readonly.duckdb"
        shutil.copy2(US_DB, tmp)
        atexit.register(lambda p=str(tmp): os.unlink(p) if os.path.exists(p) else None)
        return duckdb.connect(str(tmp), read_only=True)

N_PICKS = 10  # stocks to hold


def init_tables():
    """Create paper tracking tables if they don't exist."""
    con = duckdb.connect(FACTOR_LAB_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_picks (
            as_of DATE NOT NULL,
            side VARCHAR NOT NULL,
            rank INTEGER NOT NULL,
            symbol VARCHAR NOT NULL,
            composite_score DOUBLE,
            PRIMARY KEY (as_of, side, rank)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_returns (
            as_of DATE NOT NULL PRIMARY KEY,
            eval_date DATE NOT NULL,
            long_ret DOUBLE,
            short_ret DOUBLE,
            ls_ret DOUBLE,
            spy_ret DOUBLE,
            long_excess DOUBLE,
            n_long INTEGER,
            n_short INTEGER,
            cum_long DOUBLE,
            cum_spy DOUBLE,
            cum_excess DOUBLE
        )
    """)
    con.close()


def _get_latest_trade_date() -> str:
    """Get the most recent trade date in US prices_daily."""
    con = _open_us_db_readonly()
    r = con.execute("SELECT MAX(date) FROM prices_daily").fetchone()
    con.close()
    return str(r[0])


def _get_prev_trade_date(dt: str) -> str | None:
    """Get the trading date before dt."""
    con = _open_us_db_readonly()
    r = con.execute(
        "SELECT MAX(date) FROM prices_daily WHERE date < ?", [dt]
    ).fetchone()
    con.close()
    return str(r[0]) if r[0] else None


def record(as_of: str | None = None, n: int = N_PICKS):
    """Record today's picks using rolling best-factor strategy.

    Directly calls the strategy module instead of reading from pipeline DB.
    """
    init_tables()

    if as_of is None:
        as_of = _get_latest_trade_date()

    # Use rolling strategy directly
    try:
        from src.dsl.parser import parse
        from src.dsl.compute import compute_factor
        from src.strategy.rolling_best import select_best_factor, StrategyConfig
        import duckdb as _ddb

        prices = load_market_prices("us")
        sym_col, date_col = "symbol", "date"
        prices = prices.sort_values([sym_col, date_col])
        prices["ret_next"] = prices.groupby(sym_col)["close"].transform(lambda x: x.shift(-1) / x - 1)
        for h in [5, 10, 20]:
            prices[f"ret_{h}d"] = prices.groupby(sym_col)["close"].transform(lambda x: x.shift(-h) / x - 1)

        con = _ddb.connect(str(FACTOR_LAB_DB), read_only=True)
        promoted = con.execute("SELECT name, formula FROM factor_registry WHERE market='us' AND status='promoted'").fetchdf()
        con.close()

        all_factors = {}
        for _, row in promoted.iterrows():
            try:
                ast = parse(row["formula"])
                vals = compute_factor(ast, prices, sym_col=sym_col, date_col=date_col)
                merged = vals.merge(prices[[sym_col, date_col, "ret_next", "ret_20d"]], on=[sym_col, date_col]).dropna(subset=["factor_value"])
                all_factors[row["name"]] = merged
            except Exception:
                pass

        dates = sorted(prices[date_col].unique())
        latest = dates[-1]
        cfg = StrategyConfig(lookback=40, hold_max=5, n_picks=n)
        lb_dates = dates[-cfg.lookback - 1:-1]

        factor_name, side, sharpe = select_best_factor(all_factors, lb_dates, cfg.hold_max, date_col, n)

        if factor_name is None:
            print(f"  No factor selected for {as_of}")
            return

        fdata = all_factors[factor_name]
        today = fdata[fdata[date_col] == latest].dropna(subset=["factor_value"])
        if side == "top":
            picks = today.nlargest(n, "factor_value")
        else:
            picks = today.nsmallest(n, "factor_value")

        bottom = picks[[sym_col, "factor_value"]].rename(columns={sym_col: "symbol", "factor_value": "score"}).reset_index(drop=True)
        # For paper tracking, "long" = strategy picks, "short" = opposite end
        if side == "top":
            top = today.nsmallest(n, "factor_value")[[sym_col, "factor_value"]].rename(columns={sym_col: "symbol", "factor_value": "score"}).reset_index(drop=True)
        else:
            top = today.nlargest(n, "factor_value")[[sym_col, "factor_value"]].rename(columns={sym_col: "symbol", "factor_value": "score"}).reset_index(drop=True)

        print(f"  Strategy: {factor_name} ({side}), lookback_sharpe={sharpe:.2f}")

    except Exception as e:
        print(f"  Rolling strategy failed ({e}), skipping")
        return

    # Write picks
    con = duckdb.connect(FACTOR_LAB_DB)

    # Check if already recorded
    existing = con.execute(
        "SELECT COUNT(*) FROM paper_picks WHERE as_of = ?", [as_of]
    ).fetchone()[0]
    if existing > 0:
        print(f"  Already recorded {existing} picks for {as_of}")
        con.close()
        return

    for i, row in bottom.iterrows():
        con.execute(
            "INSERT INTO paper_picks VALUES (?, 'long', ?, ?, ?)",
            [as_of, i + 1, row["symbol"], float(row["score"])],
        )
    for i, row in top.iterrows():
        con.execute(
            "INSERT INTO paper_picks VALUES (?, 'short', ?, ?, ?)",
            [as_of, i + 1, row["symbol"], float(row["score"])],
        )

    con.close()
    print(f"  Recorded {n} long + {n} short picks for {as_of}")
    print(f"  Long (Q1): {', '.join(bottom['symbol'].head(5))}...")
    print(f"  Short(Q5): {', '.join(top['symbol'].tail(5))}...")


def evaluate(as_of: str | None = None):
    """Evaluate returns for a previous day's picks using next-day close.

    as_of: the signal date (picks were made on this date).
    Returns are measured from as_of close to the next trading day's close.
    """
    init_tables()

    con = duckdb.connect(FACTOR_LAB_DB)

    if as_of is None:
        # Find the latest recorded but unevaluated pick date
        r = con.execute("""
            SELECT DISTINCT as_of FROM paper_picks
            WHERE as_of NOT IN (SELECT as_of FROM paper_returns)
            ORDER BY as_of
            LIMIT 1
        """).fetchone()
        if r is None:
            print("  No unevaluated picks found")
            con.close()
            return
        as_of = str(r[0])

    # Get picks
    picks = con.execute(
        "SELECT side, symbol FROM paper_picks WHERE as_of = ?", [as_of]
    ).fetchdf()

    if picks.empty:
        print(f"  No picks found for {as_of}")
        con.close()
        return

    long_syms = picks[picks["side"] == "long"]["symbol"].tolist()
    short_syms = picks[picks["side"] == "short"]["symbol"].tolist()

    # Get next trading day
    us_con = _open_us_db_readonly()
    eval_date_r = us_con.execute(
        "SELECT MIN(date) FROM prices_daily WHERE date > ?", [as_of]
    ).fetchone()

    if eval_date_r[0] is None:
        print(f"  No price data after {as_of} yet")
        us_con.close()
        con.close()
        return

    eval_date = str(eval_date_r[0])

    # Get 1-day returns (as_of close → eval_date close)
    all_syms = long_syms + short_syms + ["SPY"]
    placeholders = ", ".join(["?"] * len(all_syms))
    ret_df = us_con.execute(f"""
        WITH prices AS (
            SELECT symbol, date, adj_close
            FROM prices_daily
            WHERE date IN (?, ?) AND symbol IN ({placeholders})
        )
        SELECT a.symbol,
               b.adj_close / a.adj_close - 1 AS ret_1d
        FROM prices a
        JOIN prices b ON a.symbol = b.symbol
        WHERE a.date = ? AND b.date = ?
    """, [as_of, eval_date] + all_syms + [as_of, eval_date]).fetchdf()
    us_con.close()

    ret_map = dict(zip(ret_df["symbol"], ret_df["ret_1d"]))

    # Compute portfolio returns
    long_rets = [ret_map[s] for s in long_syms if s in ret_map]
    short_rets = [ret_map[s] for s in short_syms if s in ret_map]
    spy_ret = ret_map.get("SPY", 0.0)

    long_ret = float(np.mean(long_rets)) if long_rets else 0.0
    short_ret = float(np.mean(short_rets)) if short_rets else 0.0
    ls_ret = long_ret - short_ret
    long_excess = long_ret - spy_ret

    # Cumulative returns
    prev = con.execute(
        "SELECT cum_long, cum_spy, cum_excess FROM paper_returns ORDER BY as_of DESC LIMIT 1"
    ).fetchone()

    if prev:
        cum_long = (1 + prev[0]) * (1 + long_ret) - 1
        cum_spy = (1 + prev[1]) * (1 + spy_ret) - 1
        cum_excess = (1 + prev[2]) * (1 + long_excess) - 1
    else:
        cum_long = long_ret
        cum_spy = spy_ret
        cum_excess = long_excess

    con.execute(
        "INSERT OR REPLACE INTO paper_returns VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [as_of, eval_date, long_ret, short_ret, ls_ret, spy_ret, long_excess,
         len(long_rets), len(short_rets), cum_long, cum_spy, cum_excess],
    )
    con.close()

    print(f"  {as_of} → {eval_date}:")
    print(f"    Long:  {long_ret*100:+.2f}% ({len(long_rets)} stocks)")
    print(f"    SPY:   {spy_ret*100:+.2f}%")
    print(f"    Excess:{long_excess*100:+.2f}%")
    print(f"    Cum:   Long {cum_long*100:+.1f}%  SPY {cum_spy*100:+.1f}%  Excess {cum_excess*100:+.1f}%")


def report():
    """Print performance summary."""
    init_tables()

    con = duckdb.connect(FACTOR_LAB_DB, read_only=True)
    df = con.execute("SELECT * FROM paper_returns ORDER BY as_of").fetchdf()

    if df.empty:
        print("No paper trading data yet.")
        con.close()
        return

    n = len(df)
    ann_factor = 252

    long_daily = df["long_ret"].values
    spy_daily = df["spy_ret"].values
    excess_daily = df["long_excess"].values

    ann_long = np.mean(long_daily) * ann_factor
    ann_spy = np.mean(spy_daily) * ann_factor
    ann_excess = np.mean(excess_daily) * ann_factor
    vol_long = np.std(long_daily) * np.sqrt(ann_factor)
    vol_excess = np.std(excess_daily) * np.sqrt(ann_factor)

    sharpe_long = ann_long / vol_long if vol_long > 0 else 0
    info_ratio = ann_excess / vol_excess if vol_excess > 0 else 0

    # Max drawdown
    cum_eq = np.cumprod(1 + long_daily)
    max_dd = np.min(cum_eq / np.maximum.accumulate(cum_eq) - 1)

    # Win rate
    win_rate = (excess_daily > 0).mean()

    # Turnover
    picks_df = con.execute("""
        SELECT as_of, side, symbol FROM paper_picks WHERE side='long' ORDER BY as_of
    """).fetchdf()
    con.close()

    turnovers = []
    prev_set = None
    for dt, g in picks_df.groupby("as_of"):
        cur_set = set(g["symbol"])
        if prev_set is not None:
            to = 1 - len(cur_set & prev_set) / max(len(cur_set), 1)
            turnovers.append(to)
        prev_set = cur_set
    avg_turnover = np.mean(turnovers) if turnovers else 0

    print(f"{'='*50}")
    print(f"  Paper Trading Report")
    print(f"  {df['as_of'].min()} → {df['as_of'].max()} ({n} days)")
    print(f"{'='*50}")
    print(f"  Ann. Return (long):   {ann_long*100:+.1f}%")
    print(f"  Ann. Return (SPY):    {ann_spy*100:+.1f}%")
    print(f"  Ann. Excess:          {ann_excess*100:+.1f}%")
    print(f"  Sharpe (long):        {sharpe_long:.2f}")
    print(f"  Info Ratio:           {info_ratio:.2f}")
    print(f"  Vol (long):           {vol_long*100:.1f}%")
    print(f"  Max Drawdown:         {max_dd*100:.1f}%")
    print(f"  Win vs SPY:           {win_rate*100:.0f}%")
    print(f"  Avg Turnover:         {avg_turnover*100:.0f}%/day")
    print(f"  Cum Long:             {df['cum_long'].iloc[-1]*100:+.1f}%")
    print(f"  Cum SPY:              {df['cum_spy'].iloc[-1]*100:+.1f}%")
    print(f"  Cum Excess:           {df['cum_excess'].iloc[-1]*100:+.1f}%")
    print()
    print("  Recent days:")
    for _, row in df.tail(5).iterrows():
        print(f"    {row['as_of']}: Long {row['long_ret']*100:+.2f}%  SPY {row['spy_ret']*100:+.2f}%  Excess {row['long_excess']*100:+.2f}%")


def backfill(start: str, end: str | None = None):
    """Backfill historical picks and returns from existing analysis_daily data."""
    if end is None:
        end = _get_latest_trade_date()

    # Get all trade dates with lab_factor data
    con = duckdb.connect(US_DB, read_only=True)
    dates = con.execute("""
        SELECT DISTINCT date FROM analysis_daily
        WHERE module_name = 'lab_factor' AND date >= ? AND date <= ?
        ORDER BY date
    """, [start, end]).fetchdf()["date"].tolist()
    con.close()

    print(f"Backfilling {len(dates)} dates from {start} to {end}")

    for dt in dates:
        dt_str = str(dt)
        record(as_of=dt_str)

    # Evaluate returns (need next-day prices)
    for dt in dates[:-1]:  # skip last date (no next-day return yet)
        evaluate(as_of=str(dt))

    report()
