#!/usr/bin/env python3
"""Weekly data maintenance: check gaps → backfill → report.

Run every Saturday morning to ensure Monday's pipeline has clean data.

Usage:
    python scripts/weekly_maintenance.py              # check + backfill
    python scripts/weekly_maintenance.py --dry-run    # check only, no writes
    python scripts/weekly_maintenance.py --days 365   # backfill up to 365 days

Cron:
    17 10 * * 6 cd $FACTOR_LAB_ROOT && python3 scripts/weekly_maintenance.py >> logs/maintenance.log 2>&1
"""
import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import tushare as ts
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import QUANT_CN_DB, QUANT_CN_ROOT

# ── Config ───────────────────────────────────────────────────────────────────

CN_DB = str(QUANT_CN_DB)
CN_CONFIG = QUANT_CN_ROOT / "config.yaml"

# Tables to check and backfill
TABLES = {
    "moneyflow": {
        "date_col": "trade_date",
        "sym_col": "ts_code",
        "api": "moneyflow",
        "min_stocks_per_day": 3000,  # expect at least this many stocks
    },
    "margin_detail": {
        "date_col": "trade_date",
        "sym_col": "ts_code",
        "api": "margin_detail",
        "min_stocks_per_day": 1000,
    },
    "daily_basic": {
        "date_col": "trade_date",
        "sym_col": "ts_code",
        "api": "daily_basic",
        "min_stocks_per_day": 3000,
    },
}

# Tushare API rate limit: ~200 req/min → 350ms between calls to be safe
API_DELAY = 0.35


def load_token() -> str:
    with CN_CONFIG.open() as f:
        cfg = yaml.safe_load(f)
    return cfg["api"]["tushare_token"]


def get_trading_dates(pro, start: str, end: str) -> list[str]:
    """Get all A-share trading dates in range (returns YYYYMMDD format)."""
    df = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    return sorted(df[df["is_open"] == 1]["cal_date"].tolist())


def _normalize_date(d: str) -> str:
    """Normalize date to YYYYMMDD for comparison."""
    return d.replace("-", "")


def check_table(con, table: str, cfg: dict, trading_dates: list[str]) -> list[str]:
    """Find missing or sparse dates in a table. Returns list of dates to backfill."""
    date_col = cfg["date_col"]
    sym_col = cfg["sym_col"]
    min_stocks = cfg["min_stocks_per_day"]

    # Get existing date counts (normalize to YYYYMMDD for matching)
    existing = con.execute(f"""
        SELECT {date_col}::VARCHAR, COUNT(DISTINCT {sym_col})
        FROM {table}
        GROUP BY {date_col}
    """).fetchall()
    date_counts = {_normalize_date(row[0]): row[1] for row in existing}

    missing = []
    sparse = []
    for td in trading_dates:
        count = date_counts.get(_normalize_date(td), 0)
        if count == 0:
            missing.append(td)
        elif count < min_stocks:
            sparse.append((td, count))

    return missing, sparse


def _get_table_columns(con, table: str) -> list[str]:
    """Get column names for a DuckDB table."""
    desc = con.execute(f"DESCRIBE {table}").fetchall()
    return [row[0] for row in desc]


def backfill_table(pro, con, table: str, cfg: dict, dates: list[str], dry_run: bool):
    """Backfill missing dates for a table."""
    api_name = cfg["api"]
    total_rows = 0

    # Get target table columns for filtering
    if not dry_run:
        table_cols = _get_table_columns(con, table)

    for i, td in enumerate(dates):
        if dry_run:
            if i < 3 or i == len(dates) - 1:
                print(f"    [DRY-RUN] Would fetch {api_name} for {td}")
            elif i == 3:
                print(f"    ... ({len(dates) - 4} more dates)")
            continue

        try:
            if api_name == "moneyflow":
                df = pro.moneyflow(trade_date=td)
            elif api_name == "margin_detail":
                df = pro.margin_detail(trade_date=td)
            elif api_name == "daily_basic":
                df = pro.daily_basic(trade_date=td)
            else:
                continue

            if df is not None and len(df) > 0:
                # Convert YYYYMMDD date strings to YYYY-MM-DD for DuckDB DATE columns
                if "trade_date" in df.columns:
                    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                # Filter to only columns that exist in the target table
                common_cols = [c for c in table_cols if c in df.columns]
                df_filtered = df[common_cols]
                col_list = ", ".join(common_cols)
                con.execute(f"INSERT OR REPLACE INTO {table} ({col_list}) SELECT {col_list} FROM df_filtered")
                total_rows += len(df_filtered)
                if (i + 1) % 20 == 0:
                    print(f"    Progress: {i+1}/{len(dates)} dates, {total_rows} rows")

            time.sleep(API_DELAY)

        except Exception as e:
            print(f"    Error fetching {api_name} for {td}: {e}")
            time.sleep(1)

    return total_rows


def main():
    parser = argparse.ArgumentParser(description="Weekly data maintenance")
    parser.add_argument("--dry-run", action="store_true", help="Check only, no writes")
    parser.add_argument("--days", type=int, default=250, help="Max days to look back (default: 250 ≈ 1 year)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Weekly Maintenance — {date.today()}")
    print("=" * 60)

    # Setup
    token = load_token()
    pro = ts.pro_api(token)

    end_date = date.today().strftime("%Y%m%d")
    start_date = (date.today() - timedelta(days=args.days)).strftime("%Y%m%d")

    print(f"\nChecking range: {start_date} → {end_date}")

    # Get trading calendar
    trading_dates = get_trading_dates(pro, start_date, end_date)
    print(f"Trading dates in range: {len(trading_dates)}")

    # Connect to DB (read-write for backfill)
    if args.dry_run:
        con = duckdb.connect(CN_DB, read_only=True)
    else:
        con = duckdb.connect(CN_DB)

    total_backfilled = 0

    for table, cfg in TABLES.items():
        print(f"\n--- {table} ---")
        missing, sparse = check_table(con, table, cfg, trading_dates)

        print(f"  Missing dates: {len(missing)}")
        if sparse:
            print(f"  Sparse dates (< {cfg['min_stocks_per_day']} stocks): {len(sparse)}")
            for td, count in sparse[:5]:
                print(f"    {td}: {count} stocks")
            if len(sparse) > 5:
                print(f"    ... and {len(sparse) - 5} more")

        # Backfill missing + sparse dates
        dates_to_fill = missing + [td for td, _ in sparse]
        if dates_to_fill:
            dates_to_fill.sort()
            print(f"  Backfilling {len(dates_to_fill)} dates...")
            rows = backfill_table(pro, con, table, cfg, dates_to_fill, args.dry_run)
            total_backfilled += rows
            if not args.dry_run:
                print(f"  Done: {rows} rows inserted")
        else:
            print("  Complete — no gaps found")

    con.close()

    print(f"\n{'=' * 60}")
    print(f"  Total rows backfilled: {total_backfilled}")
    print(f"  Finished: {date.today()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
