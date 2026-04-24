from __future__ import annotations

import argparse
from datetime import date, datetime
from zoneinfo import ZoneInfo

import duckdb

from src.paths import QUANT_CN_DB, QUANT_US_DB


MARKET_SPECS = {
    "cn": {
        "db_path": QUANT_CN_DB,
        "table": "prices",
        "date_col": "trade_date",
    },
    "us": {
        "db_path": QUANT_US_DB,
        "table": "prices_daily",
        "date_col": "date",
    },
}


def _coerce_date(value: object | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def latest_trade_date(market: str) -> date | None:
    spec = MARKET_SPECS[market]
    con = duckdb.connect(str(spec["db_path"]), read_only=True)
    try:
        row = con.execute(
            f"SELECT MAX({spec['date_col']}) FROM {spec['table']}"
        ).fetchone()
    finally:
        con.close()
    return _coerce_date(row[0] if row else None)


def expected_us_data_date(now: datetime | None = None) -> date:
    ny_tz = ZoneInfo("America/New_York")
    current = now or datetime.now(ny_tz)
    if current.tzinfo is None:
        current = current.replace(tzinfo=ny_tz)
    return current.astimezone(ny_tz).date()


def market_data_ready(
    market: str,
    *,
    expected_date: date | str | None = None,
) -> tuple[bool, date | None, date | None]:
    latest = latest_trade_date(market)
    expected = _coerce_date(expected_date)

    if expected is None:
        expected = expected_us_data_date() if market == "us" else latest

    ready = latest is not None and expected is not None and latest >= expected
    return ready, latest, expected


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether market data is fresh enough.")
    parser.add_argument("--market", choices=["cn", "us"], required=True)
    parser.add_argument("--expected-date", help="Required latest trade date (YYYY-MM-DD).")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ready, latest, expected = market_data_ready(
        args.market,
        expected_date=args.expected_date,
    )

    if not args.quiet:
        if ready:
            print(
                f"{args.market.upper()} data ready: latest={latest} expected>={expected}"
            )
        else:
            print(
                f"{args.market.upper()} data stale: latest={latest} expected>={expected}"
            )

    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
