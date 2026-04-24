from __future__ import annotations

import unittest
from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from src import data_readiness


class DataReadinessTests(unittest.TestCase):
    def test_expected_us_data_date_uses_new_york_calendar_day(self):
        now = datetime(2026, 4, 1, 4, 35, tzinfo=ZoneInfo("Asia/Shanghai"))

        expected = data_readiness.expected_us_data_date(now)

        self.assertEqual(expected, date(2026, 3, 31))

    @patch("src.data_readiness.latest_trade_date", return_value=date(2026, 3, 30))
    def test_market_data_ready_fails_when_latest_date_is_stale(self, _latest_trade_date):
        ready, latest, expected = data_readiness.market_data_ready(
            "us",
            expected_date=date(2026, 3, 31),
        )

        self.assertFalse(ready)
        self.assertEqual(latest, date(2026, 3, 30))
        self.assertEqual(expected, date(2026, 3, 31))

    @patch("src.data_readiness.latest_trade_date", return_value=date(2026, 3, 31))
    def test_market_data_ready_passes_when_latest_date_matches_expected(self, _latest_trade_date):
        ready, latest, expected = data_readiness.market_data_ready(
            "us",
            expected_date=date(2026, 3, 31),
        )

        self.assertTrue(ready)
        self.assertEqual(latest, date(2026, 3, 31))
        self.assertEqual(expected, date(2026, 3, 31))


if __name__ == "__main__":
    unittest.main()
