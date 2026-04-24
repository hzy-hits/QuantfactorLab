from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
RUN_STRATEGY_PATH = REPO_ROOT / "scripts" / "run_strategy.py"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_strategy = _load_module("factor_lab_run_strategy", RUN_STRATEGY_PATH)


class RunStrategyGuardTests(unittest.TestCase):
    def test_us_rejection_reason_blocks_non_equity_symbol(self) -> None:
        reason = run_strategy._us_rejection_reason("GC=F", None, None)
        self.assertEqual(reason, "代码不是股票/ADR")

    def test_build_us_quality_gate_flags_discontinuous_price_series(self) -> None:
        dates = pd.date_range("2026-01-01", periods=8, freq="B")
        rows = []
        for idx, date in enumerate(dates):
            stable_close = 100.0 + idx
            broken_close = 4000.0 if idx < 7 else 180.0
            rows.append(
                {
                    "symbol": "SAFE",
                    "date": date,
                    "open": stable_close,
                    "high": stable_close + 1.0,
                    "low": stable_close - 1.0,
                    "close": stable_close,
                }
            )
            rows.append(
                {
                    "symbol": "BROKEN",
                    "date": date,
                    "open": broken_close,
                    "high": broken_close + 10.0,
                    "low": broken_close - 10.0,
                    "close": broken_close,
                }
            )

        quality = run_strategy._build_us_quality_gate(
            pd.DataFrame(rows),
            sym_col="symbol",
            date_col="date",
            latest=pd.Timestamp(dates[-1]),
        )

        self.assertEqual(quality["SAFE"]["reasons"], [])
        self.assertIn("price_gap_break", quality["BROKEN"]["reasons"])

    def test_show_today_applies_us_stop_floor_and_filters_non_equity_candidates(self) -> None:
        dates = pd.date_range("2026-02-02", periods=20, freq="B")
        price_rows = []
        factor_rows = []
        for date in dates:
            for symbol, close in (("SAFE", 100.0), ("GC=F", 50.0)):
                price_rows.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "open": close,
                        "high": close * 3.0,
                        "low": 0.01,
                        "close": close,
                        "volume": 1_000_000,
                    }
                )
                factor_rows.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "factor_value": 2.0 if symbol == "SAFE" else 1.0,
                        "ret_next": 0.01,
                    }
                )

        prices = pd.DataFrame(price_rows)
        factor_df = pd.DataFrame(factor_rows)
        cfg = run_strategy.StrategyConfig(
            lookback=10,
            hold_max=5,
            rebalance=5,
            n_picks=2,
            ic_exit_threshold=-0.02,
        )

        output = io.StringIO()
        with (
            mock.patch.object(
                run_strategy,
                "load_data",
                return_value=(prices, {"demo_factor": factor_df}, "symbol", "date"),
            ),
            mock.patch.object(
                run_strategy,
                "select_best_factor",
                return_value=("demo_factor", "top", 0.5),
            ),
            mock.patch.object(
                run_strategy,
                "_load_us_symbol_metadata",
                return_value={"SAFE": {"name": "Safe Corp", "type": "Common Stock", "exchange": "US"}},
            ),
            mock.patch.object(
                run_strategy,
                "_build_us_quality_gate",
                return_value={
                    "SAFE": {"reasons": []},
                    "GC=F": {"reasons": []},
                },
            ),
            contextlib.redirect_stdout(output),
        ):
            run_strategy.show_today("us", cfg, as_of="2026-02-27")

        rendered = output.getvalue()
        self.assertIn("SAFE", rendered)
        self.assertNotRegex(rendered, r"(?m)^\s*\d+\s+GC=F\b")
        self.assertIn("85.00", rendered)
        self.assertIn("代码不是股票/ADR", rendered)


if __name__ == "__main__":
    unittest.main()
