from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mining import daily_pipeline


class DailyPipelineFeedbackTests(unittest.TestCase):
    def test_load_report_feedback_prefers_algorithm_postmortem_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "pipeline.duckdb"
            con = duckdb.connect(str(db_path))
            con.execute(
                """
                CREATE TABLE algorithm_postmortem (
                    symbol VARCHAR,
                    label VARCHAR,
                    evaluation_date DATE,
                    feedback_action VARCHAR,
                    feedback_weight DOUBLE
                )
                """
            )
            con.executemany(
                """
                INSERT INTO algorithm_postmortem
                VALUES (?, ?, CURRENT_DATE, ?, ?)
                """,
                [
                    ("AAA", "missed_alpha", "boost_recall", 1.0),
                    ("BBB", "right_but_no_fill", "penalize_stale_chase", 0.8),
                    ("CCC", "false_positive_executable", "penalize_false_positive", 0.7),
                    ("DDD", "won_and_executable", "reward_executable_capture", 0.5),
                ],
            )
            con.close()

            with (
                mock.patch.object(daily_pipeline, "QUANT_US_DB", db_path),
                mock.patch.object(daily_pipeline, "QUANT_US_REPORT_DB", Path(tmpdir) / "none.duckdb"),
            ):
                feedback = daily_pipeline._load_report_feedback("us")

        self.assertEqual(feedback["missed_alpha"], {"AAA": 1.0})
        self.assertEqual(feedback["stale"], {"BBB": 0.8})
        self.assertEqual(feedback["false_positive"], {"CCC": 0.7})
        self.assertEqual(feedback["captured"], {"DDD": 0.5})

    def test_report_feedback_records_shadow_alpha_overlap_aliases(self) -> None:
        candidate = {
            "factor_id": "demo",
            "direction": "long",
            "composite_score": 1.0,
            "rank_score": 1.0,
            "sigreg_n_eff_shrinkage": 0.5,
            "_latest_values": pd.Series(
                {
                    "AAA": 3.0,
                    "BBB": 2.0,
                    "CCC": 1.0,
                }
            ),
        }

        with mock.patch.object(
            daily_pipeline,
            "_load_report_feedback",
            return_value={
                "missed_alpha": {"AAA": 1.0},
                "stale": {"BBB": 1.0},
                "false_positive": {},
                "captured": {"CCC": 1.0},
            },
        ):
            adjusted = daily_pipeline._apply_report_feedback([candidate], "us")

        detail = adjusted[0]["report_feedback_detail"]
        self.assertIn("missed_alpha_overlap", detail)
        self.assertIn("stale_chase_overlap", detail)
        self.assertEqual(detail["stale_chase_overlap"], detail["stale_overlap"])
        self.assertIn("captured_overlap", detail)
        self.assertEqual(detail["captured_overlap"], detail["capture_overlap"])
        self.assertLessEqual(adjusted[0]["report_feedback_multiplier"], 1.20)


if __name__ == "__main__":
    unittest.main()
