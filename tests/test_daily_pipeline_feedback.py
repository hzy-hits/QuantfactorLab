from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mining import daily_pipeline


class DailyPipelineFeedbackTests(unittest.TestCase):
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
