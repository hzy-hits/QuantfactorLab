from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.autoresearch.dashboard import build_dashboard_summary, export_dashboard_bundle, load_session_runs
from src.autoresearch.finalize import build_finalize_plan


SAMPLE_RUNS = [
    {
        "session_id": "sess1",
        "market": "us",
        "name": "alpha_keep",
        "formula": "rank(close)",
        "is_ic": 0.03,
        "is_ic_ir": 0.41,
        "gates": "PASS",
        "oos": "PASS",
        "checks_status": "passed",
        "decision": "keep",
        "status": "kept",
        "session_branch": "feature/autoresearch",
        "base_ref": "main",
        "merge_base": "abc123",
    },
    {
        "session_id": "sess1",
        "market": "us",
        "name": "alpha_revert",
        "formula": "rank(volume)",
        "is_ic": 0.01,
        "is_ic_ir": 0.12,
        "gates": "FAIL",
        "decision": "revert",
        "status": "reverted",
        "session_branch": "feature/autoresearch",
        "base_ref": "main",
        "merge_base": "abc123",
    },
]


class AutoresearchDashboardTests(unittest.TestCase):
    def test_dashboard_summary_counts_keep_and_revert(self) -> None:
        summary = build_dashboard_summary(SAMPLE_RUNS)
        self.assertEqual(summary["total_runs"], 2)
        self.assertEqual(summary["keep_count"], 1)
        self.assertEqual(summary["revert_count"], 1)
        self.assertEqual(summary["oos_pass"], 1)
        self.assertEqual(summary["session_branch"], "feature/autoresearch")

    def test_export_bundle_writes_markdown_json_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_path = root / "autoresearch.jsonl"
            log_path.write_text("\n".join(json.dumps(run) for run in SAMPLE_RUNS), encoding="utf-8")
            bundle = export_dashboard_bundle(market="us", log_path=log_path, output_dir=root / "exports")
            runs = load_session_runs(log_path)

            self.assertEqual(len(runs), 2)
            self.assertTrue(bundle["markdown"].exists())
            self.assertTrue(bundle["json"].exists())
            self.assertTrue(bundle["html"].exists())
            self.assertIn("Autoresearch Dashboard", bundle["markdown"].read_text(encoding="utf-8"))

    def test_finalize_plan_only_includes_kept_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_path = root / "autoresearch.jsonl"
            log_path.write_text("\n".join(json.dumps(run) for run in SAMPLE_RUNS), encoding="utf-8")
            plans = build_finalize_plan(log_path, "us")

            self.assertEqual(len(plans), 1)
            self.assertEqual(plans[0].experiment_name, "alpha_keep")
            self.assertIn("autoresearch/us/sess1/alpha-keep", plans[0].branch_name)


if __name__ == "__main__":
    unittest.main()
