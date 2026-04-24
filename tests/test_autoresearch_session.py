from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.autoresearch.session_state import ensure_session_files
from src.agent.prompts import build_system_prompt


class AutoresearchSessionTests(unittest.TestCase):
    def test_ensure_session_files_creates_pi_style_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = ensure_session_files("us", goal="Optimize US factor mining loop", root=root)

            self.assertTrue(paths.session_doc.exists())
            self.assertTrue(paths.benchmark_script.exists())
            self.assertTrue(paths.checks_script.exists())
            self.assertTrue(paths.log_file.exists())

            context_text = paths.session_doc.read_text(encoding="utf-8")
            benchmark_text = paths.benchmark_script.read_text(encoding="utf-8")
            checks_text = paths.checks_script.read_text(encoding="utf-8")

            self.assertIn("Optimize US factor mining loop", context_text)
            self.assertIn("METRIC is_ic_ir", benchmark_text)
            self.assertIn("--eval-composite --market \"$MARKET\"", checks_text)

    def test_system_prompt_embeds_resumable_session_context(self) -> None:
        prompt = build_system_prompt(
            market="cn",
            session_context="# Session Notes\nTry volume-stability families first.",
        )
        self.assertIn("Resumable Session Context", prompt)
        self.assertIn("Try volume-stability families first.", prompt)


if __name__ == "__main__":
    unittest.main()
