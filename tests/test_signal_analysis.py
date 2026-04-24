from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from src.evaluate import signal_analysis


class SignalAnalysisDependencyTests(unittest.TestCase):
    def test_wavelet_health_degrades_gracefully_without_pywavelets(self) -> None:
        ic = np.linspace(-0.02, 0.03, 120)

        with mock.patch.object(signal_analysis, "pywt", None):
            health = signal_analysis.wavelet_health(ic)

        self.assertEqual(health["overall"], "unavailable")
        self.assertIn("PyWavelets missing", health["reason"])
        self.assertEqual(health["weakening_scales"], 0)


if __name__ == "__main__":
    unittest.main()
