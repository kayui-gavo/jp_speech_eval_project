from __future__ import annotations

import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data/calibration_candidates/three_dimensions_objective_validation_summary.csv"


class ThreeDimensionsObjectiveValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with SUMMARY.open(encoding="utf-8") as stream:
            cls.rows = list(csv.DictReader(stream))
        cls.lookup = {(row["section"], row["metric"]): row for row in cls.rows}

    def test_timing_controls_lower_pronunciation_and_rhythm(self) -> None:
        pronunciation = self.lookup[("paired_control", "pronunciation_normal_vs_timing_jitter")]
        rhythm = self.lookup[("paired_control", "rhythm_normal_vs_timing_jitter")]
        self.assertGreater(float(pronunciation["paired_delta_mean"]), 10.0)
        self.assertGreater(float(rhythm["paired_delta_mean"]), 10.0)
        self.assertGreater(float(pronunciation["roc_auc"]), 0.65)

    def test_segmental_substitution_gap_is_not_hidden(self) -> None:
        row = self.lookup[("paired_control", "pronunciation_normal_vs_segment_substitution")]
        self.assertEqual(float(row["paired_delta_mean"]), 0.0)
        self.assertEqual(float(row["roc_auc"]), 0.5)

    def test_fluency_controls_lower_score(self) -> None:
        for control in ("slow", "fast", "hesitation"):
            row = self.lookup[("paired_control", f"fluency_normal_vs_{control}")]
            self.assertGreater(float(row["paired_delta_mean"]), 5.0)

    def test_native_and_fallback_failures_remain_explicit(self) -> None:
        pronunciation = self.lookup[("distribution", "jvs_pronunciation_normal")]
        rhythm_fallback = self.lookup[("paired_control", "rhythm_normal_vs_equal_fallback")]
        janon_learner = self.lookup[("distribution", "janon_learner_fluency_descriptive")]
        self.assertLess(float(pronunciation["mean"]), 30.0)
        self.assertLess(float(rhythm_fallback["paired_delta_mean"]), 0.0)
        self.assertGreater(float(janon_learner["ceiling_rate"]), 0.70)

    def test_report_does_not_use_learners_as_bad_label(self) -> None:
        report = (ROOT / "reports/three_dimensions_objective_validation.md").read_text(encoding="utf-8")
        self.assertIn("JANON learners are not treated as bad-pronunciation labels", report)
        self.assertIn("phoneme substitution with unchanged timing also produces exactly the same score", report)
        self.assertIn("fallback inflates score", report)
        self.assertIn("None of the three has pitch-v2-level validation", report)


if __name__ == "__main__":
    unittest.main()
