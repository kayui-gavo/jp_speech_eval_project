from __future__ import annotations

import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data/calibration_candidates/pitch_v2_objective_validation_summary.csv"


class PitchV2ObjectiveValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with SUMMARY.open(encoding="utf-8") as stream:
            cls.rows = list(csv.DictReader(stream))
        cls.lookup = {(row["section"], row["metric"]): row for row in cls.rows}

    def test_paired_controls_are_strongly_separated(self) -> None:
        flat = self.lookup[("paired_control", "normal_vs_flat")]
        shuffled = self.lookup[("paired_control", "normal_vs_shuffled")]
        self.assertGreater(float(flat["paired_delta_ci95_low"]), 55.0)
        self.assertGreater(float(shuffled["paired_delta_ci95_low"]), 30.0)
        self.assertGreater(float(flat["roc_auc"]), 0.98)
        self.assertGreater(float(shuffled["roc_auc"]), 0.95)

    def test_every_speaker_disjoint_fold_preserves_ordering(self) -> None:
        folds = [row for row in self.rows if row["section"] == "speaker_disjoint_cv"]
        self.assertEqual(len(folds), 5)
        self.assertTrue(all(row["ordering_pass"] == "True" for row in folds))

    def test_same_sentence_analysis_requires_broad_speaker_coverage(self) -> None:
        repeat = self.lookup[("jvs_repeatability_summary", "parallel100_same_sentences_across_speakers")]
        sentence_rows = [row for row in self.rows if row["section"] == "jvs_sentence_repeatability"]
        minimum = int(repeat["minimum_sentence_speaker_coverage"])
        self.assertEqual(len(sentence_rows), int(repeat["sentence_count"]))
        self.assertTrue(all(int(row["n"]) >= minimum for row in sentence_rows))

    def test_fresh_native_scores_do_not_collapse_to_endpoints(self) -> None:
        native = self.lookup[("cross_corpus_distribution", "jvs_native_fresh")]
        self.assertGreater(float(native["mean"]), 85.0)
        self.assertLess(float(native["floor_or_ceiling_rate"]), 0.05)
        self.assertGreater(int(native["unique_integer_scores"]), 20)

    def test_report_states_external_validity_limits(self) -> None:
        report = (ROOT / "reports/pitch_v2_objective_validation.md").read_text(encoding="utf-8")
        self.assertIn("Learner identity is not an error label", report)
        self.assertIn("Teacher-grade pitch-accent correctness", report)
        self.assertIn("Device/noise robustness", report)


if __name__ == "__main__":
    unittest.main()
