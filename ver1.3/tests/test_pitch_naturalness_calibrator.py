from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from scripts.train_pitch_naturalness_calibrator import DEV_SPEAKERS, FEATURES, TEST_SPEAKERS, TRAIN_SPEAKERS


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data/calibration_candidates/pitch_naturalness_calibrator_candidate.json"
SUMMARY = ROOT / "data/calibration_candidates/pitch_naturalness_calibration_summary.csv"


class PitchNaturalnessCalibratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = json.loads(MODEL.read_text(encoding="utf-8"))
        cls.summary = {
            (row["split"], row["condition"]): row
            for row in csv.DictReader(SUMMARY.open(encoding="utf-8"))
        }

    def test_planned_speaker_splits_are_disjoint(self) -> None:
        train = set(TRAIN_SPEAKERS)
        dev = set(DEV_SPEAKERS)
        test = set(TEST_SPEAKERS)
        self.assertFalse(train & dev)
        self.assertFalse(train & test)
        self.assertFalse(dev & test)

    def test_artifact_is_offline_and_not_teacher_pitch_correctness(self) -> None:
        self.assertEqual(self.model["status"], "offline_candidate_not_active_in_runtime")
        self.assertIn("teacher_pitch_accent_correctness", self.model["not_a_target"])
        self.assertEqual(self.model["features"], list(FEATURES))
        self.assertEqual(len(self.model["coefficients"]), len(FEATURES))

    def test_artifact_records_actual_speaker_disjoint_usage(self) -> None:
        splits = self.model["speaker_splits"]
        self.assertEqual(len(splits["train"]), 28)
        self.assertEqual(len(splits["dev"]), 10)
        self.assertEqual(len(splits["locked_test"]), 60)
        self.assertFalse(set(splits["train"]) & set(splits["dev"]))
        self.assertFalse(set(splits["train"]) & set(splits["locked_test"]))

    def test_locked_test_separates_native_from_flat_and_shuffle(self) -> None:
        native = float(self.summary[("locked_test", "native_normal")]["mean"])
        flat = float(self.summary[("locked_test", "flat")]["mean"])
        shuffled = float(self.summary[("locked_test", "shuffled")]["mean"])
        self.assertGreater(native, 85.0)
        self.assertGreater(native - flat, 70.0)
        self.assertGreater(native - shuffled, 60.0)
        self.assertGreater(float(self.model["metrics"]["locked_test_auc"]), 0.95)

    def test_wrong_drop_remains_an_explicit_limitation(self) -> None:
        native = float(self.summary[("locked_test", "native_normal")]["mean"])
        wrong_drop = float(self.summary[("locked_test", "wrong_drop")]["mean"])
        self.assertLess(native - wrong_drop, 15.0)
        report = (ROOT / "reports/pitch_naturalness_calibration_experiment.md").read_text(encoding="utf-8")
        self.assertIn("not lexical pitch-accent correctness", report)
        self.assertIn("Do not activate this candidate", report)

    def test_runtime_does_not_load_candidate_artifact(self) -> None:
        runtime = "\n".join(
            (ROOT / path).read_text(encoding="utf-8")
            for path in (
                "src/jp_speech_eval/scoring.py",
                "src/jp_speech_eval/evaluator.py",
                "src/jp_speech_eval/feedback_renderer.py",
            )
        )
        self.assertNotIn("pitch_naturalness_calibrator_candidate.json", runtime)


if __name__ == "__main__":
    unittest.main()
