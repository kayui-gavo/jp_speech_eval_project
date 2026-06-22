from __future__ import annotations

import unittest
from pathlib import Path
import csv

from scripts.audit_pitch_heldout_validation import (
    JANON_LEARNER_TARGET,
    JVS_TEST_SPEAKERS,
    JVS_UTTERANCES_PER_SPEAKER,
    _cluster_ci,
    build_manifest,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]


class PitchHeldoutValidationTests(unittest.TestCase):
    def test_manifest_uses_heldout_jvs_speakers_and_both_datasets(self) -> None:
        jvs = ROOT.parent / "JVS"
        janon = ROOT.parent / "JANON"
        if not jvs.exists() or not janon.exists():
            self.skipTest("local JVS/JANON datasets unavailable")
        rows = build_manifest(jvs, janon)
        jvs_rows = [row for row in rows if row["dataset"] == "JVS"]
        janon_native = [row for row in rows if row["group"] == "janon_native_external"]
        janon_learner = [row for row in rows if row["group"] == "janon_learner_external"]
        self.assertEqual(len(jvs_rows), len(JVS_TEST_SPEAKERS) * JVS_UTTERANCES_PER_SPEAKER)
        self.assertEqual({row["speaker_id"] for row in jvs_rows}, set(JVS_TEST_SPEAKERS))
        self.assertTrue(all("nonpara30" in row["audio_path"] for row in jvs_rows))
        self.assertEqual(len(janon_native), 284)
        self.assertEqual(len(janon_learner), JANON_LEARNER_TARGET)
        self.assertEqual(len({row["sample_id"] for row in rows}), 1200)

    def test_summary_keeps_unavailable_rows_out_of_score_mean(self) -> None:
        rows = [
            {"group": "g", "condition": "normal", "speaker_id": "a", "score": 90},
            {"group": "g", "condition": "normal", "speaker_id": "b", "score": None},
        ]
        item = summarize(rows)[0]
        self.assertEqual(item["n"], 1)
        self.assertEqual(item["mean"], 90.0)
        self.assertEqual(item["unavailable_count"], 1)

    def test_cluster_ci_operates_on_speaker_macro_values(self) -> None:
        mean, low, high = _cluster_ci({"a": [10, 10], "b": [20, 20]})
        self.assertEqual(mean, 15.0)
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)

    def test_audit_does_not_modify_runtime_scoring(self) -> None:
        source = (ROOT / "scripts/audit_pitch_heldout_validation.py").read_text(encoding="utf-8")
        self.assertIn('"score_formula_frozen": True', source)
        self.assertIn('"scoring_commit": "95b9510"', source)
        self.assertNotIn("calibration transform", source.lower())

    def test_generated_heldout_result_preserves_expected_ordering(self) -> None:
        path = ROOT / "results/calibration/pitch_heldout_validation_summary.csv"
        if not path.exists():
            self.skipTest("held-out audit result not generated")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        scores = {(row["group"], row["condition"]): row for row in rows}
        native = float(scores[("jvs_native_heldout", "native_normal")]["mean"])
        flat = float(scores[("jvs_native_heldout", "flat")]["mean"])
        shuffled = float(scores[("jvs_native_heldout", "shuffled")]["mean"])
        wrong_drop = float(scores[("jvs_native_heldout", "wrong_drop")]["mean"])
        self.assertGreater(native, flat)
        self.assertGreater(native, shuffled)
        self.assertGreater(native, wrong_drop)
        self.assertEqual(scores[("jvs_native_heldout", "low_f0")]["unavailable_count"], "600")


if __name__ == "__main__":
    unittest.main()
