from __future__ import annotations

import csv
import unittest
from pathlib import Path

from jp_speech_eval.pitch_naturalness_v2 import (
    combine_naturalness_with_hint,
    load_pitch_naturalness_v2_config,
    predict_continuous_naturalness,
    score_pitch_naturalness_v2,
    soft_accent_hint_score,
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data/calibration_candidates/pitch_naturalness_v2_audit_summary.csv"


def _details(**overrides):
    base = {
        "available": True,
        "f0_coverage": 0.95,
        "utterance_f0_range_log": 0.40,
        "local_pitch_movement": 0.55,
        "transition_smoothness": 0.78,
        "flatness_penalty": 0.0,
        "instability_penalty": 0.02,
    }
    base.update(overrides)
    return base


class PitchNaturalnessV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_pitch_naturalness_v2_config.cache_clear()
        cls.config = load_pitch_naturalness_v2_config()
        cls.summary = {
            (row["group"], row["condition"]): row
            for row in csv.DictReader(SUMMARY.open(encoding="utf-8"))
        }

    def test_active_config_is_continuous_not_binary_probability(self) -> None:
        self.assertTrue(self.config["active"])
        self.assertEqual(self.config["model_type"], "standard_scaler_plus_ridge_regression")
        self.assertIn("teacher_grade_pitch_accent_correctness", self.config["not_a_target"])

    def test_continuous_model_penalizes_flat_and_unstable_features(self) -> None:
        natural = predict_continuous_naturalness(_details(), self.config)
        flat = predict_continuous_naturalness(_details(
            utterance_f0_range_log=0.01,
            local_pitch_movement=0.01,
            flatness_penalty=0.45,
        ), self.config)
        unstable = predict_continuous_naturalness(_details(
            local_pitch_movement=1.2,
            transition_smoothness=0.25,
            instability_penalty=0.32,
        ), self.config)
        self.assertIsNotNone(natural)
        self.assertGreater(natural, flat)
        self.assertGreater(natural, unstable)

    def test_automatic_accent_hint_never_boosts_naturalness(self) -> None:
        self.assertEqual(combine_naturalness_with_hint(40.0, 90.0, 0.08), 40.0)
        self.assertLess(combine_naturalness_with_hint(90.0, 60.0, 0.08), 90.0)

    def test_accent_hint_uses_events_and_excludes_sentence_final_region(self) -> None:
        score, details = soft_accent_hint_score(
            [100, 125, 135, 90, 95, 130, 150, 175],
            ["L", "H", "H", "L", "L", "H", "H", "L"],
            [{"moras": ["ア", "イ", "ウ", "エ"], "accent_position": 3},
             {"moras": ["オ", "カ", "キ", "ク"], "accent_position": 3}],
            is_question=True,
        )
        self.assertIsNotNone(score)
        self.assertEqual(details["excluded_final_moras"], 3)
        self.assertTrue(all(event["expected_index"] < 4 for event in details["events"]))

    def test_v2_low_f0_stays_internally_unavailable(self) -> None:
        score, details = score_pitch_naturalness_v2(
            [120.0, float("nan"), float("nan")],
            {"available": False, "unavailable_reason": "insufficient_valid_mora_f0"},
            ["L", "H", "L"],
            [],
            pitch_target_source="openjtalk_accent_phrase_chain",
            is_question=False,
            config=self.config,
        )
        self.assertIsNone(score)
        self.assertEqual(details["reason"], "insufficient_valid_mora_f0")

    def test_fresh_jvs_and_janon_audit_meets_activation_guardrails(self) -> None:
        native = self.summary[("fresh_jvs_parallel100", "normal")]
        flat = self.summary[("fresh_jvs_parallel100", "flat")]
        shuffled = self.summary[("fresh_jvs_parallel100", "shuffled")]
        janon_native = self.summary[("janon_native_external", "observed")]
        self.assertGreater(float(native["v2_mean"]), 85.0)
        self.assertGreater(float(native["v2_mean"]) - float(flat["v2_mean"]), 55.0)
        self.assertGreater(float(native["v2_mean"]) - float(shuffled["v2_mean"]), 30.0)
        self.assertLess(float(native["v2_at_ceiling"]), 0.10)
        self.assertGreater(float(janon_native["v2_mean"]), 75.0)

    def test_wrong_drop_is_not_a_required_negative_for_naturalness(self) -> None:
        wrong = self.summary[("fresh_jvs_parallel100", "wrong_drop")]
        self.assertGreater(float(wrong["v2_mean"]), 75.0)
        report = (ROOT / "reports/pitch_naturalness_v2_audit.md").read_text(encoding="utf-8")
        self.assertIn("not required to be low for broad naturalness", report)


if __name__ == "__main__":
    unittest.main()
