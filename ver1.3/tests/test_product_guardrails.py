from __future__ import annotations

import unittest

from jp_speech_eval.asr_confirmation import build_confirmed_weak_target
from jp_speech_eval.eval_modes import evaluate_mode
from jp_speech_eval.feedback_renderer import render_user_facing_result
from jp_speech_eval.scoring_policy import policy_from_result
from jp_speech_eval.special_mora_scorer import score_special_mora_timing


def _result(**overrides):
    base = {
        "target_text": "ラーメンをください",
        "kana": "ラーメンヲクダサイ",
        "moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"],
        "mora_table": [
            {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2},
            {"mora": "ー", "start_sec": 0.2, "end_sec": 0.25},
            {"mora": "メ", "start_sec": 0.25, "end_sec": 0.45},
            {"mora": "ン", "start_sec": 0.45, "end_sec": 0.65},
            {"mora": "ヲ", "start_sec": 0.65, "end_sec": 0.85},
            {"mora": "ク", "start_sec": 0.85, "end_sec": 1.05},
            {"mora": "ダ", "start_sec": 1.05, "end_sec": 1.25},
            {"mora": "サ", "start_sec": 1.25, "end_sec": 1.45},
            {"mora": "イ", "start_sec": 1.45, "end_sec": 1.65},
        ],
        "total_score": 88,
        "pronunciation_score": 80,
        "prosody_score": 90,
        "fluency_score": 95,
        "tone_score": 70,
        "feedback": ["整体音高和示范音比较接近。", "語速は自然です。"],
        "alignment_mode": "cached_dtw",
        "details": {
            "mode": "reference_based",
            "pitch_target_source": "ojad_checked",
            "verified_level": "ojad_checked",
            "reliability": {"level": "high", "overall": 0.95, "alignment": 0.9, "f0_coverage": 0.9},
            "recording_quality": {"score": 0.9},
            "content_match": {"status": "pass"},
            "alignment": {"mode": "cached_dtw"},
            "pronunciation": {"mora_duration_cv": 0.1, "special_mora_diagnostics": []},
            "prosody": {"contour_corr": 0.8, "transition_agreement": 0.8, "pitch_target_source": "ojad_checked"},
            "mora_evidence": [
                {"judgement_available": True, "boundary_confidence": 0.9, "energy_coverage": 0.9}
                for _ in range(9)
            ],
        },
    }
    for key, value in overrides.items():
        if key == "details":
            base["details"].update(value)
        else:
            base[key] = value
    return base


class ProductGuardrailsTest(unittest.TestCase):
    def test_human_checked_fixed_reference_allows_special_mora_feedback(self) -> None:
        rendered = render_user_facing_result(_result())
        self.assertFalse(rendered["display_total_score"])
        self.assertEqual(rendered["focus_feedback"]["category"], "special_mora")
        self.assertIn("もう少し", rendered["focus_feedback"]["message"])

    def test_auto_pyopenjtalk_blocks_pitch_feedback(self) -> None:
        result = _result(details={"pitch_target_source": "auto_pyopenjtalk", "verified_level": "auto_pyopenjtalk"})
        rendered = render_user_facing_result(result)
        self.assertFalse(rendered["debug"]["scoring_policy"]["allow_pitch_feedback"])
        self.assertFalse(any("音高" in msg for msg in rendered["user_messages"]))

    def test_asr_raw_result_cannot_score(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_mode("asr_pseudo_reference", "dummy.wav", cache_path="cache/ramen_kudasai")

    def test_confirmed_text_builds_weak_target(self) -> None:
        target = build_confirmed_weak_target("ラーメンをください")
        self.assertTrue(target["weak_reference"])
        self.assertEqual(target["target_source"], "user_confirmed_asr")
        self.assertFalse(target["scoring_policy"]["allow_pitch_feedback"])

    def test_kanade_is_demo_only_and_excluded(self) -> None:
        result = _result(details={"mode": "kanade_asr_voice_reference", "demo_only": True, "exclude_from_pronunciation_score": True})
        policy = policy_from_result(result)
        self.assertTrue(policy.demo_only)
        self.assertTrue(policy.exclude_from_pronunciation_score)

    def test_short_utterance_blocks_pitch(self) -> None:
        result = _result(moras=["バ", "グ"], mora_table=[{"start_sec": 0.0, "end_sec": 0.2}, {"start_sec": 0.2, "end_sec": 0.4}])
        rendered = render_user_facing_result(result)
        self.assertIn("pitch", rendered["debug"]["reliability_gate"]["blocked_categories"])

    def test_low_alignment_makes_special_mora_uncertain(self) -> None:
        result = _result(details={"mora_evidence": [{"judgement_available": False, "boundary_confidence": 0.1, "energy_coverage": 0.1} for _ in range(9)]})
        rows = score_special_mora_timing(result)
        self.assertTrue(any(row.status == "uncertain" for row in rows))

    def test_pitch_accent_proxy_does_not_drive_display_score(self) -> None:
        result = _result(
            total_score=65,
            pronunciation_score=90,
            fluency_score=88,
            prosody_score=20,
            details={
                "fluency": {"rhythm_timing_score": 88, "delivery_fluency_score": 90},
                "prosody": {"pitch_accent_score": 0, "final_intonation_score": 85},
                "pronunciation": {"mora_duration_cv": 0.08, "special_mora_penalty": 0},
            },
        )
        rendered = render_user_facing_result(result)
        self.assertGreaterEqual(rendered["display_score"], 88)

    def test_weak_reference_downweights_prosody_proxy(self) -> None:
        result = _result(
            total_score=65,
            pronunciation_score=92,
            prosody_score=10,
            fluency_score=88,
            details={
                "weak_reference": True,
                "target_source": "user_confirmed_asr",
                "fluency": {"rhythm_timing_score": 87, "delivery_fluency_score": 90},
                "prosody": {"contour_corr": 0.1, "transition_agreement": 0.1},
                "pronunciation": {"mora_duration_cv": 0.08, "special_mora_penalty": 0},
            },
        )
        rendered = render_user_facing_result(result, mode="asr_pseudo_reference")
        self.assertGreaterEqual(rendered["display_score"], 85)
        self.assertTrue(rendered["debug"]["weak_reference"])

    def test_missing_split_fluency_is_not_treated_as_zero(self) -> None:
        result = _result(
            total_score=65,
            pronunciation_score=86,
            fluency_score=84,
            details={
                "fluency": {},
                "pronunciation": {"mora_duration_cv": 0.12, "special_mora_penalty": 0},
            },
        )
        rendered = render_user_facing_result(result)
        self.assertGreaterEqual(rendered["display_score"], 80)


if __name__ == "__main__":
    unittest.main()
