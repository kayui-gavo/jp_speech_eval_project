from __future__ import annotations

import unittest

from jp_speech_eval.asr_confirmation import build_confirmed_weak_target
from jp_speech_eval.eval_modes import evaluate_mode
from jp_speech_eval.feedback_renderer import render_user_facing_result
from jp_speech_eval.scoring_policy import policy_from_result
from jp_speech_eval.special_mora_scorer import (
    decide_special_mora_feature_value,
    decide_special_mora_runtime,
    load_special_mora_thresholds,
    score_special_mora_timing,
    special_mora_score_from_decisions,
)


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
    def test_human_checked_fixed_reference_shadows_special_mora_by_default(self) -> None:
        rendered = render_user_facing_result(_result())
        self.assertFalse(rendered["display_total_score"])
        self.assertIsNone(rendered["focus_feedback"])
        self.assertTrue(rendered["debug"]["special_mora_decisions"])
        self.assertFalse(any(item["user_feedback_allowed"] for item in rendered["debug"]["special_mora_decisions"]))

    def test_feature_flag_allows_calibrated_special_mora_feedback(self) -> None:
        result = _result(mora_table=[
            {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2},
            {"mora": "ー", "start_sec": 0.2, "end_sec": 0.23},
            {"mora": "メ", "start_sec": 0.23, "end_sec": 0.43},
            {"mora": "ン", "start_sec": 0.43, "end_sec": 0.63},
            {"mora": "ヲ", "start_sec": 0.63, "end_sec": 0.83},
            {"mora": "ク", "start_sec": 0.83, "end_sec": 1.03},
            {"mora": "ダ", "start_sec": 1.03, "end_sec": 1.23},
            {"mora": "サ", "start_sec": 1.23, "end_sec": 1.43},
            {"mora": "イ", "start_sec": 1.43, "end_sec": 1.63},
        ])
        rendered = render_user_facing_result(result, enable_user_facing_calibrated_special_mora=True)
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

    def test_special_mora_thresholds_can_be_loaded_from_json(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thresholds.json"
            path.write_text(json.dumps({"thresholds": {"long_vowel": {"low_ratio": 0.6, "high_ratio": 1.8}}}), encoding="utf-8")
            thresholds = load_special_mora_thresholds(path)
        self.assertEqual(thresholds["long_vowel"]["low_ratio"], 0.6)
        self.assertIn("sokuon", thresholds)

    def test_equal_fallback_suppresses_special_mora_correction(self) -> None:
        result = _result(alignment_mode="cached_dtw_fallback_equal")
        rows = score_special_mora_timing(result)
        self.assertTrue(all(row.status == "uncertain" for row in rows if row.type in {"long_vowel", "moraic_nasal"}))

    def test_runtime_missing_threshold_metadata_is_debug_uncertain(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing_thresholds.json"
            decisions = decide_special_mora_runtime(_result(), threshold_path=missing, enable_user_facing=True)
        self.assertTrue(decisions)
        self.assertTrue(all(item.decision == "uncertain" for item in decisions))
        self.assertTrue(all(not item.user_feedback_allowed for item in decisions))
        self.assertIn("missing_or_invalid_threshold_metadata", {item.suppression_reason for item in decisions})

    def test_sokuon_and_yoon_do_not_leak_to_user_facing(self) -> None:
        result = _result(
            target_text="きってきゃ",
            kana="キッテキャ",
            moras=["キ", "ッ", "テ", "キャ"],
            mora_table=[
                {"mora": "キ", "start_sec": 0.0, "end_sec": 0.2},
                {"mora": "ッ", "start_sec": 0.2, "end_sec": 0.22},
                {"mora": "テ", "start_sec": 0.22, "end_sec": 0.42},
                {"mora": "キャ", "start_sec": 0.42, "end_sec": 0.62},
            ],
            details={"mora_evidence": [{"judgement_available": True, "boundary_confidence": 0.9, "energy_coverage": 0.9} for _ in range(4)]},
        )
        decisions = decide_special_mora_runtime(result, enable_user_facing=True)
        by_type = {item.type: item for item in decisions}
        self.assertFalse(by_type["sokuon"].user_feedback_allowed)
        self.assertEqual(by_type["sokuon"].suppression_reason, "insufficient_native_evidence")
        self.assertFalse(by_type["yoon"].user_feedback_allowed)
        self.assertEqual(by_type["yoon"].suppression_reason, "debug_only_threshold")

    def test_special_mora_score_unavailable_is_not_zero(self) -> None:
        result = _result(moras=["バ", "グ"], mora_table=[{"start_sec": 0.0, "end_sec": 0.2}, {"start_sec": 0.2, "end_sec": 0.4}])
        decisions = decide_special_mora_runtime(result)
        self.assertIsNone(special_mora_score_from_decisions(decisions))
        rendered = render_user_facing_result(result)
        self.assertIsNone(rendered["debug"]["special_mora_score"])

    def test_counterfactual_decision_uses_runtime_threshold_function(self) -> None:
        threshold = {"status": "active", "low_ratio": 0.5, "high_ratio": 1.5}
        self.assertEqual(decide_special_mora_feature_value(threshold, 1.0), "ok")
        self.assertEqual(decide_special_mora_feature_value(threshold, 0.4), "too_short")
        self.assertEqual(decide_special_mora_feature_value(threshold, 1.6), "too_long")

    def test_shortened_feature_monotonically_increases_too_short_tendency(self) -> None:
        threshold = {"status": "active", "low_ratio": 0.5, "high_ratio": 1.5}
        values = [1.0, 0.8, 0.6, 0.4, 0.25]
        ranks = [1 if decide_special_mora_feature_value(threshold, value) == "too_short" else 0 for value in values]
        self.assertEqual(ranks, sorted(ranks))

    def test_weak_reference_uses_mild_special_mora_feedback(self) -> None:
        result = _result(
            details={"weak_reference": True},
            mora_table=[
                {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2},
                {"mora": "ー", "start_sec": 0.2, "end_sec": 0.22},
                {"mora": "メ", "start_sec": 0.22, "end_sec": 0.42},
                {"mora": "ン", "start_sec": 0.42, "end_sec": 0.62},
                {"mora": "ヲ", "start_sec": 0.62, "end_sec": 0.82},
                {"mora": "ク", "start_sec": 0.82, "end_sec": 1.02},
                {"mora": "ダ", "start_sec": 1.02, "end_sec": 1.22},
                {"mora": "サ", "start_sec": 1.22, "end_sec": 1.42},
                {"mora": "イ", "start_sec": 1.42, "end_sec": 1.62},
            ],
        )
        rows = score_special_mora_timing(result, weak_reference=True)
        self.assertTrue(any(row.status == "too_short" and row.message.startswith("参考として見ると") for row in rows))

    def test_only_one_user_facing_special_mora_feedback_is_emitted(self) -> None:
        result = _result(
            mora_table=[
                {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2},
                {"mora": "ー", "start_sec": 0.2, "end_sec": 0.22},
                {"mora": "メ", "start_sec": 0.22, "end_sec": 0.42},
                {"mora": "ン", "start_sec": 0.42, "end_sec": 0.44},
                {"mora": "ヲ", "start_sec": 0.44, "end_sec": 0.64},
                {"mora": "ク", "start_sec": 0.64, "end_sec": 0.84},
                {"mora": "ダ", "start_sec": 0.84, "end_sec": 1.04},
                {"mora": "サ", "start_sec": 1.04, "end_sec": 1.24},
                {"mora": "イ", "start_sec": 1.24, "end_sec": 1.44},
            ],
        )
        rendered = render_user_facing_result(result)
        special_messages = [msg for msg in rendered["user_messages"] if "自然" in msg or "聞き取りやすく" in msg]
        self.assertLessEqual(len(special_messages), 1)

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
