from __future__ import annotations

import unittest
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.audit_fixed_reference_scoring import write_markdown_summary
from scripts.build_fixed_reference_audit_manifest import (
    ALLOWED_EXPECTED_BEHAVIORS,
    FIELDNAMES as AUDIT_MANIFEST_FIELDNAMES,
    build_manifest,
    normalize_row,
    validate_rows,
)
from jp_speech_eval.asr_confirmation import build_confirmed_weak_target
from jp_speech_eval.eval_modes import evaluate_asr_confirmed_weak_reference, evaluate_mode
from jp_speech_eval.eval_modes import _known_pregenerated_reference_cache
from jp_speech_eval.feedback_renderer import render_user_facing_result
from jp_speech_eval.user_facing_policy import load_user_facing_messages
from jp_speech_eval.scoring_policy import policy_from_result
from jp_speech_eval.special_mora_scorer import (
    decide_special_mora_feature_value,
    decide_special_mora_runtime,
    load_special_mora_thresholds,
    score_special_mora_timing,
    special_mora_score_from_decisions,
)
from jp_speech_eval.special_mora_profiles import load_threshold_profile


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
    def test_human_checked_fixed_reference_shows_mild_special_mora_when_evidence_is_strong(self) -> None:
        rendered = render_user_facing_result(_result(mora_table=[
            {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2},
            {"mora": "ー", "start_sec": 0.2, "end_sec": 0.23},
            {"mora": "メ", "start_sec": 0.23, "end_sec": 0.43},
            {"mora": "ン", "start_sec": 0.43, "end_sec": 0.63},
            {"mora": "ヲ", "start_sec": 0.63, "end_sec": 0.83},
            {"mora": "ク", "start_sec": 0.83, "end_sec": 1.03},
            {"mora": "ダ", "start_sec": 1.03, "end_sec": 1.23},
            {"mora": "サ", "start_sec": 1.23, "end_sec": 1.43},
            {"mora": "イ", "start_sec": 1.43, "end_sec": 1.63},
        ]))
        self.assertFalse(rendered["display_total_score"])
        self.assertEqual(rendered["focus_feedback"]["category"], "special_mora")
        self.assertIn("全体としては問題ありません", rendered["focus_feedback"]["message"])
        self.assertTrue(rendered["debug"]["special_mora_decisions"])
        self.assertTrue(any(item["user_feedback_allowed"] for item in rendered["debug"]["special_mora_decisions"]))

    def test_weak_reference_still_blocks_special_mora_user_feedback(self) -> None:
        result = _result(
            mora_table=[
                {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2},
                {"mora": "ー", "start_sec": 0.2, "end_sec": 0.23},
                {"mora": "メ", "start_sec": 0.23, "end_sec": 0.43},
                {"mora": "ン", "start_sec": 0.43, "end_sec": 0.63},
                {"mora": "ヲ", "start_sec": 0.63, "end_sec": 0.83},
                {"mora": "ク", "start_sec": 0.83, "end_sec": 1.03},
                {"mora": "ダ", "start_sec": 1.03, "end_sec": 1.23},
                {"mora": "サ", "start_sec": 1.23, "end_sec": 1.43},
                {"mora": "イ", "start_sec": 1.43, "end_sec": 1.63},
            ],
            details={"mode": "asr_pseudo_reference", "weak_reference": True},
        )
        rendered = render_user_facing_result(result, mode="asr_pseudo_reference")
        self.assertEqual(rendered["focus_feedback"]["category"], "weak_reference")
        self.assertFalse(any(item["user_feedback_allowed"] for item in rendered["debug"]["special_mora_decisions"]))

    def test_legacy_threshold_metadata_blocks_user_facing_even_with_flag(self) -> None:
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
        rendered = render_user_facing_result(result, special_mora_threshold_profile="v1_debug", enable_user_facing_calibrated_special_mora=True)
        self.assertIsNone(rendered["focus_feedback"])
        reasons = {item["suppression_reason"] for item in rendered["debug"]["special_mora_decisions"]}
        self.assertTrue({"legacy_threshold_metadata", "debug_only_by_profile", "missing_or_invalid_threshold_metadata"}.intersection(reasons))

    def test_auto_pyopenjtalk_blocks_pitch_feedback(self) -> None:
        result = _result(details={"pitch_target_source": "auto_pyopenjtalk", "verified_level": "auto_pyopenjtalk"})
        rendered = render_user_facing_result(result)
        self.assertFalse(rendered["debug"]["scoring_policy"]["allow_pitch_feedback"])
        self.assertFalse(any("音高" in msg for msg in rendered["user_messages"]))

    def test_pitch_or_prosody_feedback_is_hidden_when_f0_evidence_is_low(self) -> None:
        result = _result(
            feedback=["イントネーションはよくできています。", "pitch is close to the reference."],
            details={"reliability": {"level": "medium", "overall": 0.8, "alignment": 0.8, "f0_coverage": 0.2}},
        )
        rendered = render_user_facing_result(result)
        joined = "\n".join(rendered["user_messages"])
        self.assertFalse(rendered["debug"]["reliability_gate"]["allow_pitch_feedback"])
        self.assertNotIn("イントネーション", joined)
        self.assertNotIn("pitch", joined.lower())

    def test_fallback_alignment_suppresses_pitch_even_when_raw_prosody_is_high(self) -> None:
        result = _result(
            alignment_mode="cached_dtw_fallback_equal",
            prosody_score=99,
            feedback=["韵律很好，アクセントも自然です。"],
        )
        rendered = render_user_facing_result(result)
        joined = "\n".join(rendered["user_messages"])
        self.assertIsNotNone(rendered["display_score"])
        self.assertEqual(rendered["practice_score"]["value"], rendered["display_score"])
        self.assertFalse(rendered["debug"]["reliability_gate"]["allow_pitch_feedback"])
        self.assertFalse(rendered["detail_feedback_allowed"])
        self.assertNotEqual(rendered["status"], "debug_only")
        self.assertEqual(rendered["debug"]["prosody_score"], 99)
        self.assertNotIn("韵律", joined)
        self.assertNotIn("アクセント", joined)

    def test_asr_raw_result_cannot_score(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_mode("asr_pseudo_reference", "dummy.wav", cache_path="cache/ramen_kudasai")

    def test_known_asr_reference_uses_pregenerated_aivis_cache(self) -> None:
        prefix = _known_pregenerated_reference_cache("ラーメンをください")
        self.assertIsNotNone(prefix)
        self.assertEqual(prefix.name, "ramen_kudasai_aivis")
        self.assertTrue(prefix.with_suffix(".ref.wav").exists())

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

    def test_equal_fallback_keeps_broad_score_and_hides_local_detail(self) -> None:
        rendered = render_user_facing_result(_result(alignment_mode="cached_dtw_fallback_equal"))
        self.assertIsNotNone(rendered["display_score"])
        self.assertIsNotNone(rendered["pronunciation_clarity_score"])
        self.assertIn("alignment_fallback_broad_score_only", rendered["score_policy_warnings"])
        self.assertFalse(rendered["detail_feedback_allowed"])
        dims = {item["key"]: item for item in rendered["score_dimensions"]}
        self.assertTrue(dims["pronunciation_clarity"]["available"])
        self.assertTrue(dims["mora_rhythm"]["available"])
        self.assertTrue(dims["delivery_fluency"]["available"])
        self.assertFalse(dims["pitch_accent"]["available"])

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
        self.assertEqual(by_type["sokuon"].suppression_reason, "blocked_by_profile")
        self.assertFalse(by_type["yoon"].user_feedback_allowed)
        self.assertEqual(by_type["yoon"].suppression_reason, "debug_only_by_profile")

    def test_special_mora_score_unavailable_is_not_zero(self) -> None:
        result = _result(moras=["バ", "グ"], mora_table=[{"start_sec": 0.0, "end_sec": 0.2}, {"start_sec": 0.2, "end_sec": 0.4}])
        decisions = decide_special_mora_runtime(result)
        self.assertIsNone(special_mora_score_from_decisions(decisions))
        rendered = render_user_facing_result(result)
        self.assertIsNone(rendered["debug"]["special_mora_score"])

    def test_v2_too_long_is_debug_only_and_near_boundary_suppressed(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thresholds_v2.json"
            path.write_text(json.dumps({"thresholds": {
                "long_vowel": {
                    "status": "active",
                    "debug_low": 0.5,
                    "debug_high": 1.1,
                    "low_ratio": 0.5,
                    "high_ratio": 1.1,
                    "user_low": 0.25,
                    "user_high": None,
                    "user_feedback_direction": "too_short_only",
                    "near_boundary_margin": 0.03,
                    "rollout_status": "limited_candidate",
                }
            }}), encoding="utf-8")
            too_long = _result(mora_table=[
                {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.1},
                {"mora": "ー", "start_sec": 0.1, "end_sec": 0.6},
                {"mora": "メ", "start_sec": 0.6, "end_sec": 0.7},
                {"mora": "ン", "start_sec": 0.7, "end_sec": 0.8},
                {"mora": "ヲ", "start_sec": 0.8, "end_sec": 0.9},
                {"mora": "ク", "start_sec": 0.9, "end_sec": 1.0},
                {"mora": "ダ", "start_sec": 1.0, "end_sec": 1.1},
                {"mora": "サ", "start_sec": 1.1, "end_sec": 1.2},
                {"mora": "イ", "start_sec": 1.2, "end_sec": 1.3},
            ])
            decisions = decide_special_mora_runtime(too_long, threshold_path=path, threshold_profile="v2_limited_candidate", mode_name="reference_based", enable_user_facing=True)
            long_vowel = next(item for item in decisions if item.type == "long_vowel")
            self.assertEqual(long_vowel.decision, "too_long")
            self.assertFalse(long_vowel.user_feedback_allowed)
            self.assertEqual(long_vowel.suppression_reason, "no_correction_needed")

            near = _result(mora_table=[
                {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2},
                {"mora": "ー", "start_sec": 0.2, "end_sec": 0.245},
                {"mora": "メ", "start_sec": 0.245, "end_sec": 0.445},
                {"mora": "ン", "start_sec": 0.445, "end_sec": 0.645},
                {"mora": "ヲ", "start_sec": 0.645, "end_sec": 0.845},
                {"mora": "ク", "start_sec": 0.845, "end_sec": 1.045},
                {"mora": "ダ", "start_sec": 1.045, "end_sec": 1.245},
                {"mora": "サ", "start_sec": 1.245, "end_sec": 1.445},
                {"mora": "イ", "start_sec": 1.445, "end_sec": 1.645},
            ])
            near_decision = next(item for item in decide_special_mora_runtime(near, threshold_path=path, threshold_profile="v2_limited_candidate", mode_name="reference_based", enable_user_facing=True) if item.type == "long_vowel")
            self.assertTrue(near_decision.near_boundary)
            self.assertFalse(near_decision.user_feedback_allowed)
            self.assertEqual(near_decision.suppression_reason, "near_boundary_debug_only")

    def test_user_facing_threshold_is_stricter_than_debug_threshold(self) -> None:
        threshold = {"status": "active", "debug_low": 0.5, "debug_high": 1.5, "user_low": 0.25, "user_feedback_direction": "too_short_only"}
        self.assertEqual(decide_special_mora_feature_value(threshold, 0.4), "too_short")
        from jp_speech_eval.special_mora_scorer import decide_special_mora_user_feature_value
        self.assertEqual(decide_special_mora_user_feature_value(threshold, 0.4), "ok")

    def test_missing_threshold_profile_falls_back_to_default_safe(self) -> None:
        profile = load_threshold_profile("missing_profile_name")
        self.assertEqual(profile.profile_name, "default_safe")
        self.assertIn("unknown_profile", profile.fallback_reason)

    def test_default_safe_and_shadow_never_emit_user_facing(self) -> None:
        result = _result()
        for profile in ("default_safe", "v2_shadow"):
            rendered = render_user_facing_result(
                result,
                special_mora_threshold_profile=profile,
                enable_user_facing_calibrated_special_mora=True,
            )
            self.assertIsNone(rendered["focus_feedback"])
            self.assertFalse(any(item["user_feedback_allowed"] for item in rendered["debug"]["special_mora_decisions"]))

    def test_v2_limited_candidate_emits_allowed_types_by_default(self) -> None:
        result = _result(mora_table=[
            {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2},
            {"mora": "ー", "start_sec": 0.2, "end_sec": 0.235},
            {"mora": "メ", "start_sec": 0.235, "end_sec": 0.435},
            {"mora": "ン", "start_sec": 0.435, "end_sec": 0.635},
            {"mora": "ヲ", "start_sec": 0.635, "end_sec": 0.835},
            {"mora": "ク", "start_sec": 0.835, "end_sec": 1.035},
            {"mora": "ダ", "start_sec": 1.035, "end_sec": 1.235},
            {"mora": "サ", "start_sec": 1.235, "end_sec": 1.435},
            {"mora": "イ", "start_sec": 1.435, "end_sec": 1.635},
        ])
        rendered = render_user_facing_result(result, special_mora_threshold_profile="v2_limited_candidate")
        self.assertTrue(any(item["user_feedback_allowed"] for item in rendered["debug"]["special_mora_decisions"]))
        self.assertEqual(rendered["focus_feedback"]["category"], "special_mora")
        self.assertEqual(rendered["focus_feedback"]["type"], "long_vowel")

    def test_kanade_demo_cannot_enable_special_mora_correction(self) -> None:
        result = _result(details={"mode": "kanade_asr_voice_reference", "demo_only": True})
        rendered = render_user_facing_result(
            result,
            mode="kanade_asr_voice_reference",
            special_mora_threshold_profile="v2_limited_candidate",
            enable_user_facing_calibrated_special_mora=True,
        )
        self.assertEqual(rendered["focus_feedback"]["category"], "demo")
        self.assertFalse(any(item["user_feedback_allowed"] for item in rendered["debug"]["special_mora_decisions"]))

    def test_weak_reference_hint_disabled_by_default(self) -> None:
        result = _result(details={"weak_reference": True})
        rendered = render_user_facing_result(
            result,
            mode="asr_confirmed_weak_reference",
            special_mora_threshold_profile="v2_limited_candidate",
            enable_user_facing_calibrated_special_mora=True,
        )
        self.assertFalse(any(item["user_feedback_allowed"] for item in rendered["debug"]["special_mora_decisions"]))

    def test_evidence_card_fields_are_present(self) -> None:
        rendered = render_user_facing_result(_result(), special_mora_threshold_profile="v2_limited_candidate")
        cards = rendered["debug"]["special_mora_evidence_cards"]
        self.assertTrue(cards)
        for key in ("profile_name", "feature_value", "debug_low", "user_low", "suppression_reason"):
            self.assertIn(key, cards[0])

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
        self.assertEqual(rendered["practice_score"]["value"], rendered["display_score"])
        self.assertIn("練習用の目安", rendered["practice_score"]["explanation"])

    def test_weak_reference_keeps_broad_score_but_blocks_strict_pitch(self) -> None:
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
        self.assertIsNotNone(rendered["display_score"])
        self.assertIn("weak_reference_broad_score_only", rendered["score_policy_warnings"])
        self.assertTrue(rendered["debug"]["weak_reference"])
        self.assertNotEqual(rendered["status"], "debug_only")
        self.assertEqual(rendered["practice_score"]["value"], rendered["display_score"])
        dims = {item["key"]: item for item in rendered["score_dimensions"]}
        self.assertFalse(dims["pitch_accent"]["available"])

    def test_confirmed_weak_reference_downgrades_tts_pitch_proxy(self) -> None:
        fake_cache = Mock()
        fake_cache.meta.sr = 16000
        fake_eval = Mock()
        fake_eval.to_dict.return_value = _result(
            total_score=96,
            pronunciation_score=92,
            prosody_score=99,
            details={"prosody": {"contour_corr": 0.99, "transition_agreement": 0.99}},
        )
        with patch("jp_speech_eval.eval_modes.load_sentence_cache", return_value=fake_cache), \
             patch("jp_speech_eval.eval_modes._known_pregenerated_reference_cache", return_value=Path("cache/fake_ref")), \
             patch("jp_speech_eval.eval_modes.evaluate_utterance", return_value=fake_eval):
            result = evaluate_asr_confirmed_weak_reference(
                "user.wav",
                user_confirmed_text="ラーメンをください",
                base_cache_path="cache/base",
            )
        self.assertEqual(result["details"]["mode"], "asr_confirmed_weak_reference")
        self.assertTrue(result["details"]["weak_reference"])
        self.assertLessEqual(result["prosody_score"], 35)
        self.assertLessEqual(result["total_score"], 60)
        self.assertFalse(result["details"]["prosody"]["user_facing_available"])
        self.assertFalse(result["details"]["prosody"]["pitch_correctness_available"])
        self.assertEqual(result["details"]["prosody"]["raw_prosody_score_before_downgrade"], 99)

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

    def test_special_mora_score_does_not_enter_practice_score(self) -> None:
        base = _result(
            total_score=10,
            pronunciation_score=90,
            fluency_score=90,
            prosody_score=10,
            details={
                "fluency": {"rhythm_timing_score": 90, "delivery_fluency_score": 90},
                "pronunciation": {"mora_duration_cv": 0.08, "special_mora_penalty": 0},
            },
        )
        rendered = render_user_facing_result(base)
        self.assertGreaterEqual(rendered["practice_score"]["value"], 85)
        self.assertNotIn("ネイティブ", rendered["practice_score"]["explanation"])

    def test_user_facing_result_has_safe_contract_fields(self) -> None:
        rendered = render_user_facing_result(_result())
        for key in (
            "status",
            "practice_score",
            "confidence",
            "summary_text",
            "primary_suggestion_text",
            "suggestion_type",
            "mode_notice",
            "debug_available",
            "suppressed_reasons",
        ):
            self.assertIn(key, rendered)
        self.assertFalse(rendered["display_total_score"])

    def test_poor_recording_triggers_retry_status(self) -> None:
        rendered = render_user_facing_result(_result(details={"recording_quality": {"score": 0.1}}))
        self.assertEqual(rendered["status"], "retry")
        self.assertEqual(rendered["practice_score"]["label"], "録音を確認")
        self.assertIsNone(rendered["display_score"])
        self.assertIsNone(rendered["pronunciation_clarity_score"])
        self.assertIn("recording_unusable", rendered["suppressed_reasons"])

    def test_kanade_mode_is_debug_only_playback_notice(self) -> None:
        rendered = render_user_facing_result(
            _result(details={"mode": "kanade_asr_voice_reference", "demo_only": True}),
            mode="kanade_asr_voice_reference",
        )
        self.assertEqual(rendered["status"], "debug_only")
        self.assertIn("声", rendered["mode_notice"])
        self.assertIsNone(rendered["practice_score"]["value"])

    def test_user_facing_message_config_has_no_forbidden_wording(self) -> None:
        messages = load_user_facing_messages()
        forbidden = ("間違っています", "発音できていません", "ネイティブ度", "完全な発音正確度", "発音は不正確")
        joined = "\n".join(messages.values())
        for word in forbidden:
            self.assertNotIn(word, joined)
        self.assertEqual(messages["status.pass"], "全体としてよくできています。")

    def test_demo_fixed_targets_are_present_and_pitch_policy_is_clear(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "demo_fixed_targets.json"
        targets = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(targets), 5)
        by_id = {item["target_id"]: item for item in targets}
        self.assertEqual(by_id["ramen_kudasai"]["verified_level"], "human_checked")
        self.assertEqual(by_id["coffee_kudasai"]["verified_level"], "auto_pyopenjtalk")
        self.assertTrue(by_id["coffee_kudasai"]["moras"])

    def test_fixed_reference_audit_summary_flags_negative_pitch_or_score(self) -> None:
        rows = [
            {
                "sample_id": "english_1",
                "audio_type": "english",
                "reference_type": "pseudo_reference",
                "expected_behavior": "content_mismatch_should_not_score",
                "score_available": True,
                "display_score": 88,
                "display_score_before_cap": 94,
                "display_score_after_cap": 88,
                "display_cap_applied": True,
                "display_cap_reason": "pronunciation_margin_cap",
                "display_cap_reduction": 6,
                "pronunciation_score": 80,
                "raw_prosody_score": 95,
                "pitch_feedback_allowed": True,
                "pitch_text_leakage_warning": True,
                "pitch_text_leakage_terms": "pitch;accent",
                "alignment_gate": "ok",
                "content_gate": "pass",
                "recording_gate": "ok",
                "pronunciation_evidence_gate": "ok",
                "special_mora_user_facing_count": 0,
                "special_mora_suppressed": False,
                "special_mora_evidence_level": "",
                "special_mora_suppression_reason": "",
                "rhythm_timing_penalty_reason": "",
                "warning_codes": "",
                "suppressed_reasons": "",
                "user_message_type": "",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.md"
            write_markdown_summary(path, rows)
            text = path.read_text(encoding="utf-8")
        self.assertIn("Failures and Warnings", text)
        self.assertIn("FAIL_negative_user_facing_score", text)
        self.assertIn("WARN_pitch_text_leakage", text)
        self.assertIn("Recommended Next Action", text)
        self.assertIn("Do not tune pronunciation_score yet", text)
        self.assertIn("english_1", text)

    def test_fixed_reference_audit_summary_flags_native_and_bad_learner_groups(self) -> None:
        rows = [
            {
                "sample_id": "native_low",
                "audio_type": "native",
                "reference_type": "human",
                "expected_behavior": "native_should_score_high",
                "score_available": True,
                "display_score": 72,
                "display_score_before_cap": 72,
                "display_score_after_cap": 72,
                "display_cap_applied": False,
                "display_cap_reason": "",
                "display_cap_reduction": 0,
                "pronunciation_score": 72,
                "raw_prosody_score": 70,
                "pitch_feedback_allowed": False,
                "pitch_text_leakage_warning": False,
                "pitch_text_leakage_terms": "",
                "alignment_gate": "ok",
                "content_gate": "pass",
                "recording_gate": "ok",
                "pronunciation_evidence_gate": "ok",
                "special_mora_user_facing_count": 1,
                "special_mora_suppressed": False,
                "special_mora_evidence_level": "high",
                "special_mora_suppression_reason": "",
                "rhythm_timing_penalty_reason": "",
                "warning_codes": "",
                "suppressed_reasons": "",
                "user_message_type": "",
            },
            {
                "sample_id": "bad_high",
                "audio_type": "learner_bad",
                "reference_type": "pseudo_reference",
                "expected_behavior": "bad_learner_should_not_score_high",
                "score_available": True,
                "display_score": 86,
                "display_score_before_cap": 96,
                "display_score_after_cap": 86,
                "display_cap_applied": True,
                "display_cap_reason": "pronunciation_margin_cap",
                "display_cap_reduction": 10,
                "pronunciation_score": 81,
                "raw_prosody_score": 99,
                "pitch_feedback_allowed": False,
                "pitch_text_leakage_warning": False,
                "pitch_text_leakage_terms": "",
                "alignment_gate": "ok",
                "content_gate": "pass",
                "recording_gate": "ok",
                "pronunciation_evidence_gate": "ok",
                "special_mora_user_facing_count": 0,
                "special_mora_suppressed": True,
                "special_mora_evidence_level": "low",
                "special_mora_suppression_reason": "low_evidence",
                "rhythm_timing_penalty_reason": "",
                "warning_codes": "",
                "suppressed_reasons": "",
                "user_message_type": "",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.md"
            write_markdown_summary(path, rows)
            text = path.read_text(encoding="utf-8")
        self.assertIn("WARN_native_median_display_low", text)
        self.assertIn("WARN_special_mora_native_user_facing", text)
        self.assertIn("WARN_bad_learner_suspicious_high_rate", text)
        self.assertIn("reduction_gte_10", text)
        self.assertIn("bad_high", text)

    def test_fixed_reference_audit_summary_has_experiment_overview_and_suspicious_lists(self) -> None:
        rows = [
            {
                "sample_id": "wrong_1",
                "audio_type": "wrong_japanese_sentence",
                "reference_type": "tts_reference",
                "expected_behavior": "content_mismatch_should_not_score",
                "score_available": True,
                "display_score": 82,
                "display_score_before_cap": 88,
                "display_score_after_cap": 82,
                "display_cap_applied": True,
                "display_cap_reason": "pronunciation_margin_cap",
                "display_cap_reduction": 6,
                "pronunciation_score": 77,
                "raw_prosody_score": 90,
                "pitch_feedback_allowed": True,
                "pitch_text_leakage_warning": False,
                "pitch_text_leakage_terms": "",
                "alignment_gate": "ok",
                "content_gate": "pass",
                "recording_gate": "ok",
                "pronunciation_evidence_gate": "ok",
                "special_mora_user_facing_count": 0,
                "special_mora_suppressed": False,
                "special_mora_evidence_level": "",
                "special_mora_suppression_reason": "",
                "rhythm_timing_penalty_reason": "",
                "warning_codes": "",
                "suppressed_reasons": "",
                "user_message_type": "",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.md"
            write_markdown_summary(path, rows, manifest_path="data/audit/fixed_reference_manifest_v0.csv")
            text = path.read_text(encoding="utf-8")
        self.assertIn("Experiment Overview", text)
        self.assertIn("Main Diagnostic Questions", text)
        self.assertIn("Suspicious Sample List", text)
        self.assertIn("negative_controls_with_display_score: wrong_1", text)
        self.assertIn("negative_controls_with_pitch_feedback_allowed: wrong_1", text)
        self.assertIn("Decision Hints", text)
        self.assertIn("Recommended Next Action", text)

    def test_fixed_reference_manifest_template_has_expected_behavior_column(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "audit" / "fixed_reference_manifest_template.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        self.assertTrue(rows)
        self.assertEqual(list(rows[0].keys()), AUDIT_MANIFEST_FIELDNAMES)
        expected = {row["expected_behavior"] for row in rows}
        self.assertIn("native_should_score_high", expected)
        self.assertIn("bad_learner_should_not_score_high", expected)
        self.assertIn("content_mismatch_should_not_score", expected)
        self.assertTrue(all("placeholder" in row["notes"] for row in rows))

    def test_fixed_reference_manifest_allowed_expected_behaviors_are_recognized(self) -> None:
        rows = []
        for expected in sorted(ALLOWED_EXPECTED_BEHAVIORS):
            rows.append({
                "sample_id": expected,
                "audio_path": "assets/reference_cache/ramen_kudasai_aivis.ref.wav",
                "target_text": "ラーメンをください",
                "audio_type": "self_recording_clear",
                "reference_type": "tts_reference",
                "expected_behavior": expected,
                "notes": "",
            })
        report = validate_rows(rows, strict=True)
        self.assertTrue(report.ok)
        self.assertEqual(report.unknown_expected_behavior_count, 0)
        self.assertEqual(report.missing_audio_path_count, 0)

    def test_fixed_reference_manifest_unknown_expected_behavior_warns_or_strict_fails(self) -> None:
        row = {
            "sample_id": "unknown_case",
            "audio_path": "data/ramen.wav",
            "target_text": "ラーメンをください",
            "audio_type": "self_recording_clear",
            "reference_type": "tts_reference",
            "expected_behavior": "unknown_behavior",
            "notes": "",
        }
        loose = validate_rows([row], strict=False)
        strict = validate_rows([row], strict=True)
        self.assertTrue(loose.ok)
        self.assertTrue(loose.warnings)
        self.assertFalse(strict.ok)
        self.assertTrue(strict.errors)
        self.assertEqual(strict.unknown_expected_behavior_count, 1)

    def test_build_fixed_reference_audit_manifest_maps_group_to_expected_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "manual.csv"
            source.write_text(
                "sample_id,audio_path,target_text,group,notes\n"
                "bad_1,audio/bad.wav,ラーメンをください,janon_learner_bad,bad learner\n"
                "eng_1,audio/eng.wav,ラーメンをください,english_or_chinese_speech,negative\n",
                encoding="utf-8",
            )
            out = Path(tmp) / "manifest.csv"
            build_manifest(source, out)
            rows = list(csv.DictReader(out.open(encoding="utf-8")))
        self.assertEqual(rows[0]["audio_type"], "janon_learner_bad")
        self.assertEqual(rows[0]["expected_behavior"], "bad_learner_should_not_score_high")
        self.assertEqual(rows[1]["expected_behavior"], "content_mismatch_should_not_score")

    def test_fixed_reference_manifest_normalizes_legacy_group_rows(self) -> None:
        row = normalize_row({
            "sample_id": "native_1",
            "audio_path": "data/ramen.wav",
            "target_text": "ラーメンをください",
            "group": "jvs_native_clear",
        })
        self.assertEqual(row["audio_type"], "jvs_native_clear")
        self.assertEqual(row["reference_type"], "human_reference")
        self.assertEqual(row["expected_behavior"], "native_should_score_high")

    def test_build_manifest_dry_run_prints_counts_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "manual.csv"
            out = Path(tmp) / "manifest.csv"
            source.write_text(
                "sample_id,audio_path,target_text,group,notes\n"
                "native_1,data/ramen.wav,ラーメンをください,jvs_native_clear,native\n"
                "bad_1,data/ramen.wav,ラーメンをください,janon_learner_bad,bad\n",
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_fixed_reference_audit_manifest.py",
                    "--input",
                    str(source),
                    "--out",
                    str(out),
                    "--dry-run",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("total rows: 2", proc.stdout)
        self.assertIn("expected_behavior counts", proc.stdout)
        self.assertIn("audio_type counts", proc.stdout)
        self.assertFalse(out.exists())

    def test_build_manifest_strict_fails_on_missing_audio_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "manual.csv"
            source.write_text(
                "sample_id,audio_path,target_text,audio_type,reference_type,expected_behavior,notes\n"
                "missing_1,missing/audio.wav,ラーメンをください,self_recording_clear,tts_reference,clear_learner_should_score,missing\n",
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_fixed_reference_audit_manifest.py",
                    "--input",
                    str(source),
                    "--dry-run",
                    "--strict",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("audio_path does not exist", proc.stdout)

    def test_run_fixed_reference_audit_v0_script_prompts_when_manifest_is_missing(self) -> None:
        proc = subprocess.run(
            ["bash", "scripts/run_fixed_reference_audit_v0.sh", "/tmp/jp_speech_eval_missing_manifest_v0.csv"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("Missing manifest", proc.stderr)
        self.assertIn("fixed_reference_manifest_template.csv", proc.stderr)

    def test_fixed_reference_smoke_manifest_is_valid_for_pipeline_only(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "audit" / "fixed_reference_manifest_smoke.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        self.assertGreaterEqual(len(rows), 3)
        self.assertTrue(all("smoke only" in row["notes"] for row in rows))
        report = validate_rows(rows, strict=True)
        self.assertTrue(report.ok)
        self.assertEqual(report.missing_audio_path_count, 0)

    def test_fixed_reference_audit_summary_does_not_modify_scoring_config(self) -> None:
        config = Path(__file__).resolve().parents[1] / "configs" / "scoring_config.json"
        before = config.read_text(encoding="utf-8")
        rows = [
            {
                "sample_id": "ok_1",
                "audio_type": "self_recording_clear",
                "reference_type": "tts_reference",
                "expected_behavior": "clear_learner_should_score",
                "score_available": True,
                "display_score": 82,
                "display_score_before_cap": 82,
                "display_score_after_cap": 82,
                "display_cap_applied": False,
                "display_cap_reason": "",
                "display_cap_reduction": 0,
                "pronunciation_score": 82,
                "raw_prosody_score": 80,
                "pitch_feedback_allowed": False,
                "pitch_text_leakage_warning": False,
                "alignment_gate": "ok",
                "content_gate": "pass",
                "recording_gate": "ok",
                "pronunciation_evidence_gate": "ok",
                "special_mora_user_facing_count": 0,
                "special_mora_suppressed": False,
                "special_mora_evidence_level": "",
                "special_mora_suppression_reason": "",
                "rhythm_timing_penalty_reason": "",
                "warning_codes": "",
                "suppressed_reasons": "",
                "user_message_type": "",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            write_markdown_summary(Path(tmp) / "summary.md", rows)
        after = config.read_text(encoding="utf-8")
        self.assertEqual(before, after)

    def test_demo_smoke_test_script_generates_expected_rows(self) -> None:
        from scripts.run_demo_flow_smoke_tests import run

        rows = run()
        self.assertEqual(len(rows), 11)
        self.assertTrue(all(row["passed"] for row in rows))
        kanade = next(row for row in rows if row["scenario"] == "kanade_excluded_from_scoring")
        self.assertFalse(kanade["kanade_scoring_leakage"])

    def test_too_long_never_user_facing(self) -> None:
        result = _result(mora_table=[
            {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.1},
            {"mora": "ー", "start_sec": 0.1, "end_sec": 0.7},
            {"mora": "メ", "start_sec": 0.7, "end_sec": 0.8},
            {"mora": "ン", "start_sec": 0.8, "end_sec": 0.9},
            {"mora": "ヲ", "start_sec": 0.9, "end_sec": 1.0},
            {"mora": "ク", "start_sec": 1.0, "end_sec": 1.1},
            {"mora": "ダ", "start_sec": 1.1, "end_sec": 1.2},
            {"mora": "サ", "start_sec": 1.2, "end_sec": 1.3},
            {"mora": "イ", "start_sec": 1.3, "end_sec": 1.4},
        ])
        rendered = render_user_facing_result(
            result,
            special_mora_threshold_profile="v2_limited_candidate",
            enable_user_facing_calibrated_special_mora=True,
        )
        self.assertFalse(any(item["decision"] == "too_long" and item["user_feedback_allowed"] for item in rendered["debug"]["special_mora_decisions"]))


if __name__ == "__main__":
    unittest.main()
