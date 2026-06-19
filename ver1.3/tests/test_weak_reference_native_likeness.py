from __future__ import annotations

import csv
import unittest
from pathlib import Path

from jp_speech_eval.feedback_renderer import render_user_facing_result
from jp_speech_eval.scoring import score_weak_reference_native_likeness


ROOT = Path(__file__).resolve().parents[1]
AUDIT_CSV = ROOT / "results" / "calibration" / "weak_reference_native_likeness_audit.csv"


def _fake_result(**overrides):
    result = {
        "target_text": "今日はいい天気です",
        "kana": "キョウハイイテンキデス",
        "moras": ["キョ", "ウ", "ハ", "イ", "イ", "テ", "ン", "キ", "デ", "ス"],
        "mora_table": [],
        "total_score": 62,
        "pronunciation_score": 88,
        "prosody_score": 38,
        "fluency_score": 91,
        "tone_score": 99,
        "weak_pronunciation_naturalness_score": 88,
        "weak_prosody_naturalness_score": 92,
        "weak_rhythm_naturalness_score": 90,
        "weak_overall_practice_score": 90,
        "score_type": "weak_reference_native_likeness",
        "strict_reference_available": False,
        "feedback": ["音高变化仅供参考，不等同于严格高低重音判定。"],
        "alignment_mode": "cached_dtw",
        "details": {
            "mode": "asr_confirmed_weak_reference",
            "weak_reference": True,
            "verified_level": "auto_pyopenjtalk",
            "pitch_target_source": "openjtalk_accent_phrase_chain",
            "pitch_target_reliability": "heuristic",
            "reference_source": "tts_pseudo_reference",
            "reliability": {"level": "high", "overall": 0.92, "alignment": 0.9, "f0_coverage": 0.95},
            "recording_quality": {"score": 0.95},
            "content_match": {"status": "pass"},
            "alignment": {"mode": "cached_dtw"},
            "pronunciation": {"mora_duration_cv": 0.12, "special_mora_diagnostics": []},
            "prosody": {
                "pitch_target_source": "openjtalk_accent_phrase_chain",
                "pitch_target_reliability": "heuristic",
                "contour_corr": -0.2,
                "transition_agreement": 0.3,
            },
            "weak_reference_native_likeness": {
                "score_type": "weak_reference_native_likeness",
                "weak_pronunciation_naturalness_score": 88,
                "weak_prosody_naturalness_score": 92,
                "weak_rhythm_naturalness_score": 90,
                "weak_overall_practice_score": 90,
                "strict_reference_available": False,
                "strict_pitch_accent_correctness": False,
                "uses_openjtalk_as_strict_reference": False,
            },
            "fluency": {"rhythm_timing_score": 90, "delivery_fluency_score": 91},
        },
    }
    for key, value in overrides.items():
        if key == "details":
            result["details"].update(value)
        else:
            result[key] = value
    return result


class WeakReferenceNativeLikenessTests(unittest.TestCase):
    def test_native_like_contour_scores_above_flat_and_random(self) -> None:
        native, _fb, native_details = score_weak_reference_native_likeness(
            [100, 120, 145, 138, 126, 115, 122, 108, 96, 92]
        )
        flat, _flat_fb, flat_details = score_weak_reference_native_likeness([120] * 10)
        random, _random_fb, random_details = score_weak_reference_native_likeness(
            [100, 170, 92, 160, 105, 180, 96, 155, 102, 175]
        )
        self.assertIsNotNone(native)
        self.assertGreaterEqual(native, 85)
        self.assertLess(flat, native)
        self.assertLess(random, native)
        self.assertFalse(native_details["strict_pitch_accent_correctness"])
        self.assertFalse(native_details["uses_openjtalk_as_strict_reference"])
        self.assertGreater(flat_details["flatness_penalty"], native_details["flatness_penalty"])
        self.assertGreater(random_details["instability_penalty"], native_details["instability_penalty"])

    def test_low_f0_coverage_is_unavailable(self) -> None:
        score, feedback, details = score_weak_reference_native_likeness(
            [100, float("nan"), float("nan"), float("nan"), float("nan"), 95]
        )
        self.assertIsNone(score)
        self.assertFalse(details["available"])
        self.assertEqual(details["unavailable_reason"], "insufficient_valid_mora_f0")
        self.assertTrue(any("音高信息不够清楚" in item for item in feedback))

    def test_asr_confirmed_weak_reference_uses_weak_score_type(self) -> None:
        rendered = render_user_facing_result(_fake_result(), mode="asr_confirmed_weak_reference")
        self.assertEqual(rendered["debug"]["score_type"], "weak_reference_native_likeness")
        self.assertEqual(rendered["debug"]["weak_prosody_naturalness_score"], 92)
        self.assertEqual(rendered["debug"]["visible_prosody_score"], 92)
        self.assertEqual(rendered["display_score"], 90)
        self.assertFalse(rendered["debug"]["prosody_debug"]["strict_pitch_accent_correctness"])

    def test_openjtalk_only_target_is_not_strict_pitch_correctness(self) -> None:
        rendered = render_user_facing_result(_fake_result(), mode="asr_confirmed_weak_reference")
        self.assertEqual(rendered["debug"]["prosody_debug"]["pitch_target_reliability"], "heuristic")
        self.assertEqual(rendered["debug"]["visible_prosody_score"], 92)
        self.assertNotEqual(rendered["debug"]["visible_prosody_score"], rendered["debug"]["prosody_score"])

    def test_weak_pitch_feedback_does_not_assert_accent_error(self) -> None:
        rendered = render_user_facing_result(_fake_result(), mode="asr_confirmed_weak_reference")
        text = " ".join(rendered.get("user_messages") or []) + " " + str(rendered.get("primary_suggestion_text") or "")
        self.assertNotIn("高低重音错", text)
        self.assertNotIn("アクセントが違", text)

    def test_content_mismatch_still_vetoes_weak_reference(self) -> None:
        rendered = render_user_facing_result(_fake_result(details={
            "content_match": {
                "status": "pass",
                "asr_provider": "whisper",
                "transcript": "please give me ramen",
                "transcript_kana": "プリーズギブミーラーメン",
                "target_kana": "キョウハイイテンキデス",
                "kana_similarity": 0.1,
            }
        }), mode="asr_confirmed_weak_reference")
        self.assertEqual(rendered["status"], "retry")
        self.assertIsNone(rendered["display_score"])
        self.assertIn("content_mismatch_veto", rendered["suppressed_reasons"])

    def test_verified_fixed_reference_path_still_uses_visible_strict_prosody(self) -> None:
        result = _fake_result(
            score_type="strict_reference",
            strict_reference_available=True,
            prosody_score=87,
            weak_prosody_naturalness_score=None,
            details={
                "mode": "reference_based",
                "weak_reference": False,
                "verified_level": "human_checked",
                "pitch_target_source": "reference_audio_f0_cache",
                "pitch_target_reliability": "reliable",
                "prosody": {"pitch_target_source": "reference_audio_f0_cache", "pitch_target_reliability": "reliable"},
                "weak_reference_native_likeness": {},
            },
        )
        rendered = render_user_facing_result(result, mode="reference_based")
        self.assertEqual(rendered["debug"]["visible_prosody_score"], 87)
        self.assertFalse(rendered["debug"]["weak_reference"])

    def test_tone_score_not_in_core_four(self) -> None:
        html = (ROOT / "debug_ui" / "index.html").read_text(encoding="utf-8")
        score_key_block = html.split("const scoreKeys = [", 1)[1].split("];", 1)[0]
        self.assertNotIn('"tone_score"', score_key_block)

    def test_jvs_audit_native_high_and_controls_lower(self) -> None:
        self.assertTrue(AUDIT_CSV.exists())
        rows = list(csv.DictReader(AUDIT_CSV.open(encoding="utf-8")))
        def mean(case: str) -> float:
            vals = [
                float(row["weak_prosody_naturalness_score"])
                for row in rows
                if row.get("case_name") == case and row.get("weak_prosody_naturalness_score")
            ]
            self.assertTrue(vals, case)
            return sum(vals) / len(vals)
        native = mean("jvs_native")
        flat = mean("flat_pitch_control")
        random = mean("shuffled_random_pitch_control")
        self.assertGreaterEqual(native, 85.0)
        self.assertLess(flat, native - 25.0)
        self.assertLess(random, native - 20.0)


if __name__ == "__main__":
    unittest.main()
