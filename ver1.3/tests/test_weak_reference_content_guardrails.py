from __future__ import annotations

import csv
import unittest
from pathlib import Path

from jp_speech_eval.feedback_renderer import render_user_facing_result
from jp_speech_eval.weak_reference_guardrails import apply_weak_overall_guardrail


ROOT = Path(__file__).resolve().parents[1]
AUDIT_CSV = ROOT / "results" / "calibration" / "weak_reference_native_likeness_audit.csv"


def _guardrail_for_text(text: str, *, moras: list[str], score: int = 92, **overrides):
    return apply_weak_overall_guardrail(
        weak_overall_score=score,
        target_text=text,
        kana=overrides.pop("kana", ""),
        moras=moras,
        duration_sec=overrides.pop("duration_sec", 1.2),
        content_match=overrides.pop("content_match", {"status": "pass"}),
        weak_prosody_details=overrides.pop(
            "weak_prosody_details",
            {"available": True, "f0_coverage": 0.92, "valid_f0_mora_count": max(3, len(moras)), "voiced_mora_count": max(3, len(moras))},
        ),
        mora_evidence_summary=overrides.pop("mora_evidence_summary", {"judgement_available_count": len(moras)}),
    )


def _weak_result(**overrides):
    result = {
        "target_text": "今日はいい天気です",
        "kana": "キョウハイイテンキデス",
        "moras": ["キョ", "ウ", "ハ", "イ", "イ", "テ", "ン", "キ", "デ", "ス"],
        "mora_table": [],
        "total_score": 60,
        "pronunciation_score": 88,
        "prosody_score": 35,
        "fluency_score": 91,
        "tone_score": 99,
        "weak_pronunciation_naturalness_score": 88,
        "weak_prosody_naturalness_score": 92,
        "weak_rhythm_naturalness_score": 90,
        "weak_overall_practice_score": 90,
        "score_type": "weak_reference_native_likeness",
        "strict_reference_available": False,
        "feedback": [],
        "alignment_mode": "cached_dtw",
        "details": {
            "mode": "asr_confirmed_weak_reference",
            "weak_reference": True,
            "pitch_target_source": "openjtalk_accent_phrase_chain",
            "pitch_target_reliability": "heuristic",
            "verified_level": "auto_pyopenjtalk",
            "reliability": {"level": "high", "overall": 0.92, "alignment": 0.9, "f0_coverage": 0.95},
            "recording_quality": {"score": 0.95},
            "content_match": {"status": "pass"},
            "alignment": {"mode": "cached_dtw"},
            "pronunciation": {},
            "prosody": {"pitch_target_source": "openjtalk_accent_phrase_chain", "pitch_target_reliability": "heuristic"},
            "fluency": {"rhythm_timing_score": 90, "delivery_fluency_score": 91},
            "weak_reference_native_likeness": {
                "score_type": "weak_reference_native_likeness",
                "weak_pronunciation_naturalness_score": 88,
                "weak_prosody_naturalness_score": 92,
                "weak_rhythm_naturalness_score": 90,
                "weak_overall_practice_score": 90,
                "weak_overall_guardrail": {
                    "status": "ok",
                    "display_allowed": True,
                    "reasons": [],
                    "weak_overall_practice_score_after_guardrail": 90,
                },
                "strict_reference_available": False,
                "strict_pitch_accent_correctness": False,
                "uses_openjtalk_as_strict_reference": False,
            },
        },
    }
    for key, value in overrides.items():
        if key == "details":
            result["details"].update(value)
        else:
            result[key] = value
    return result


class WeakReferenceContentGuardrailTests(unittest.TestCase):
    def test_random_english_has_no_formal_weak_overall(self) -> None:
        guardrail = _guardrail_for_text("please give me ramen", moras=[], score=95)
        self.assertEqual(guardrail["status"], "no_score")
        self.assertIsNone(guardrail["weak_overall_practice_score_after_guardrail"])
        self.assertIn("latin_dominant_confirmed_text", guardrail["reasons"])

    def test_latin_dominant_transcript_has_no_formal_weak_overall(self) -> None:
        guardrail = _guardrail_for_text("I want sushi kudasai", moras=["ク", "ダ"], score=88)
        self.assertEqual(guardrail["status"], "no_score")
        self.assertIn("latin_dominant_confirmed_text", guardrail["reasons"])

    def test_kana_mora_extraction_failed_has_no_formal_weak_overall(self) -> None:
        guardrail = _guardrail_for_text("12345", moras=[], score=84)
        self.assertEqual(guardrail["status"], "no_score")
        self.assertIn("kana_or_mora_unavailable", guardrail["reasons"])

    def test_very_short_input_is_unavailable_not_high_overall(self) -> None:
        guardrail = _guardrail_for_text("はい", moras=["ハ", "イ"], score=97)
        self.assertEqual(guardrail["status"], "no_score")
        self.assertIsNone(guardrail["weak_overall_practice_score_after_guardrail"])
        self.assertIn("short_utterance_insufficient_evidence", guardrail["reasons"])

    def test_asr_confirmed_japanese_sentence_can_receive_practice_score(self) -> None:
        guardrail = _guardrail_for_text(
            "今日はいい天気です",
            moras=["キョ", "ウ", "ハ", "イ", "イ", "テ", "ン", "キ", "デ", "ス"],
            score=91,
        )
        self.assertEqual(guardrail["status"], "ok")
        self.assertEqual(guardrail["weak_overall_practice_score_after_guardrail"], 91)

    def test_low_f0_coverage_caps_weak_overall(self) -> None:
        guardrail = _guardrail_for_text(
            "今日はいい天気です",
            moras=["キョ", "ウ", "ハ", "イ", "イ", "テ", "ン", "キ", "デ", "ス"],
            score=94,
            weak_prosody_details={"available": False, "f0_coverage": 0.2, "valid_f0_mora_count": 2, "voiced_mora_count": 2},
        )
        self.assertEqual(guardrail["status"], "capped")
        self.assertLessEqual(guardrail["weak_overall_practice_score_after_guardrail"], 70)
        self.assertIn("low_f0_coverage_practice_cap", guardrail["reasons"])

    def test_renderer_hides_weak_overall_and_prosody_when_guardrail_blocks(self) -> None:
        result = _weak_result(
            target_text="please give me ramen",
            kana="",
            moras=[],
            weak_overall_practice_score=None,
            details={
                "weak_reference_native_likeness": {
                    "weak_prosody_naturalness_score": 95,
                    "weak_overall_practice_score": None,
                    "weak_overall_guardrail": {
                        "status": "no_score",
                        "display_allowed": False,
                        "reasons": ["latin_dominant_confirmed_text"],
                    },
                }
            },
        )
        rendered = render_user_facing_result(result, mode="asr_confirmed_weak_reference")
        self.assertIsNone(rendered["display_score"])
        self.assertIsNone(rendered["debug"]["visible_prosody_score"])
        self.assertFalse(rendered["debug"]["prosody_score_visible"])

    def test_verified_fixed_reference_path_not_changed(self) -> None:
        result = _weak_result(
            score_type="strict_reference",
            strict_reference_available=True,
            prosody_score=88,
            weak_overall_practice_score=None,
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
        self.assertEqual(rendered["debug"]["visible_prosody_score"], 88)
        self.assertFalse(rendered["debug"]["weak_reference"])

    def test_tone_score_not_in_core_four(self) -> None:
        html = (ROOT / "debug_ui" / "index.html").read_text(encoding="utf-8")
        score_key_block = html.split("const scoreKeys = [", 1)[1].split("];", 1)[0]
        self.assertNotIn('"tone_score"', score_key_block)

    def test_jvs_native_stays_high_and_flat_random_stay_lower(self) -> None:
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
        self.assertLess(flat, native)
        self.assertLess(random, native)


if __name__ == "__main__":
    unittest.main()
