from __future__ import annotations

import unittest
from pathlib import Path

from jp_speech_eval.feedback_renderer import render_user_facing_result
from jp_speech_eval.scoring import score_prosody


ROOT = Path(__file__).resolve().parents[1]


def _prosody_case(f0_by_mora: list[float]):
    moras = ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ"]
    target_pattern = ["L", "H", "H", "H", "L", "L", "L"]
    reference = [100.0, 130.0, 150.0, 145.0, 110.0, 105.0, 100.0]
    accent_phrases = [{"moras": moras, "accent_position": 4}]
    score, feedback, details = score_prosody(
        moras=moras,
        target_pattern=target_pattern,
        f0_by_mora=f0_by_mora,
        reference_f0_by_mora=reference,
        pitch_target_source="manual",
        accent_phrases=accent_phrases,
    )
    return score, feedback, details


def _raw_user_facing_result(**overrides):
    result = {
        "target_text": "ラーメンをください",
        "kana": "ラーメンヲクダサイ",
        "moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"],
        "mora_table": [],
        "total_score": 88,
        "pronunciation_score": 90,
        "prosody_score": 80,
        "fluency_score": 92,
        "tone_score": 5,
        "feedback": ["整体音高和示范音比较接近。"],
        "alignment_mode": "cached_dtw",
        "details": {
            "mode": "reference_based",
            "verified_level": "human_checked",
            "pitch_target_source": "human_checked",
            "reliability": {"level": "high", "overall": 0.95, "alignment": 0.9, "f0_coverage": 0.9},
            "recording_quality": {"score": 0.95},
            "content_match": {"status": "pass"},
            "alignment": {"mode": "cached_dtw"},
            "prosody": {"contour_corr": 0.8, "transition_agreement": 0.8, "final_intonation_score": 85},
            "fluency": {"rhythm_timing_score": 90, "delivery_fluency_score": 92},
            "mora_evidence": [
                {"judgement_available": True, "boundary_confidence": 0.9, "energy_coverage": 0.9}
                for _ in range(9)
            ],
        },
    }
    for key, value in overrides.items():
        if key == "details":
            result["details"].update(value)
        else:
            result[key] = value
    return result


class ProsodyCounterfactualTests(unittest.TestCase):
    def test_native_like_contour_ranks_above_flat_random_and_wrong_drop(self) -> None:
        native_score, _native_feedback, native_details = _prosody_case(
            [100.0, 130.0, 150.0, 145.0, 110.0, 105.0, 100.0]
        )
        flat_score, _flat_feedback, flat_details = _prosody_case([120.0] * 7)
        random_score, _random_feedback, random_details = _prosody_case(
            [100.0, 150.0, 105.0, 145.0, 130.0, 90.0, 160.0]
        )
        wrong_drop_score, _wrong_feedback, wrong_drop_details = _prosody_case(
            [100.0, 130.0, 150.0, 145.0, 160.0, 165.0, 170.0]
        )

        self.assertGreater(native_score, flat_score)
        self.assertGreater(native_score, random_score)
        self.assertGreater(native_score, wrong_drop_score)
        self.assertLess(flat_details["transition_agreement"], native_details["transition_agreement"])
        self.assertLess(random_details["contour_corr"], native_details["contour_corr"])
        self.assertEqual(native_details["accent_drop_agreement"], 1.0)
        self.assertEqual(wrong_drop_details["accent_drop_agreement"], 0.0)

    def test_low_f0_coverage_returns_insufficient_evidence_details(self) -> None:
        score, feedback, details = _prosody_case(
            [100.0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 100.0]
        )
        self.assertEqual(score, 50)
        self.assertEqual(details["note"], "insufficient_valid_mora_f0")
        self.assertLess(details["valid_mora_count"], 4)
        self.assertTrue(any("音高信息不够清楚" in item for item in feedback))

    def test_low_f0_coverage_hides_visible_prosody_but_keeps_raw_debug(self) -> None:
        rendered = render_user_facing_result(_raw_user_facing_result(
            prosody_score=98,
            details={"reliability": {"level": "high", "overall": 0.82, "alignment": 0.9, "f0_coverage": 0.2}},
        ))
        self.assertIsNotNone(rendered["display_score"])
        self.assertEqual(rendered["debug"]["prosody_score"], 98)
        self.assertIsNone(rendered["debug"]["visible_prosody_score"])
        self.assertFalse(rendered["debug"]["prosody_score_visible"])
        self.assertIn("low_f0_coverage", rendered["suppressed_reasons"])

    def test_content_mismatch_hides_formal_score(self) -> None:
        rendered = render_user_facing_result(_raw_user_facing_result(details={
            "content_match": {
                "status": "pass",
                "method": "asr_kana_match+mfcc_dtw_reference_gate",
                "asr_provider": "whisper",
                "transcript": "please give me ramen",
                "transcript_kana": "プリーズギブミーラーメン",
                "target_kana": "ラーメンヲクダサイ",
                "kana_similarity": 0.2,
            }
        }))
        self.assertEqual(rendered["status"], "retry")
        self.assertIsNone(rendered["display_score"])
        self.assertIsNone(rendered["practice_score"]["value"])
        self.assertIn("content_mismatch_veto", rendered["suppressed_reasons"])

    def test_debug_ui_core_dimensions_do_not_use_tone_score(self) -> None:
        html = (ROOT / "debug_ui" / "index.html").read_text(encoding="utf-8")
        score_key_block = html.split("const scoreKeys = [", 1)[1].split("];", 1)[0]
        label_block = html.split('scoreLabels: {', 1)[1].split('},', 1)[0]
        self.assertIn('"rhythm_score"', score_key_block)
        self.assertIn('"pitch_score"', score_key_block)
        self.assertNotIn('"tone_score"', score_key_block)
        self.assertIn('rhythm_score: "韵律（特殊拍）"', label_block)
        self.assertIn('pitch_score: "音调（F0高低）"', label_block)
        self.assertNotIn("expression_proxy_score", label_block)


if __name__ == "__main__":
    unittest.main()
