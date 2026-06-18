from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

from jp_speech_eval.feedback_renderer import render_user_facing_result
from jp_speech_eval.scoring import score_prosody

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_prosody_reference_targets import (  # noqa: E402
    flat_f0,
    pattern_from_f0,
    shifted_f0,
    shuffled_f0,
    smoothed_f0,
)


MORAS = ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"]
NATIVE_LIKE_F0 = [100.0, 126.0, 148.0, 142.0, 108.0, 103.0, 99.0, 96.0, 92.0]
ACCENT_PHRASES = [{"moras": MORAS, "accent_position": 4}]


def _score(
    user_f0: list[float],
    *,
    reference_f0: list[float] | None,
    target_pattern: list[str] | None = None,
    pitch_target_source: str = "manual",
) -> tuple[int, dict]:
    score, _feedback, details = score_prosody(
        moras=MORAS,
        target_pattern=target_pattern or pattern_from_f0(reference_f0 or user_f0),
        f0_by_mora=user_f0,
        reference_f0_by_mora=reference_f0,
        pitch_target_source=pitch_target_source,
        accent_phrases=ACCENT_PHRASES,
    )
    return score, details


def _raw_user_facing_result(**overrides):
    result = {
        "target_text": "ラーメンをください",
        "kana": "ラーメンヲクダサイ",
        "moras": MORAS,
        "mora_table": [],
        "total_score": 91,
        "pronunciation_score": 88,
        "prosody_score": 93,
        "fluency_score": 95,
        "tone_score": 0,
        "feedback": [],
        "alignment_mode": "cached_dtw",
        "details": {
            "mode": "reference_based",
            "verified_level": "human_checked",
            "pitch_target_source": "human_checked",
            "reliability": {"level": "high", "overall": 0.95, "alignment": 0.9, "f0_coverage": 0.9},
            "content_match": {"status": "pass"},
            "alignment": {"mode": "cached_dtw"},
            "prosody": {"contour_corr": 0.9, "transition_agreement": 0.9},
            "fluency": {"rhythm_timing_score": 94, "delivery_fluency_score": 95},
            "mora_evidence": [{"judgement_available": True} for _ in MORAS],
        },
    }
    for key, value in overrides.items():
        if key == "details":
            result["details"].update(value)
        else:
            result[key] = value
    return result


class ProsodyReferenceTargetAuditTests(unittest.TestCase):
    def test_self_oracle_ranks_above_flat_shuffled_and_shifted_contours(self) -> None:
        self_score, self_details = _score(NATIVE_LIKE_F0, reference_f0=NATIVE_LIKE_F0)
        flat_score, flat_details = _score(flat_f0(NATIVE_LIKE_F0), reference_f0=NATIVE_LIKE_F0)
        shuffled_score, shuffled_details = _score(
            shuffled_f0(NATIVE_LIKE_F0, seed=42),
            reference_f0=NATIVE_LIKE_F0,
        )
        shifted_score, shifted_details = _score(
            NATIVE_LIKE_F0,
            reference_f0=shifted_f0(NATIVE_LIKE_F0, 1),
            target_pattern=pattern_from_f0(shifted_f0(NATIVE_LIKE_F0, 1)),
        )

        self.assertGreaterEqual(self_score, 95)
        self.assertGreater(self_score, flat_score)
        self.assertGreater(self_score, shuffled_score)
        self.assertGreater(self_score, shifted_score)
        self.assertEqual(self_details["contour_corr"], 1.0)
        self.assertLess(flat_details["transition_agreement"], self_details["transition_agreement"])
        self.assertLess(shuffled_details["contour_corr"], self_details["contour_corr"])
        self.assertLess(shifted_details["contour_corr"], self_details["contour_corr"])

    def test_smoothed_self_target_stays_close_to_self_oracle(self) -> None:
        self_score, _self_details = _score(NATIVE_LIKE_F0, reference_f0=NATIVE_LIKE_F0)
        smooth = smoothed_f0(NATIVE_LIKE_F0)
        smooth_score, smooth_details = _score(
            NATIVE_LIKE_F0,
            reference_f0=smooth,
            target_pattern=pattern_from_f0(smooth),
        )
        self.assertGreaterEqual(smooth_score, self_score - 12)
        self.assertGreaterEqual(smooth_details["contour_corr"], 0.80)

    def test_current_openjtalk_like_mismatched_target_can_score_below_self_oracle(self) -> None:
        self_score, _self_details = _score(NATIVE_LIKE_F0, reference_f0=NATIVE_LIKE_F0)
        mismatched_target = ["H" if p == "L" else "L" if p == "H" else "?" for p in pattern_from_f0(NATIVE_LIKE_F0)]
        mismatch_score, mismatch_details = _score(
            NATIVE_LIKE_F0,
            reference_f0=None,
            target_pattern=mismatched_target,
            pitch_target_source="openjtalk_accent_phrase_chain",
        )
        self.assertGreater(self_score, mismatch_score)
        self.assertLess(mismatch_details["hl_match_rate"], 0.5)

    def test_low_f0_coverage_still_unavailable(self) -> None:
        low_f0 = [NATIVE_LIKE_F0[0], math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, NATIVE_LIKE_F0[-1]]
        score, details = _score(low_f0, reference_f0=NATIVE_LIKE_F0)
        self.assertEqual(score, 50)
        self.assertEqual(details["note"], "insufficient_valid_mora_f0")

    def test_helpers_preserve_missing_f0_positions(self) -> None:
        values = [100.0, math.nan, 120.0, 130.0]
        self.assertTrue(math.isnan(smoothed_f0(values)[1]))
        self.assertTrue(math.isnan(flat_f0(values)[1]))
        self.assertEqual(len(shifted_f0(values, 1)), len(values))
        self.assertEqual(len(shuffled_f0(values, seed=1)), len(values))

    def test_no_user_facing_or_core_dimension_regression(self) -> None:
        rendered = render_user_facing_result(_raw_user_facing_result())
        self.assertIsNotNone(rendered["display_score"])
        self.assertEqual(rendered["debug"]["expression_proxy_score"], 0)

        html = (ROOT / "debug_ui" / "index.html").read_text(encoding="utf-8")
        score_key_block = html.split("const scoreKeys = [", 1)[1].split("];", 1)[0]
        self.assertIn('"rhythm_score"', score_key_block)
        self.assertIn('"pitch_score"', score_key_block)
        self.assertNotIn('"tone_score"', score_key_block)


if __name__ == "__main__":
    unittest.main()
