from __future__ import annotations

import unittest

from scripts.audit_feedback_actionability import _fixture
from jp_speech_eval.feedback_renderer import render_user_facing_result


def _render(case_id: str):
    return render_user_facing_result(
        _fixture(case_id),
        special_mora_threshold_profile="v2_limited_candidate",
    )


class FeedbackActionabilityTests(unittest.TestCase):
    def test_no_score_does_not_emit_pronunciation_or_pitch_detail(self) -> None:
        rendered = _render("random_english_latin")
        self.assertIsNone(rendered["display_score"])
        self.assertEqual([item["dimension"] for item in rendered["feedback_candidates"]], ["content_match"])
        text = " ".join(rendered["user_messages"])
        self.assertNotIn("音高变化偏平", text)
        self.assertNotIn("第 ", text)

    def test_too_short_only_requests_a_complete_sentence(self) -> None:
        rendered = _render("too_short_japanese")
        focus = rendered["feedback_candidates"][0]
        self.assertEqual(focus["evidence_type"], "too_short")
        self.assertIn("完整短句", focus["practice_tip"])
        self.assertEqual(len(rendered["feedback_candidates"]), 1)

    def test_random_english_requests_confirmed_japanese(self) -> None:
        focus = _render("random_english_latin")["feedback_candidates"][0]
        self.assertEqual(focus["evidence_type"], "content_mismatch")
        self.assertIn("日语", focus["practice_tip"])

    def test_flat_pitch_is_naturalness_feedback_not_accent_error(self) -> None:
        focus = _render("flat_pitch")["feedback_candidates"][0]
        text = f"{focus['user_message']} {focus['practice_tip']} {focus['caveat']}"
        self.assertEqual(focus["evidence_type"], "flat_pitch")
        self.assertIn("仅供参考", text)
        self.assertNotIn("高低重音错", text)
        self.assertNotIn("重音核", text)

    def test_low_f0_reports_insufficient_pitch_evidence(self) -> None:
        focus = _render("low_f0_coverage")["feedback_candidates"][0]
        self.assertEqual(focus["evidence_type"], "low_f0_coverage")
        self.assertIn("音高信息太少", focus["user_message"])
        self.assertIn("不代表音高有问题", focus["caveat"])

    def test_special_mora_has_real_location_and_practice_tip(self) -> None:
        for case_id in ("special_mora_long_vowel_short", "special_mora_nasal_short"):
            focus = _render(case_id)["feedback_candidates"][0]
            self.assertEqual(focus["evidence_type"], "special_mora_duration_issue")
            self.assertIsNotNone(focus["location"])
            self.assertGreaterEqual(focus["location"]["mora_index"], 1)
            self.assertTrue(focus["location"]["mora_text"])
            self.assertTrue(focus["practice_tip"])

    def test_alignment_fallback_has_no_special_mora_or_pitch_detail(self) -> None:
        rendered = _render("alignment_fallback")
        self.assertIsNotNone(rendered["display_score"])
        self.assertIsNotNone(rendered["debug"]["visible_prosody_score"])
        self.assertEqual([item["evidence_type"] for item in rendered["feedback_candidates"]], ["alignment_uncertain"])
        self.assertIsNone(rendered["feedback_candidates"][0]["location"])

    def test_visible_feedback_is_limited_to_two_messages(self) -> None:
        for case_id in (
            "normal_japanese_native", "flat_pitch", "long_pause_many", "fast_rate",
            "special_mora_long_vowel_short",
        ):
            self.assertLessEqual(len(_render(case_id)["user_messages"]), 2, case_id)

    def test_every_selected_candidate_has_a_practice_tip(self) -> None:
        for case_id in (
            "normal_japanese_native", "learner_japanese_ok", "too_short_japanese",
            "random_english_latin", "flat_pitch", "random_pitch", "low_f0_coverage",
            "long_pause_many", "fast_rate", "special_mora_long_vowel_short",
            "special_mora_nasal_short", "alignment_fallback",
        ):
            candidates = _render(case_id)["feedback_candidates"]
            self.assertTrue(candidates, case_id)
            self.assertTrue(all(item["practice_tip"] for item in candidates), case_id)

    def test_weak_pitch_always_has_reference_limited_caveat(self) -> None:
        for case_id in ("flat_pitch", "random_pitch"):
            focus = _render(case_id)["feedback_candidates"][0]
            self.assertIn("不是严格高低重音判定", focus["caveat"])

    def test_verified_fixed_reference_remains_scored(self) -> None:
        rendered = _render("special_mora_long_vowel_short")
        self.assertIsNotNone(rendered["display_score"])
        self.assertFalse(rendered["debug"]["weak_reference"])
        self.assertEqual(rendered["feedback_candidates"][0]["dimension"], "rhythm_special_mora")

    def test_tone_score_does_not_enter_feedback_candidates(self) -> None:
        result = _fixture("normal_japanese_native")
        result["tone_score"] = 0
        low = render_user_facing_result(result, special_mora_threshold_profile="v2_limited_candidate")
        result["tone_score"] = 100
        high = render_user_facing_result(result, special_mora_threshold_profile="v2_limited_candidate")
        self.assertEqual(low["feedback_candidates"], high["feedback_candidates"])


if __name__ == "__main__":
    unittest.main()
