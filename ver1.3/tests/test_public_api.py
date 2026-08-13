from __future__ import annotations

import unittest
from unittest.mock import patch

from jp_speech_eval import (
    EvaluationRequest,
    SpeechEvalConfig,
    SpeechEvaluationClient,
    build_asr_confirmation,
    evaluate_speech,
)


def _raw_result(mode: str = "reference_based") -> dict:
    return {
        "target_text": "ラーメンをください",
        "kana": "ラーメンヲクダサイ",
        "moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"],
        "mora_table": [],
        "total_score": 88,
        "pronunciation_score": 90,
        "prosody_score": 80,
        "fluency_score": 92,
        "tone_score": 70,
        "feedback": ["今回の練習は大きな問題なく確認できました。"],
        "alignment_mode": "cached_dtw",
        "details": {
            "mode": mode,
            "verified_level": "human_checked",
            "pitch_target_source": "human_checked",
            "reliability": {"level": "high", "overall": 0.95, "alignment": 0.9, "f0_coverage": 0.9},
            "recording_quality": {"score": 0.95},
            "content_match": {"status": "pass"},
            "alignment": {"mode": "cached_dtw"},
            "pronunciation": {"mora_duration_cv": 0.1, "special_mora_penalty": 0, "special_mora_diagnostics": []},
            "prosody": {"contour_corr": 0.8, "transition_agreement": 0.8, "final_intonation_score": 85},
            "fluency": {"rhythm_timing_score": 92, "delivery_fluency_score": 94},
            "mora_evidence": [
                {"judgement_available": True, "boundary_confidence": 0.9, "energy_coverage": 0.9}
                for _ in range(9)
            ],
        },
    }


class PublicApiTest(unittest.TestCase):
    def test_client_returns_user_facing_and_raw_result(self) -> None:
        client = SpeechEvaluationClient(SpeechEvalConfig(cache_path="cache/ramen_kudasai"))
        with patch("jp_speech_eval.api.evaluate_mode", return_value=_raw_result()):
            response = client.evaluate(EvaluationRequest(audio_path="user.wav", mode="reference"))
        self.assertTrue(response.ok)
        self.assertIn("display_score", response.user_facing)
        self.assertIn("practice_score", response.user_facing)
        self.assertIn("summary_text", response.user_facing)
        self.assertEqual(response.raw_result["target_text"], "ラーメンをください")
        self.assertIn("debug_total_score", response.user_facing["debug"])
        self.assertNotIn("total_score", response.user_facing["summary_text"])
        self.assertFalse(response.user_facing["display_total_score"])

    def test_one_shot_helper_uses_public_request(self) -> None:
        with patch("jp_speech_eval.api.evaluate_mode", return_value=_raw_result()):
            response = evaluate_speech(EvaluationRequest(audio_path="user.wav", mode="reference"))
        self.assertTrue(response["ok"])
        self.assertIn("user_facing", response)

    def test_asr_confirmation_helper_returns_serializable_dict(self) -> None:
        fake_prompt = type(
            "FakePrompt",
            (),
            {
                "to_dict": lambda self: {
                    "mode": "asr_confirm",
                    "session_id": "abc",
                    "asr_candidates": [{"id": 1, "text": "ラーメンをください", "confidence": 0.9}],
                    "editable_text": "ラーメンをください",
                    "message": "猜你想说的是哪一句？如果不对，请手动修改。",
                    "asr_raw": {},
                }
            },
        )()
        with patch("jp_speech_eval.api.build_asr_confirmation_prompt", return_value=fake_prompt):
            response = build_asr_confirmation("user.wav")
        self.assertTrue(response["ok"])
        self.assertEqual(response["prompt"]["mode"], "asr_confirm")

    def test_c_end_response_can_ignore_raw_total_score(self) -> None:
        raw = _raw_result()
        raw["total_score"] = 5
        raw["pronunciation_score"] = 90
        raw["fluency_score"] = 90
        with patch("jp_speech_eval.api.evaluate_mode", return_value=raw):
            response = evaluate_speech(EvaluationRequest(audio_path="user.wav", mode="reference"))
        self.assertTrue(response["ok"])
        self.assertIn("practice_score", response["user_facing"])
        self.assertGreaterEqual(response["user_facing"]["practice_score"]["value"], 80)

    def test_plausible_japanese_target_mismatch_routes_to_broad_mode(self) -> None:
        fixed = _raw_result()
        fixed["cache_prefix"] = "/tmp/fixed-target"
        fixed["details"]["content_match"] = {
            "status": "fail",
            "transcript": "今日はいい天気です",
        }
        broad = _raw_result("transcript_assisted_light")
        broad["details"]["content_match"] = {"status": "unknown"}
        language = type("Language", (), {"available": True, "language": "ja", "language_probability": 0.9})()
        with patch("jp_speech_eval.api.evaluate_mode", side_effect=[fixed, broad]), patch(
            "jp_speech_eval.api._fallback_language_evidence", return_value=(True, language)
        ):
            response = evaluate_speech(EvaluationRequest(audio_path="user.wav", mode="reference"))
        self.assertTrue(response["ok"])
        self.assertEqual(response["mode"], "reference_mismatch_general_japanese")
        self.assertIsNotNone(response["user_facing"]["practice_score"]["value"])
        match = response["raw_result"]["details"]["content_match"]
        self.assertFalse(match["content_verified"])
        self.assertTrue(match["japanese_content_plausible"])
        self.assertEqual(
            response["raw_result"]["details"]["fixed_reference_debug"]["cache_prefix"],
            "/tmp/fixed-target",
        )
        self.assertIn("目標文とは違う", response["raw_result"]["feedback"][0])
        dims = {item["key"]: item for item in response["user_facing"]["score_dimensions"]}
        self.assertFalse(dims["pitch_accent"]["available"])

    def test_nonsense_target_mismatch_does_not_enter_broad_fallback(self) -> None:
        fixed = _raw_result()
        fixed["details"]["content_match"] = {"status": "fail", "transcript": "ああああああ"}
        with patch("jp_speech_eval.api.evaluate_mode", return_value=fixed) as evaluate:
            response = evaluate_speech(EvaluationRequest(audio_path="user.wav", mode="reference"))
        self.assertTrue(response["ok"])
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(response["mode"], "reference_based")
        self.assertIsNone(response["user_facing"]["practice_score"]["value"])


if __name__ == "__main__":
    unittest.main()
