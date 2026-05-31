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


if __name__ == "__main__":
    unittest.main()
