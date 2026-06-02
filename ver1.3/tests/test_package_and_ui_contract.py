from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import jp_speech_eval
from jp_speech_eval import EvaluationRequest, SpeechEvalConfig, SpeechEvaluationClient


ROOT = Path(__file__).resolve().parents[1]


def _raw_result() -> dict:
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
            "mode": "reference_based",
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


class PackageAndUiContractTest(unittest.TestCase):
    def test_public_package_exports_client_api(self) -> None:
        for name in (
            "EvaluationRequest",
            "EvaluationResponse",
            "SpeechEvalConfig",
            "SpeechEvaluationClient",
            "build_asr_confirmation",
            "evaluate_speech",
        ):
            self.assertTrue(hasattr(jp_speech_eval, name), name)

    def test_client_api_returns_user_facing_contract(self) -> None:
        client = SpeechEvaluationClient(SpeechEvalConfig(cache_path="cache/ramen_kudasai"))
        with patch("jp_speech_eval.api.evaluate_mode", return_value=_raw_result()):
            response = client.evaluate(EvaluationRequest(audio_path="user.wav", mode="reference"))
        self.assertTrue(response.ok)
        self.assertIn("user_facing", response.to_dict())
        self.assertIn("practice_score", response.user_facing)
        self.assertFalse(response.user_facing["display_total_score"])

    def test_demo_ui_default_modes_hide_diagnostic_tools(self) -> None:
        import scripts.debug_ui as debug_ui

        self.assertEqual(debug_ui.CORE_MODES, ["reference", "asr_pseudo_reference", "kanade_asr_voice_reference"])
        self.assertEqual(debug_ui.PUBLIC_DEMO_MODES, ["reference", "asr_pseudo_reference", "kanade_asr_voice_reference"])
        self.assertNotIn("transcript_assisted_light", debug_ui.CORE_MODES)
        self.assertNotIn("acoustic", debug_ui.PUBLIC_DEMO_MODES)

    def test_package_api_docs_and_example_exist(self) -> None:
        doc = ROOT / "docs" / "python_package_api.md"
        example = ROOT / "examples" / "package_api_quickstart.py"
        self.assertTrue(doc.exists())
        self.assertTrue(example.exists())
        text = doc.read_text(encoding="utf-8")
        self.assertIn("pip install -e", text)
        self.assertIn("SpeechEvaluationClient", text)
        self.assertIn("user_facing", text)

    def test_hosted_demo_fast_start_contract(self) -> None:
        repo_root = ROOT.parent
        start = (repo_root / "deploy" / "start_full_demo.sh").read_text(encoding="utf-8")
        publish = (repo_root / "deploy" / "publish_hf_space.sh").read_text(encoding="utf-8")
        self.assertIn('ENABLE_AIVIS="${ENABLE_AIVIS:-0}"', start)
        self.assertIn('PREWARM_REFERENCES="${PREWARM_REFERENCES:-1}"', start)
        self.assertNotIn("deadline = time.time() + 900", start)
        self.assertIn("--exclude 'JANON/'", publish)
        self.assertIn("--exclude 'JVS/'", publish)

    def test_kanade_async_status_endpoints_are_registered(self) -> None:
        ui_source = (ROOT / "scripts" / "debug_ui.py").read_text(encoding="utf-8")
        self.assertIn("/api/kanade/status", ui_source)
        self.assertIn("/api/kanade/reference.wav", ui_source)
        self.assertIn("ThreadPoolExecutor", ui_source)


if __name__ == "__main__":
    unittest.main()
