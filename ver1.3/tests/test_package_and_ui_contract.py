from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

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

    def test_debug_ui_json_sanitizes_nan_values(self) -> None:
        import scripts.debug_ui as debug_ui

        payload = {
            "ok": True,
            "f0": [120.0, float("nan"), float("inf"), -float("inf")],
            "nested": {"value": math.nan},
        }
        text = json.dumps(debug_ui._json_safe(payload), allow_nan=False)
        self.assertNotIn("NaN", text)
        self.assertEqual(json.loads(text)["f0"], [120.0, None, None, None])

    def test_pitch_visualization_uses_frame_trace_and_is_pitch_scale_invariant(self) -> None:
        import scripts.debug_ui as debug_ui

        times = [0.05, 0.15, 0.25, 0.35]
        boundaries = [(0.0, 0.2), (0.2, 0.4)]
        trace_a = debug_ui._normalized_pitch_trace(times, [100.0, 120.0, 0.0, 150.0], boundaries)
        trace_b = debug_ui._normalized_pitch_trace(times, [200.0, 240.0, 0.0, 300.0], boundaries)
        self.assertEqual(len(trace_a), 4)
        self.assertEqual(
            [row["semitone"] for row in trace_a],
            [row["semitone"] for row in trace_b],
        )
        self.assertIsNone(trace_a[2]["semitone"])

    def test_pitch_ui_uses_soft_guide_not_per_mora_red_green_correctness(self) -> None:
        html = (ROOT / "debug_ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn("softPitchGuide", html)
        self.assertIn("framePitchSummary", html)
        self.assertIn("pitch-scroll", html)
        self.assertIn("逐帧 F0", html)
        self.assertNotIn('ctx.fillText(`${t("observedAbbr")}:${obs}`', html)

    def test_pitch_plot_dtw_maps_user_frames_to_reference_mora_axis(self) -> None:
        from jp_speech_eval.alignment import align_user_times_to_reference_mora_axis
        from jp_speech_eval.audio_features import load_audio, trim_silence
        from jp_speech_eval.sentence_cache import load_sentence_cache

        cache = load_sentence_cache(ROOT / "cache" / "ramen_kudasai")
        audio = load_audio(str(ROOT / "data" / "ramen.wav"), sr=cache.meta.sr)
        speech, _ = trim_silence(audio.y, top_db=30.0)
        frame_times = np.arange(0.0, len(speech) / audio.sr, 0.01)
        x_mora, metadata = align_user_times_to_reference_mora_axis(
            cache,
            speech,
            audio.sr,
            frame_times,
        )
        finite = x_mora[np.isfinite(x_mora)]
        self.assertTrue(metadata["available"])
        self.assertEqual(metadata["method"], "mfcc_dtw_reference_time")
        self.assertGreater(float(metadata["mapped_ratio"]), 0.95)
        self.assertGreater(len(finite), 20)
        self.assertTrue(np.all(np.diff(finite) >= 0))

    def test_pitch_ui_labels_generated_reference_as_demonstration(self) -> None:
        html = (ROOT / "debug_ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn("demonstrationContour", html)
        self.assertIn("dtwAlignedLabel", html)
        self.assertIn('drawTrace(referenceTrace, "#0f766e", 3.0', html)
        self.assertIn('drawTrace(userTrace, "#b45309", 3.0', html)

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

    def test_debug_ui_does_not_fallback_to_raw_scores_when_user_facing_hides_them(self) -> None:
        html = (ROOT / "debug_ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn("const rawTotal = userFacing ? userFacing.display_score : result.total_score;", html)
        self.assertIn("const scoreSuppressed = Boolean(userFacing && userFacing.display_score == null);", html)
        self.assertIn("const hideFormalDimensions = scoreSuppressed", html)
        self.assertIn('hiddenReasons.includes("content_mismatch_veto")', html)
        self.assertNotIn('!weakPractice && hiddenReasons.includes("fallback_alignment")', html)
        self.assertNotIn('!weakPractice && hiddenReasons.includes("low_f0_coverage")', html)
        self.assertIn("const debug = userFacing?.debug || {};", html)
        self.assertIn("debug.rhythm_score ?? debug.rhythm_timing_score", html)
        self.assertIn("Object.prototype.hasOwnProperty.call(userDims, name)", html)
        self.assertIn('dimensionValue("pitch", debug.visible_prosody_score)', html)
        self.assertNotIn("debug.visible_prosody_score ?? debug.prosody_score", html)
        self.assertNotIn('result[key] ?? 0', html)


if __name__ == "__main__":
    unittest.main()
