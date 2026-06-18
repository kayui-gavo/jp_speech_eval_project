from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jp_speech_eval.evaluator import evaluate_utterance
from jp_speech_eval.feedback_renderer import render_user_facing_result
from jp_speech_eval.prosody_reference_cache import (
    PROSODY_REFERENCE_CACHE_VERSION,
    build_prosody_reference_cache_payload,
    prosody_reference_cache_path,
    select_prosody_reference_target,
    write_prosody_reference_cache,
)
from jp_speech_eval.sentence_cache import SentenceCache, SentenceMeta


ROOT = Path(__file__).resolve().parents[1]
MORAS = ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"]
F0 = [100.0, 125.0, 145.0, 140.0, 112.0, 108.0, 104.0, 101.0, 98.0]


def _fake_cache(
    tmp: Path,
    *,
    reference_source: str = "jvs_native_reference",
    ref_boundary_method: str = "lab_phone_mora",
    f0_values: list[float] | None = None,
    reference_provider: str | None = None,
) -> SentenceCache:
    f0_values = f0_values or F0
    boundaries = [(i * 0.12, (i + 1) * 0.12) for i in range(len(MORAS))]
    f0_times = np.array([(s + e) / 2.0 for s, e in boundaries], dtype=float)
    meta = SentenceMeta(
        text="ラーメンをください",
        kana="ラーメンヲクダサイ",
        moras=MORAS,
        target_pitch=["H", "L", "L", "L", "L", "L", "H", "H", "L"],
        pitch_target_source="openjtalk_accent_phrase_chain",
        is_question=False,
        sr=16000,
        ref_duration_sec=1.08,
        ref_mora_boundaries=boundaries,
        frontend_raw=[],
        accent_phrases=[{"moras": MORAS, "accent_position": 1}],
        reference_text="ラーメンをください",
        reference_source=reference_source,
        ref_boundary_method=ref_boundary_method,
        reference_id="ramen_test",
        reference_provider=reference_provider,
    )
    return SentenceCache(
        prefix=tmp / "ramen_test",
        meta=meta,
        ref_y=np.zeros(16000, dtype=float),
        ref_mfcc=np.zeros((13, 4), dtype=np.float32),
        ref_f0_times=f0_times,
        ref_f0=np.asarray(f0_values, dtype=float),
    )


def _sidecar_payload(values: list[float]) -> dict:
    return {
        "cache_version": PROSODY_REFERENCE_CACHE_VERSION,
        "created_at": "2026-06-18T00:00:00+00:00",
        "target_id": "ramen_kudasai",
        "target_text": "ラーメンをください",
        "target_kana": "ラーメンヲクダサイ",
        "target_mora_sequence": MORAS,
        "reference_audio_path": "verified-native.wav",
        "reference_f0_mora_values": values,
        "reference_f0_smoothed_values": values,
        "voiced_mora_mask": [math.isfinite(v) and v > 0 for v in values],
        "mora_timing_source": "lab_phone_mora",
        "pitch_target_source": "reference_audio_f0_cache",
        "pitch_target_reliability": "reliable",
        "f0_coverage": 1.0,
        "quality_flags": [],
        "verified_reference": True,
        "reliable": True,
    }


def _result(**details_overrides):
    details = {
        "mode": "reference_based",
        "pitch_target_source": "reference_audio_f0_cache",
        "pitch_target_reliability": "reliable",
        "reliability": {"level": "high", "overall": 0.95, "alignment": 0.9, "f0_coverage": 0.9},
        "recording_quality": {"score": 0.95},
        "content_match": {"status": "pass"},
        "alignment": {"mode": "cached_dtw"},
        "prosody": {
            "contour_corr": 0.9,
            "transition_agreement": 0.9,
            "pitch_target_source": "reference_audio_f0_cache",
            "pitch_target_reliability": "reliable",
        },
        "fluency": {"rhythm_timing_score": 92, "delivery_fluency_score": 93},
        "mora_evidence": [{"judgement_available": True} for _ in MORAS],
    }
    details.update(details_overrides)
    return {
        "target_text": "ラーメンをください",
        "kana": "ラーメンヲクダサイ",
        "moras": MORAS,
        "mora_table": [],
        "total_score": 90,
        "pronunciation_score": 88,
        "prosody_score": 91,
        "fluency_score": 93,
        "tone_score": 0,
        "feedback": [],
        "alignment_mode": details.get("alignment", {}).get("mode", "cached_dtw"),
        "details": details,
    }


class ProsodyReferenceCacheTests(unittest.TestCase):
    def test_verified_reference_cache_payload_is_reliable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            cache = _fake_cache(Path(tmp_name))
            payload = build_prosody_reference_cache_payload(cache, verified_reference=True)
        self.assertTrue(payload["reliable"])
        self.assertEqual(payload["pitch_target_source"], "reference_audio_f0_cache")
        self.assertEqual(payload["pitch_target_reliability"], "reliable")
        self.assertEqual(len(payload["reference_f0_mora_values"]), len(MORAS))

    def test_low_reference_f0_coverage_marks_cache_unreliable(self) -> None:
        sparse_f0 = [100.0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, 98.0]
        with tempfile.TemporaryDirectory() as tmp_name:
            cache = _fake_cache(Path(tmp_name), f0_values=sparse_f0)
            payload = build_prosody_reference_cache_payload(cache, verified_reference=True)
        self.assertFalse(payload["reliable"])
        self.assertIn("low_reference_f0_coverage", payload["quality_flags"])

    def test_sidecar_cache_wins_over_runtime_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            cache = _fake_cache(tmp, reference_source="pyopenjtalk_tts_pseudo_reference", reference_provider="pyopenjtalk")
            write_prosody_reference_cache(cache, verified_reference=True)
            target = select_prosody_reference_target(
                cache,
                fallback_reference_f0=F0,
                text_pitch_target_source="openjtalk_accent_phrase_chain",
            )
        self.assertEqual(target.pitch_target_source, "reference_audio_f0_cache")
        self.assertEqual(target.pitch_target_reliability, "reliable")
        self.assertTrue(target.cache_used)

    def test_missing_sidecar_with_trusted_reference_uses_runtime_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            cache = _fake_cache(Path(tmp_name), reference_source="jvs_native_reference")
            target = select_prosody_reference_target(
                cache,
                fallback_reference_f0=F0,
                text_pitch_target_source="openjtalk_accent_phrase_chain",
            )
        self.assertEqual(target.pitch_target_source, "reference_audio_f0_runtime")
        self.assertEqual(target.pitch_target_reliability, "reliable")
        self.assertTrue(target.runtime_used)

    def test_tts_reference_without_verified_cache_is_weak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            cache = _fake_cache(
                Path(tmp_name),
                reference_source="pyopenjtalk_tts_pseudo_reference",
                reference_provider="pyopenjtalk",
            )
            target = select_prosody_reference_target(
                cache,
                fallback_reference_f0=F0,
                text_pitch_target_source="openjtalk_accent_phrase_chain",
            )
        self.assertEqual(target.pitch_target_source, "tts_reference_weak")
        self.assertEqual(target.pitch_target_reliability, "weak")
        self.assertFalse(target.cache_used)

    def test_reliable_reference_cache_allows_visible_pitch_score(self) -> None:
        rendered = render_user_facing_result(_result())
        self.assertTrue(rendered["debug"]["prosody_score_visible"])
        self.assertEqual(rendered["debug"]["visible_prosody_score"], 91)

    def test_openjtalk_heuristic_target_blocks_strong_pitch_feedback(self) -> None:
        rendered = render_user_facing_result(_result(
            pitch_target_source="openjtalk_accent_phrase_chain",
            pitch_target_reliability="heuristic",
            prosody={
                "pitch_target_source": "openjtalk_accent_phrase_chain",
                "pitch_target_reliability": "heuristic",
            },
        ))
        self.assertFalse(rendered["debug"]["prosody_score_visible"])
        self.assertIsNone(rendered["debug"]["visible_prosody_score"])

    def test_content_mismatch_still_hides_formal_score_with_reliable_reference(self) -> None:
        rendered = render_user_facing_result(_result(content_match={"status": "fail"}))
        self.assertIsNone(rendered["display_score"])
        self.assertIn("content_mismatch", rendered["suppressed_reasons"])
        self.assertIsNone(rendered["debug"]["visible_prosody_score"])

    def test_fallback_alignment_still_hides_pitch_with_reliable_reference(self) -> None:
        rendered = render_user_facing_result(_result(
            alignment={"mode": "cached_dtw_fallback_equal"},
        ))
        self.assertIsNone(rendered["display_score"])
        self.assertIn("fallback_alignment", rendered["suppressed_reasons"])
        self.assertIsNone(rendered["debug"]["visible_prosody_score"])

    def test_evaluator_prefers_reference_audio_sidecar_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            prefix = tmp / "ramen_kudasai"
            shutil.copy(ROOT / "cache" / "ramen_kudasai.json", prefix.with_suffix(".json"))
            shutil.copy(ROOT / "cache" / "ramen_kudasai.npz", prefix.with_suffix(".npz"))
            prosody_reference_cache_path(prefix).write_text(
                json.dumps(_sidecar_payload(F0), ensure_ascii=False),
                encoding="utf-8",
            )
            result = evaluate_utterance(
                wav_path=ROOT / "data" / "ramen.wav",
                alignment_mode="cached_dtw",
                cache_path=prefix,
                use_content_match=False,
            ).to_dict()
        self.assertEqual(result["details"]["pitch_target_source"], "reference_audio_f0_cache")
        self.assertEqual(result["details"]["pitch_target_reliability"], "reliable")
        self.assertEqual(result["prosody_metrics"]["pitch_target_source"], "reference_audio_f0_cache")
        self.assertEqual(result["details"]["prosody"]["pitch_target_source"], "reference_audio_f0_cache")


if __name__ == "__main__":
    unittest.main()
