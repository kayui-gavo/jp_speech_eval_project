from __future__ import annotations

import numpy as np
import soundfile as sf

from jp_speech_eval.reliability_counterfactual import (
    reliability_cap_triggers,
    rescore_without_reliability_caps,
)


def _result_fixture() -> dict:
    return {
        "target_text": "あいうえ",
        "kana": "アイウエ",
        "moras": ["ア", "イ", "ウ", "エ"],
        "target_pitch": ["L", "H", "H", "L"],
        "duration_sec": 1.0,
        "alignment_mode": "cached_dtw_fallback_equal",
        "pronunciation_score": 60,
        "prosody_score": 55,
        "fluency_score": 96,
        "tone_score": 85,
        "total_score": 72,
        "pause_info": {
            "pause_count": 0,
            "pause_total": 0.0,
            "pause_ratio": 0.0,
            "pause_segments": [],
            "analysis_duration_sec": 1.0,
        },
        "mora_table": [
            {"start_sec": 0.00, "end_sec": 0.25, "f0_hz": 180.0},
            {"start_sec": 0.25, "end_sec": 0.50, "f0_hz": 220.0},
            {"start_sec": 0.50, "end_sec": 0.75, "f0_hz": 215.0},
            {"start_sec": 0.75, "end_sec": 1.00, "f0_hz": 175.0},
        ],
        "details": {
            "reliability": {
                "overall": 0.60,
                "f0_coverage": 0.40,
            },
            "mora_evidence_summary": {
                "judgement_available_count": 1,
            },
            "prosody": {
                "pitch_target_source": "test",
            },
            "reference_f0_by_mora": [180.0, 220.0, 215.0, 175.0],
            "accent_phrases": [],
            "aggregate": {
                "weights": {
                    "pronunciation": 0.35,
                    "prosody": 0.40,
                    "fluency": 0.25,
                    "tone": 0.0,
                }
            },
        },
    }


def test_cap_trigger_audit_mirrors_legacy_predicates() -> None:
    triggers = reliability_cap_triggers(_result_fixture())
    assert triggers["alignment_equal_fallback"] is True
    assert triggers["mora_evidence_below_threshold"] is True
    assert triggers["f0_coverage_below_0_50"] is True
    assert triggers["overall_reliability_below_0_75"] is True


def test_counterfactual_replays_raw_scorers_without_changing_product(tmp_path) -> None:
    sr = 16000
    wav = tmp_path / "utterance.wav"
    t = np.arange(sr, dtype=float) / sr
    y = 0.15 * np.sin(2.0 * np.pi * 200.0 * t)
    sf.write(wav, y, sr, subtype="FLOAT")

    report = rescore_without_reliability_caps(_result_fixture(), wav_path=wav)
    assert report["available"] is True
    assert report["product_behavior_changed"] is False
    assert report["observed_legacy_product_scores"]["pronunciation"] == 60
    assert report["counterfactual_without_reliability_caps"]["pronunciation"] > 60
    assert report["counterfactual_minus_observed"]["pronunciation"] > 0
    assert report["counterfactual_without_reliability_caps"]["total"] >= report["observed_legacy_product_scores"]["total"]
