from __future__ import annotations

import math

import numpy as np

from jp_speech_eval.consumer_dimension_policy import build_consumer_score_components
from jp_speech_eval.free_speech_evidence import (
    build_asr_recoverability_evidence,
    build_free_speech_dimension_evidence,
    build_shadow_candidate_surface,
    build_target_independent_intonation_evidence,
    build_word_timing_rhythm_evidence,
)


def _asr_info() -> dict:
    words = [
        {"text": "今日は", "start_sec": 0.00, "end_sec": 0.42, "probability": 0.96},
        {"text": "友達と", "start_sec": 0.46, "end_sec": 0.88, "probability": 0.91},
        {"text": "駅で", "start_sec": 0.94, "end_sec": 1.20, "probability": 0.88},
        {"text": "ラーメンを", "start_sec": 1.24, "end_sec": 1.78, "probability": 0.93},
        {"text": "食べて", "start_sec": 1.84, "end_sec": 2.18, "probability": 0.86},
        {"text": "帰ります", "start_sec": 2.24, "end_sec": 2.72, "probability": 0.90},
    ]
    return {
        "available": True,
        "provider": "faster-whisper",
        "model": "small",
        "text": "今日は友達と駅でラーメンを食べて帰ります",
        "language": "ja",
        "language_probability": 0.98,
        "words": words,
        "segments": [
            {
                "start_sec": 0.0,
                "end_sec": 2.72,
                "avg_logprob": -0.18,
                "no_speech_prob": 0.01,
                "compression_ratio": 1.18,
            }
        ],
    }


def test_asr_recoverability_is_machine_proxy_not_human_comprehensibility() -> None:
    evidence = build_asr_recoverability_evidence(_asr_info(), speech_duration_sec=2.8)
    assert evidence["available"] is True
    assert evidence["confidence"] == "medium"
    assert 0.0 < evidence["asr_recoverability_index_0to1"] <= 1.0
    assert evidence["word_probability"]["count"] == 6
    assert evidence["language_probability_role"] == "routing_only_not_clarity_score"
    assert evidence["construct"] == "machine_recoverability_proxy_not_human_comprehensibility"
    assert evidence["score_mapped"] is False
    assert evidence["product_calibrated"] is False
    assert evidence["user_facing"] is False


def test_word_timing_rhythm_is_local_tempo_proxy_not_equal_mora_rule() -> None:
    evidence = build_word_timing_rhythm_evidence(_asr_info())
    assert evidence["available"] is True
    assert evidence["usable_word_count"] >= 3
    assert evidence["local_tempo_irregularity_mad_log_sec_per_mora"] is not None
    assert evidence["local_tempo_spread_p90_p10_log_sec_per_mora"] is not None
    assert evidence["construct"] == "asr_word_level_local_tempo_structure_not_mora_isochrony"
    assert evidence["score_mapped"] is False
    assert evidence["user_facing"] is False


def test_intonation_f0_failure_is_unavailable_not_flat_or_bad_score() -> None:
    evidence = build_target_independent_intonation_evidence(
        np.asarray([0.0, 0.01, 0.02]),
        np.asarray([np.nan, 0.0, np.nan]),
    )
    assert evidence["available"] is False
    assert evidence["confidence"] == "unavailable"
    assert evidence["robust_range_semitones_p90_p10"] is None
    assert evidence["score_mapped"] is False
    assert "must not be interpreted as flat" in evidence["caveat"]


def test_intonation_uses_robust_target_independent_movement_features() -> None:
    times = np.linspace(0.0, 2.0, 101)
    f0 = 200.0 * (2.0 ** (2.5 * np.sin(2.0 * np.pi * times / 2.0) / 12.0))
    evidence = build_target_independent_intonation_evidence(times, f0)
    assert evidence["available"] is True
    assert evidence["robust_range_semitones_p90_p10"] > 1.0
    assert evidence["construct"] == "target_independent_f0_movement_not_contextual_intonation_correctness"
    assert evidence["score_mapped"] is False
    assert evidence["product_calibrated"] is False


def test_shadow_candidate_is_shrunk_and_cannot_change_product_score() -> None:
    times = np.linspace(0.0, 2.8, 141)
    f0 = 195.0 * (2.0 ** (3.5 * np.sin(2.0 * np.pi * times / 2.8) / 12.0))
    evidence = build_free_speech_dimension_evidence(
        asr_info=_asr_info(),
        speech_duration_sec=2.8,
        f0_times=times,
        f0_hz=f0,
        spontaneous_fluency={
            "schema_version": "spontaneous_fluency_evidence_v1",
            "speed": {"speech_rate_mora_per_sec": 5.0},
            "breakdown": {"long_silent_pause_count": 1},
            "repair": {"filler_candidate_count": 0},
        },
    )
    candidate = build_shadow_candidate_surface(evidence, current_fluency_score=92)
    assert candidate["user_facing"] is False
    assert candidate["product_calibrated"] is False
    assert candidate["product_score_changed"] is False
    assert candidate["mapping_status"] == "heuristic_shadow_only_not_product_calibrated"
    assert candidate["available_evidence"]["clarity"] is True
    assert candidate["available_evidence"]["mora_timing"] is True
    assert candidate["available_evidence"]["intonation"] is True
    # Strong shrinkage prevents a decoder/acoustic proxy from behaving like a
    # fully calibrated 0-100 judgement before human validation.
    assert 50.0 <= candidate["component_candidates"]["clarity"] <= 85.0
    assert 50.0 <= candidate["component_candidates"]["mora_timing"] <= 85.0
    assert 50.0 <= candidate["component_candidates"]["intonation"] <= 85.0


def test_missing_shadow_evidence_returns_neutral_candidates_not_fake_failure() -> None:
    candidate = build_shadow_candidate_surface(
        {"clarity": {}, "rhythm": {}, "intonation": {}},
        current_fluency_score=None,
    )
    assert candidate["component_candidates"] == {
        "clarity": 70.0,
        "mora_timing": 70.0,
        "delivery_fluency": 70.0,
        "intonation": 70.0,
    }
    assert candidate["available_evidence"] == {
        "clarity": False,
        "mora_timing": False,
        "delivery_fluency": False,
        "intonation": False,
    }


def test_current_general_japanese_product_dimensions_remain_neutral_priors() -> None:
    # The new v4 evidence is intentionally not consumed by ProductScore yet.
    # This regression guard prevents an accidental shadow-to-product promotion.
    result = {
        "pronunciation_score": 84,
        "prosody_score": 90,
        "fluency_score": 62,
        "tone_score": 95,
        "alignment_mode": "none",
        "details": {
            "mode": "transcript_assisted_light",
            "reliability": {"endpointing": 1.0, "f0_coverage": 0.9},
            "fluency": {},
            "shadow": {
                "free_speech_dimension_evidence": {
                    "clarity": {"asr_recoverability_index_0to1": 0.98},
                    "rhythm": {"local_tempo_irregularity_mad_log_sec_per_mora": 0.03},
                    "intonation": {"robust_range_semitones_p90_p10": 6.0},
                },
                "free_speech_candidate_surface": {
                    "component_candidates": {
                        "clarity": 80.0,
                        "mora_timing": 78.0,
                        "delivery_fluency": 62.0,
                        "intonation": 77.0,
                    },
                    "user_facing": False,
                    "product_score_changed": False,
                },
            },
        },
    }
    components = {
        item["key"]: item
        for item in build_consumer_score_components(result, mode="transcript_assisted_light")
    }
    assert components["clarity"]["value"] == 70
    assert components["clarity"]["evidence_state"] == "neutral_prior"
    assert components["mora_timing"]["value"] == 70
    assert components["mora_timing"]["evidence_state"] == "neutral_prior"
    assert components["intonation"]["value"] == 70
    assert components["intonation"]["evidence_state"] == "neutral_prior"
    # Fluency is the only existing free-speech dimension allowed to reuse the
    # legacy top-level value when rate/pause sub-scores are absent.
    assert components["delivery_fluency"]["value"] == 62
