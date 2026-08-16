from __future__ import annotations

from jp_speech_eval.feedback_renderer import render_user_facing_result
from jp_speech_eval.score_evidence_quality import build_score_evidence_quality


def test_clean_recording_can_still_have_low_score_evidence() -> None:
    result = {
        "details": {
            "recording_quality": {"score": 0.96},
            "reliability": {"recording_quality": 0.96, "endpointing": 0.94},
        }
    }
    dimensions = [
        {"key": "delivery_fluency", "evidence_state": "broad_proxy", "confidence": "medium"},
        {"key": "clarity", "evidence_state": "neutral_prior", "confidence": "low"},
        {"key": "mora_timing", "evidence_state": "neutral_prior", "confidence": "low"},
        {"key": "intonation", "evidence_state": "neutral_prior", "confidence": "low"},
    ]

    summary = build_score_evidence_quality(result, dimensions)

    assert summary["recording_analyzability"]["level"] == "high"
    assert summary["score_evidence"]["level"] == "low"
    assert summary["score_evidence"]["neutral_prior_dimension_count"] == 3
    assert summary["product_score_changed"] is False


def test_four_measured_dimensions_produce_high_evidence_without_claiming_probability() -> None:
    result = {
        "details": {
            "recording_quality": {"score": 0.91},
            "reliability": {"endpointing": 0.90},
        }
    }
    dimensions = [
        {"key": "delivery_fluency", "evidence_state": "measured_proxy", "confidence": "high"},
        {"key": "clarity", "evidence_state": "measured_proxy", "confidence": "medium"},
        {"key": "mora_timing", "evidence_state": "measured_proxy", "confidence": "medium"},
        {"key": "intonation", "evidence_state": "measured_proxy", "confidence": "medium"},
    ]

    summary = build_score_evidence_quality(result, dimensions)

    assert summary["score_evidence"]["level"] == "high"
    assert summary["score_evidence"]["coverage_index"] == 1.0
    assert summary["score_evidence"]["interpretation"] == "evidence_coverage_not_probability_score_is_correct"


def test_broad_proxy_is_partial_evidence_not_full_measurement() -> None:
    result = {"details": {"recording_quality": {"score": 0.8}, "reliability": {"endpointing": 0.8}}}
    dimensions = [
        {"key": "delivery_fluency", "evidence_state": "broad_proxy"},
        {"key": "clarity", "evidence_state": "broad_proxy"},
        {"key": "mora_timing", "evidence_state": "broad_proxy"},
        {"key": "intonation", "evidence_state": "broad_proxy"},
    ]

    summary = build_score_evidence_quality(result, dimensions)

    assert summary["score_evidence"]["coverage_index"] == 0.6
    assert summary["score_evidence"]["level"] == "medium"
    assert summary["score_evidence"]["broad_dimension_count"] == 4


def test_user_facing_contract_exposes_clean_recording_but_weak_score_evidence() -> None:
    result = {
        "total_score": 72,
        "pronunciation_score": 70,
        "prosody_score": 70,
        "fluency_score": 78,
        "tone_score": 70,
        "alignment_mode": "none",
        "target_text": "今日はいい天気です",
        "moras": ["キョ", "ウ", "ワ", "イ", "イ", "テ", "ン", "キ", "デ", "ス"],
        "mora_table": [],
        "feedback": [],
        "details": {
            "mode": "transcript_assisted_light",
            "score_eligible": True,
            "content_match": {"status": "general_japanese"},
            "recording_quality": {"score": 0.96},
            "reliability": {
                "level": "high",
                "overall": 0.95,
                "recording_quality": 0.96,
                "endpointing": 0.94,
                "alignment": 1.0,
                "f0_coverage": 0.0,
            },
            "fluency": {},
            "transcript_sanity": {"ok": True},
        },
    }

    user = render_user_facing_result(result, mode="transcript_assisted_light")

    assert user["display_score"] is not None
    assert user["recording_analyzability"]["level"] == "high"
    assert user["score_evidence"]["level"] == "low"
    assert user["score_evidence"]["neutral_prior_dimension_count"] == 3
    assert user["score_evidence"]["interpretation"] == "evidence_coverage_not_probability_score_is_correct"
