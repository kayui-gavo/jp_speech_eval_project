from __future__ import annotations

from jp_speech_eval.consumer_dimension_policy import build_consumer_score_dimensions
from jp_speech_eval.user_score_policy import PRODUCT_COMPONENT_WEIGHTS, apply_user_score_policy


SEMANTIC_KEYS = ["delivery_fluency", "clarity", "mora_timing", "intonation"]


def _result() -> dict:
    return {
        "pronunciation_score": 78,
        "prosody_score": 82,
        "fluency_score": 84,
        "moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"],
        "mora_table": [
            {"f0_hz": 180.0}, {"f0_hz": 192.0}, {"f0_hz": 204.0},
            {"f0_hz": 198.0}, {"f0_hz": 188.0}, {"f0_hz": 178.0},
            {"f0_hz": 170.0}, {"f0_hz": 164.0}, {"f0_hz": 159.0},
        ],
        "alignment_mode": "cached_dtw",
        "details": {
            "reliability": {
                "level": "high",
                "overall": 0.92,
                "alignment": 0.90,
                "f0_coverage": 0.90,
                "endpointing": 1.0,
                "duration_ratio_to_reference": 1.03,
            },
            "recording_quality": {"score": 0.9},
            "content_match": {
                "status": "pass",
                "kana_similarity": 0.93,
                "score": 0.88,
                "duration_ratio": 1.03,
            },
            "alignment": {
                "available": True,
                "used_equal_fallback": False,
                "mode": "cached_dtw",
                "normalized_dtw_cost": 3.8,
            },
            "fluency": {
                "rate_score": 82,
                "pause_score": 88,
                "delivery_fluency_score": 85,
            },
            "prosody": {
                "contour_valid_mora_count": 8,
                "contour_corr": 0.72,
                "transition_agreement": 0.75,
                "note": "ok",
            },
            "reference_f0_by_mora": [178.0, 190.0, 202.0, 196.0, 186.0, 176.0, 168.0, 162.0, 157.0],
            "tone": {"pitch_range_log": 0.3, "pitch_score": 86},
            "shadow": {},
        },
    }


def test_visible_dimensions_are_exactly_the_components_used_by_total_policy() -> None:
    result = _result()
    policy = apply_user_score_policy(result, mode="reference")
    dimensions = build_consumer_score_dimensions(result, policy, mode="reference")

    assert [item["key"] for item in dimensions] == SEMANTIC_KEYS
    assert set(policy["component_scores"]) == set(SEMANTIC_KEYS)
    for dimension in dimensions:
        component = policy["component_scores"][dimension["key"]]
        assert dimension["value"] == component["value"]
        assert dimension["confidence"] == component["confidence"]
        assert dimension["evidence_tier"] == component["evidence_tier"]
    assert policy["score_formula"]["weights"] == PRODUCT_COMPONENT_WEIGHTS
    assert policy["score_formula"]["product_calibrated"] is False


def test_lexical_pitch_accent_is_not_a_hidden_fifth_total_component() -> None:
    result = _result()
    result["details"]["prosody"]["pitch_accent_score"] = 0
    policy = apply_user_score_policy(result, mode="reference")
    dimensions = build_consumer_score_dimensions(result, policy, mode="reference")
    assert "pitch_accent" not in policy["component_scores"]
    assert "pitch_accent" not in {item["key"] for item in dimensions}
    assert set(PRODUCT_COMPONENT_WEIGHTS) == {"clarity", "mora_timing", "delivery_fluency", "intonation"}


def test_valid_off_target_japanese_keeps_four_scores_but_drops_target_relative_evidence() -> None:
    result = _result()
    result["pronunciation_score"] = 5
    result["prosody_score"] = 7
    result["details"]["content_match"].update({
        "status": "fail",
        "kana_similarity": 0.01,
        "score": 0.01,
        "duration_ratio": 2.5,
    })
    result["details"]["reliability"]["duration_ratio_to_reference"] = 2.5
    result["details"]["fluency"]["rate_score"] = 80
    result["details"]["tone"] = {"pitch_range_log": 0.25, "pitch_score": 84}

    policy = apply_user_score_policy(result, mode="reference")
    dimensions = build_consumer_score_dimensions(result, policy, mode="reference")
    by_key = {item["key"]: item for item in dimensions}

    assert policy["display_score"] is not None
    assert len(dimensions) == 4
    assert all(item["available"] and item["value"] is not None for item in dimensions)
    assert by_key["clarity"]["evidence_tier"] == "reference_independent_prior_fallback"
    assert by_key["mora_timing"]["evidence_tier"] == "reference_independent_rate_fallback"
    assert by_key["intonation"]["evidence_tier"] == "pitch_range_fallback"
    assert by_key["mora_timing"]["value"] != 5
    assert by_key["intonation"]["value"] != 7
    assert policy["detail_feedback_allowed"] is False
    assert policy["main_message_key"] == "content_mismatch_general_score"
