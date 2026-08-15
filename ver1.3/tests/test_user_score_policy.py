from __future__ import annotations

from jp_speech_eval.user_score_policy import PRODUCT_COMPONENT_WEIGHTS, apply_user_score_policy


def _result(**overrides):
    result = {
        "total_score": 92,
        "pronunciation_score": 88,
        "prosody_score": 86,
        "fluency_score": 87,
        "moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"],
        "mora_table": [
            {"f0_hz": 180.0}, {"f0_hz": 190.0}, {"f0_hz": 200.0},
            {"f0_hz": 195.0}, {"f0_hz": 185.0}, {"f0_hz": 178.0},
            {"f0_hz": 170.0}, {"f0_hz": 165.0}, {"f0_hz": 160.0},
        ],
        "alignment_mode": "cached_dtw",
        "details": {
            "mode": "reference_based",
            "reliability": {
                "level": "high",
                "overall": 0.95,
                "alignment": 0.9,
                "f0_coverage": 0.9,
                "endpointing": 1.0,
                "duration_ratio_to_reference": 1.0,
            },
            "recording_quality": {"score": 0.9},
            "content_match": {
                "status": "pass",
                "kana_similarity": 0.94,
                "score": 0.88,
                "duration_ratio": 1.0,
            },
            "alignment": {
                "mode": "cached_dtw",
                "available": True,
                "used_equal_fallback": False,
                "normalized_dtw_cost": 3.7,
            },
            "fluency": {
                "rhythm_timing_score": 86,
                "delivery_fluency_score": 87,
                "rate_score": 84,
                "pause_score": 90,
            },
            "prosody": {
                "contour_valid_mora_count": 8,
                "contour_corr": 0.7,
                "note": "ok",
            },
            "reference_f0_by_mora": [178.0, 188.0, 198.0, 193.0, 184.0, 176.0, 169.0, 164.0, 159.0],
            "tone": {"pitch_range_log": 0.3, "pitch_score": 88},
            "shadow": {},
        },
    }
    for key, value in overrides.items():
        if key == "details":
            for detail_key, detail_value in value.items():
                if isinstance(detail_value, dict) and isinstance(result["details"].get(detail_key), dict):
                    result["details"][detail_key].update(detail_value)
                else:
                    result["details"][detail_key] = detail_value
        else:
            result[key] = value
    return result


def test_product_score_is_available_and_uses_four_semantic_components():
    policy = apply_user_score_policy(_result())
    assert policy["display_score"] is not None
    assert set(policy["component_scores"]) == set(PRODUCT_COMPONENT_WEIGHTS)
    assert policy["score_formula"]["policy"] == "semantic_four_component_product_heuristic_v1"
    assert policy["score_formula"]["weights"] == PRODUCT_COMPONENT_WEIGHTS
    assert policy["score_formula"]["product_calibrated"] is False
    assert policy["pronunciation_clarity_score"] == policy["component_scores"]["clarity"]["value"]
    assert policy["intonation_score"] == policy["component_scores"]["intonation"]["value"]


def test_legacy_pronunciation_field_only_moves_rhythm_not_clarity():
    low = apply_user_score_policy(_result(pronunciation_score=55))
    high = apply_user_score_policy(_result(pronunciation_score=88))
    assert low["component_scores"]["clarity"]["value"] == high["component_scores"]["clarity"]["value"]
    assert low["component_scores"]["mora_timing"]["value"] < high["component_scores"]["mora_timing"]["value"]
    assert low["display_score"] < high["display_score"]


def test_intonation_now_contributes_to_overall_score():
    low = _result(prosody_score=45)
    high = _result(prosody_score=92)
    low_policy = apply_user_score_policy(low)
    high_policy = apply_user_score_policy(high)
    assert low_policy["component_scores"]["intonation"]["value"] == 45
    assert high_policy["component_scores"]["intonation"]["value"] == 92
    assert low_policy["display_score"] < high_policy["display_score"]


def test_fallback_alignment_keeps_broad_score_but_hides_detail():
    policy = apply_user_score_policy(
        _result(
            alignment_mode="cached_dtw_fallback_equal",
            details={"alignment": {"available": False, "used_equal_fallback": True}},
        )
    )
    assert policy["display_score"] is not None
    assert policy["detail_feedback_allowed"] is False
    assert "alignment_fallback_broad_score_only" in policy["score_policy_warnings"]
    assert policy["scoring_gate"]["alignment_ok"] is False


def test_low_alignment_keeps_score_but_lowers_confidence():
    policy = apply_user_score_policy(
        _result(details={"reliability": {"level": "low", "overall": 0.3, "alignment": 0.2}})
    )
    assert policy["display_score"] is not None
    assert policy["detail_feedback_allowed"] is False
    assert policy["confidence_label"] == "low"


def test_moderately_bad_recording_keeps_score():
    policy = apply_user_score_policy(_result(details={"recording_quality": {"score": 0.4}}))
    assert policy["display_score"] is not None
    assert policy["detail_feedback_allowed"] is False
    assert "recording_quality_low_confidence_score" in policy["score_policy_warnings"]


def test_unusable_recording_can_still_be_unscorable():
    policy = apply_user_score_policy(_result(details={"recording_quality": {"score": 0.0}}))
    assert policy["display_score"] is None
    assert policy["score_available"] is False
    assert policy["component_scores"] == {}


def test_content_mismatch_keeps_general_score_without_target_relative_penalty():
    result = _result(
        pronunciation_score=10,
        prosody_score=10,
        details={
            "content_match": {
                "status": "fail",
                "kana_similarity": 0.02,
                "score": 0.05,
                "duration_ratio": 2.2,
            },
            "reliability": {"duration_ratio_to_reference": 2.2},
            "fluency": {"rate_score": 82, "pause_score": 86},
            "tone": {"pitch_range_log": 0.3, "pitch_score": 84},
        },
    )
    policy = apply_user_score_policy(result)
    assert policy["display_score"] is not None
    assert policy["detail_feedback_allowed"] is False
    assert policy["scoring_gate"]["target_match_ok"] is False
    assert policy["main_message_key"] == "content_mismatch_general_score"
    assert policy["component_scores"]["clarity"]["value"] == 70
    assert policy["component_scores"]["clarity"]["evidence_tier"] == "reference_independent_prior_fallback"
    assert policy["component_scores"]["mora_timing"]["value"] == 82
    assert policy["component_scores"]["mora_timing"]["evidence_tier"] == "reference_independent_rate_fallback"
    assert policy["component_scores"]["intonation"]["value"] != 10


def test_weak_reference_keeps_practice_score():
    policy = apply_user_score_policy(
        _result(details={"weak_reference": True}),
        mode="asr_confirmed_weak_reference",
    )
    assert policy["display_score"] is not None
    assert policy["detail_feedback_allowed"] is False
    assert policy["confidence_label"] in {"low", "medium"}
    assert "weak_reference_broad_score_only" in policy["score_policy_warnings"]


def test_short_sentence_keeps_score_but_hides_detail():
    policy = apply_user_score_policy(_result(moras=["ア", "メ", "ガ"]))
    assert policy["display_score"] is not None
    assert policy["detail_feedback_allowed"] is False
    assert "short_utterance_broad_score_only" in policy["score_policy_warnings"]


def test_special_mora_feedback_is_not_double_penalized_or_mislabeled_as_clarity():
    baseline = apply_user_score_policy(_result())
    policy = apply_user_score_policy(
        _result(),
        special_mora_decisions=[{"type": "long_vowel", "user_feedback_allowed": True}],
    )
    assert policy["display_score"] == baseline["display_score"]
    assert policy["pronunciation_clarity_score"] == baseline["pronunciation_clarity_score"]
    assert "special_mora_feedback_no_extra_double_penalty" in policy["score_policy_warnings"]


def test_explicit_non_japanese_sanity_rejection_is_unscorable():
    policy = apply_user_score_policy(_result(details={"transcript_sanity": {"ok": False}}))
    assert policy["display_score"] is None
    assert policy["score_available"] is False
