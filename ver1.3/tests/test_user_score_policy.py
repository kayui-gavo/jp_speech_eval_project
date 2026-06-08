from __future__ import annotations

from jp_speech_eval.user_score_policy import apply_user_score_policy


def _result(**overrides):
    result = {
        "total_score": 92,
        "pronunciation_score": 88,
        "prosody_score": 86,
        "fluency_score": 87,
        "moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"],
        "alignment_mode": "cached_dtw",
        "details": {
            "mode": "reference_based",
            "reliability": {"level": "high", "overall": 0.95, "alignment": 0.9, "f0_coverage": 0.9},
            "recording_quality": {"score": 0.9},
            "content_match": {"status": "pass"},
            "alignment": {"mode": "cached_dtw"},
            "fluency": {"rhythm_timing_score": 86, "delivery_fluency_score": 87},
        },
    }
    for key, value in overrides.items():
        if key == "details":
            result["details"].update(value)
        else:
            result[key] = value
    return result


def test_bad_pronunciation_cannot_be_lifted_to_high_display_score():
    policy = apply_user_score_policy(_result(pronunciation_score=55, prosody_score=98, fluency_score=98))
    assert policy["display_score"] <= 65
    assert policy["pronunciation_clarity_score"] <= 55
    assert policy["display_score"] <= policy["pronunciation_clarity_score"] + 5
    assert policy["main_message_key"] == "clear_recording_but_pronunciation_needs_practice"


def test_pronunciation_under_70_cannot_be_lifted_above_75():
    policy = apply_user_score_policy(_result(pronunciation_score=66, prosody_score=100, fluency_score=100))
    assert policy["display_score"] <= 75
    assert policy["display_score"] <= policy["pronunciation_clarity_score"] + 5


def test_fallback_alignment_caps_display_and_pronunciation_clarity():
    policy = apply_user_score_policy(_result(alignment_mode="cached_dtw_fallback_equal"))
    assert policy["display_score"] is None
    assert policy["pronunciation_clarity_score"] is None
    assert "alignment_fallback_cap" in policy["score_policy_warnings"]
    assert "alignment_fallback_no_display_score" in policy["score_policy_warnings"]
    assert policy["detail_feedback_allowed"] is False
    assert policy["scoring_gate"]["alignment_ok"] is False
    assert policy["scoring_gate"]["score_available"] is False


def test_low_alignment_hides_display_and_pronunciation_clarity():
    policy = apply_user_score_policy(_result(details={"reliability": {"level": "medium", "overall": 0.7, "alignment": 0.3}}))
    assert policy["display_score"] is None
    assert policy["pronunciation_clarity_score"] is None
    assert "low_alignment_no_display_score" in policy["score_policy_warnings"]
    assert policy["detail_feedback_allowed"] is False


def test_bad_recording_hides_display_score():
    policy = apply_user_score_policy(_result(details={"recording_quality": {"score": 0.4}}))
    assert policy["display_score"] is None
    assert policy["pronunciation_clarity_score"] is None
    assert "recording_quality_no_display_score" in policy["score_policy_warnings"]
    assert policy["scoring_gate"]["recording_ok"] is False


def test_content_failed_hides_pronunciation_score():
    policy = apply_user_score_policy(_result(details={"content_match": {"status": "fail"}}))
    assert policy["display_score"] is None
    assert policy["pronunciation_clarity_score"] is None
    assert policy["confidence_label"] == "low"
    assert policy["scoring_gate"]["target_match_ok"] is False


def test_fluency_and_prosody_cannot_lift_failed_content_case():
    policy = apply_user_score_policy(
        _result(
            pronunciation_score=95,
            prosody_score=100,
            fluency_score=100,
            details={
                "content_match": {"status": "fail"},
                "fluency": {"rhythm_timing_score": 100, "delivery_fluency_score": 100},
            },
        )
    )
    assert policy["display_score"] is None
    assert policy["pronunciation_clarity_score"] is None


def test_weak_reference_hides_display_score_and_confidence_limited():
    policy = apply_user_score_policy(
        _result(details={"weak_reference": True}),
        mode="asr_confirmed_weak_reference",
    )
    assert policy["display_score"] is None
    assert policy["confidence_label"] in {"low", "medium"}
    assert "weak_reference_no_display_score" in policy["score_policy_warnings"]


def test_short_sentence_caps_detail_display():
    policy = apply_user_score_policy(_result(moras=["ア", "メ", "ガ"]))
    assert policy["display_score"] <= 80
    assert policy["detail_feedback_allowed"] is False
    assert "short_sentence_cap" in policy["score_policy_warnings"]


def test_special_mora_user_hint_is_only_a_soft_penalty():
    policy = apply_user_score_policy(
        _result(pronunciation_score=86),
        special_mora_decisions=[{"type": "long_vowel", "user_feedback_allowed": True}],
    )
    assert policy["pronunciation_clarity_score"] == 80
    assert "special_mora_soft_penalty" in policy["score_policy_warnings"]
