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


def test_product_score_is_continuous_and_available_for_normal_case():
    low = apply_user_score_policy(_result(pronunciation_score=55))
    mid = apply_user_score_policy(_result(pronunciation_score=70))
    high = apply_user_score_policy(_result(pronunciation_score=88))
    assert low["display_score"] is not None
    assert low["display_score"] < mid["display_score"] < high["display_score"]
    assert high["display_cap_applied"] is False


def test_fallback_alignment_keeps_broad_score_but_hides_detail():
    policy = apply_user_score_policy(_result(alignment_mode="cached_dtw_fallback_equal"))
    assert policy["display_score"] is not None
    assert policy["detail_feedback_allowed"] is False
    assert "alignment_fallback_broad_score_only" in policy["score_policy_warnings"]
    assert policy["scoring_gate"]["alignment_ok"] is False


def test_low_alignment_keeps_score_but_lowers_confidence():
    policy = apply_user_score_policy(_result(details={"reliability": {"level": "low", "overall": 0.3, "alignment": 0.2}}))
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


def test_content_mismatch_keeps_general_score_but_hides_target_detail():
    policy = apply_user_score_policy(_result(details={"content_match": {"status": "fail"}}))
    assert policy["display_score"] is not None
    assert policy["pronunciation_clarity_score"] is not None
    assert policy["detail_feedback_allowed"] is False
    assert policy["scoring_gate"]["target_match_ok"] is False


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


def test_special_mora_user_hint_is_only_a_soft_penalty():
    baseline = apply_user_score_policy(_result(pronunciation_score=86))
    policy = apply_user_score_policy(
        _result(pronunciation_score=86),
        special_mora_decisions=[{"type": "long_vowel", "user_feedback_allowed": True}],
    )
    assert policy["display_score"] < baseline["display_score"]
    assert policy["pronunciation_clarity_score"] == 82
    assert "special_mora_soft_penalty" in policy["score_policy_warnings"]


def test_explicit_non_japanese_sanity_rejection_is_unscorable():
    policy = apply_user_score_policy(_result(details={"transcript_sanity": {"ok": False}}))
    assert policy["display_score"] is None
    assert policy["score_available"] is False
