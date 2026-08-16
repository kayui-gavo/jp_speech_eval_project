from __future__ import annotations

from jp_speech_eval.measurement_context import build_measurement_context


def _dimension(key: str, state: str):
    return {"key": key, "evidence_state": state, "value": 70}


def test_free_speech_clean_recording_can_have_partial_score_evidence() -> None:
    result = {
        "details": {
            "recording_quality": {"score": 0.92},
            "reliability": {"recording_quality": 0.92},
        }
    }
    dimensions = [
        _dimension("clarity", "neutral_prior"),
        _dimension("mora_timing", "neutral_prior"),
        _dimension("delivery_fluency", "broad_proxy"),
        _dimension("intonation", "neutral_prior"),
    ]
    context = build_measurement_context(
        result,
        dimensions,
        reliability_gate={"reliability": "high", "practice_check_result": "ok", "blocked_categories": ["pitch"]},
        scoring_policy={"broad_mode": True, "weak_reference": False},
    )
    assert context["recording_state"] == "good"
    assert context["dimension_evidence"]["evidence_backed_count"] == 1
    assert context["dimension_evidence"]["neutral_prior_count"] == 3
    assert context["dimension_evidence"]["coverage_state"] == "partial_evidence"
    assert context["local_detail_state"] == "broad_only"
    assert context["single_confidence_percentage_allowed"] is False
    assert context["raw_reliability_numeric_user_facing"] is False


def test_fixed_reference_local_limit_does_not_turn_into_recording_problem() -> None:
    result = {
        "details": {
            "recording_quality": {"score": 0.91},
            "reliability": {"recording_quality": 0.91},
        }
    }
    dimensions = [
        _dimension("clarity", "broad_proxy"),
        _dimension("mora_timing", "broad_proxy"),
        _dimension("delivery_fluency", "measured_proxy"),
        _dimension("intonation", "broad_proxy"),
    ]
    context = build_measurement_context(
        result,
        dimensions,
        reliability_gate={
            "reliability": "high",
            "practice_check_result": "ok",
            "blocked_categories": ["pitch", "special_mora", "pronunciation_detail"],
            "messages": ["参照音声の細かい拍位置が概算です。"],
        },
        scoring_policy={"broad_mode": False, "weak_reference": False},
    )
    assert context["recording_state"] == "good"
    assert context["local_detail_state"] == "limited"
    assert context["primary_limitation"].startswith("参照音声")
    assert context["dimension_evidence"]["evidence_backed_count"] == 4


def test_unusable_recording_is_retry_but_not_a_low_learner_confidence_percent() -> None:
    result = {"details": {"recording_quality": {"score": 0.10}}}
    context = build_measurement_context(
        result,
        [],
        reliability_gate={"reliability": "unscorable", "practice_check_result": "retry", "blocked_categories": []},
        scoring_policy={"broad_mode": True, "weak_reference": False},
    )
    assert context["recording_state"] == "retry"
    assert context["local_detail_state"] == "unavailable"
    assert context["dimension_evidence"]["coverage_state"] == "unavailable"
    assert context["single_confidence_percentage_allowed"] is False


def test_all_dimensions_with_evidence_is_described_as_coverage_not_probability() -> None:
    result = {"details": {"recording_quality": {"score": 0.80}}}
    dimensions = [
        _dimension("clarity", "measured_proxy"),
        _dimension("mora_timing", "broad_proxy"),
        _dimension("delivery_fluency", "measured_proxy"),
        _dimension("intonation", "broad_proxy"),
    ]
    context = build_measurement_context(
        result,
        dimensions,
        reliability_gate={"reliability": "high", "practice_check_result": "ok", "blocked_categories": []},
        scoring_policy={"broad_mode": False, "weak_reference": False},
    )
    assert context["dimension_evidence"]["coverage_state"] == "all_dimensions_have_evidence"
    assert context["dimension_evidence"]["evidence_backed_count"] == 4
    assert context["dimension_evidence"]["dimension_count"] == 4
    assert context["local_detail_state"] == "available"
