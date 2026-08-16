from __future__ import annotations

import pytest

from jp_speech_eval.partial_evidence_aggregate import build_partial_evidence_aggregate_candidate
from jp_speech_eval.score_contract import apply_display_transform


def _item(value, state):
    return {
        "value": value,
        "evidence_state": state,
        "confidence": "low",
    }


def test_all_measured_components_reproduce_frozen_weighted_aggregate() -> None:
    components = {
        "clarity": _item(80, "measured_proxy"),
        "mora_timing": _item(60, "measured_proxy"),
        "delivery_fluency": _item(90, "measured_proxy"),
        "intonation": _item(70, "measured_proxy"),
    }
    candidate = build_partial_evidence_aggregate_candidate(components)
    expected_raw = 0.30 * 80 + 0.25 * 60 + 0.25 * 90 + 0.20 * 70
    assert candidate["available"] is True
    assert candidate["effective_coverage"] == pytest.approx(1.0)
    assert candidate["candidate_raw_score"] == pytest.approx(expected_raw)
    assert candidate["candidate_display_score"] == pytest.approx(apply_display_transform(expected_raw))


def test_neutral_priors_do_not_pull_a_real_fluency_measurement_toward_70_as_equal_evidence() -> None:
    components = {
        "clarity": _item(70, "neutral_prior"),
        "mora_timing": _item(70, "neutral_prior"),
        "delivery_fluency": _item(90, "broad_proxy"),
        "intonation": _item(70, "neutral_prior"),
    }
    candidate = build_partial_evidence_aggregate_candidate(components)
    # Only 0.25 * 0.75 of the contract weight is real evidence. The candidate
    # therefore moves toward the observed fluency value but remains shrunk.
    assert candidate["available"] is True
    assert candidate["available_component_count"] == 1
    assert candidate["neutral_prior_count"] == 3
    assert candidate["effective_coverage"] == pytest.approx(0.1875)
    assert 70 < candidate["candidate_raw_score"] < 90
    assert candidate["components"]["clarity"]["effective_weight"] == 0
    assert candidate["components"]["clarity"]["exclusion_reason"] == "neutral_prior_not_measurement"


def test_neutral_prior_numeric_value_cannot_change_candidate() -> None:
    first = {
        "clarity": _item(70, "neutral_prior"),
        "mora_timing": _item(70, "neutral_prior"),
        "delivery_fluency": _item(82, "broad_proxy"),
        "intonation": _item(70, "neutral_prior"),
    }
    second = {
        "clarity": _item(15, "neutral_prior"),
        "mora_timing": _item(99, "neutral_prior"),
        "delivery_fluency": _item(82, "broad_proxy"),
        "intonation": _item(42, "neutral_prior"),
    }
    a = build_partial_evidence_aggregate_candidate(first)
    b = build_partial_evidence_aggregate_candidate(second)
    assert a["candidate_raw_score"] == b["candidate_raw_score"]
    assert a["candidate_display_score"] == b["candidate_display_score"]


def test_no_real_component_evidence_returns_unavailable_not_fake_70_measurement() -> None:
    components = {
        "clarity": _item(70, "neutral_prior"),
        "mora_timing": _item(70, "neutral_prior"),
        "delivery_fluency": _item(None, "unavailable"),
        "intonation": _item(70, "neutral_prior"),
    }
    candidate = build_partial_evidence_aggregate_candidate(components)
    assert candidate["available"] is False
    assert candidate["candidate_raw_score"] is None
    assert candidate["candidate_display_score"] is None
    assert candidate["fallback_numeric_anchor"] == 70
    assert candidate["product_score_changed"] is False
    assert candidate["user_facing"] is False


def test_low_fluency_moves_down_without_turning_missing_dimensions_into_penalties() -> None:
    components = {
        "clarity": _item(70, "neutral_prior"),
        "mora_timing": _item(70, "neutral_prior"),
        "delivery_fluency": _item(50, "broad_proxy"),
        "intonation": _item(70, "neutral_prior"),
    }
    candidate = build_partial_evidence_aggregate_candidate(components)
    assert 50 < candidate["candidate_raw_score"] < 70
    assert candidate["components"]["clarity"]["included"] is False
    assert candidate["components"]["intonation"]["included"] is False
