from __future__ import annotations

import pytest

from jp_speech_eval.product_score_v3 import (
    attach_ssl_pronunciation_candidate,
    build_product_score_v3_candidate,
    reference_relative_timing_features,
)


def _candidate(**extra):
    values = {
        "alignment_mode": "cached_dtw",
        "boundaries": [(0.0, .1), (.1, .3), (.3, .42), (.42, .56)],
        "reference_boundaries": [(0.0, .12), (.12, .28), (.28, .42), (.42, .60)],
        "user_duration_sec": .56,
        "reference_duration_sec": .60,
        "pause_info": {"pause_ratio": .03, "pause_count": 0, "pause_total": .0},
        "legacy_pronunciation_score": 91,
        "legacy_prosody_score": 88,
        "f0_coverage": .8,
        "alignment_confidence": .9,
    }
    values.update(extra)
    return build_product_score_v3_candidate(**values)


def test_fallback_equal_never_becomes_perfect_local_timing_evidence():
    features = reference_relative_timing_features(
        [(0, .1), (.1, .2), (.2, .3)],
        [(0, .1), (.1, .2), (.2, .3)],
        user_duration_sec=.3,
        reference_duration_sec=.3,
        pause_info={},
        alignment_mode="cached_dtw_fallback_equal",
    )
    assert not features["available"]
    assert features["evidence_source"] == "synthetic_equal_boundaries"
    candidate = _candidate(alignment_mode="cached_dtw_fallback_equal")
    assert not candidate["dimensions"]["rhythm"]["available"]
    assert not candidate["dimensions"]["intonation"]["available"]
    assert candidate["dimensions"]["pronunciation"]["value"] is None


def test_v3_reweights_available_dimensions_instead_of_imputing_80():
    candidate = _candidate()
    aggregate = candidate["product_score_v3_candidate"]
    assert aggregate["available"]
    assert "pronunciation" in aggregate["unavailable_dimensions"]
    assert sum(aggregate["weights_effective"].values()) == pytest.approx(1.0, abs=2e-4)
    assert candidate["dimensions"]["rhythm"]["source"] == "reference_relative_warp"


def test_local_distortion_changes_rhythm_more_than_uniform_tempo():
    baseline = _candidate()["dimensions"]["rhythm"]["value"]
    uniform = _candidate(
        boundaries=[(0, .09), (.09, .27), (.27, .378), (.378, .54)],
        user_duration_sec=.54,
    )["dimensions"]["rhythm"]["value"]
    local = _candidate(
        boundaries=[(0, .04), (.04, .34), (.34, .38), (.38, .56)],
        user_duration_sec=.56,
    )["dimensions"]["rhythm"]["value"]
    assert abs(uniform - baseline) < abs(local - baseline)


def test_ssl_candidate_can_be_attached_without_changing_v2_inputs():
    candidate = _candidate()
    updated = attach_ssl_pronunciation_candidate(candidate, 72.5, confidence=.8)
    assert updated["dimensions"]["pronunciation"]["value"] == 72.5
    assert updated["product_score_v3_candidate"]["weights_effective"]["pronunciation"] > 0
    assert updated["user_facing"] is False
