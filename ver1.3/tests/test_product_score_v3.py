from __future__ import annotations

import math

import pytest

from jp_speech_eval.product_score_v3 import (
    attach_ssl_pronunciation_candidate,
    attach_ssl_pronunciation_evidence,
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
    assert features["global_timing_available"]
    assert features["global_rate_log_ratio"] == pytest.approx(0.0)


def test_v3_reweights_available_dimensions_instead_of_imputing_80():
    candidate = _candidate()
    aggregate = candidate["product_score_v3_candidate"]
    assert aggregate["available"]
    assert "pronunciation" in aggregate["unavailable_dimensions"]
    assert sum(aggregate["weights_effective"].values()) == pytest.approx(1.0, abs=2e-4)
    assert candidate["dimensions"]["rhythm"]["source"] == "reference_relative_warp"
    assert aggregate["score_scope"] == "delivery_prosody"
    assert aggregate["evidence_coverage"] == pytest.approx(.55)
    assert aggregate["diagnostic_candidate_eligible"]
    assert not aggregate["overall_product_score_candidate_eligible"]
    assert not aggregate["ab_candidate_eligible"]


def test_local_alignment_missing_uses_real_global_rate_not_perfect_rate():
    candidate = _candidate(
        alignment_mode="cached_dtw_fallback_equal",
        user_duration_sec=1.2,
        reference_duration_sec=.6,
        pause_info={"pause_ratio": 0.0, "pause_count": 0, "pause_total": 0.0},
    )
    timing = candidate["timing_features"]
    assert timing["global_rate_log_ratio"] == pytest.approx(math.log(2.0))
    assert candidate["dimensions"]["fluency"]["value"] < 100


def test_single_dimension_candidate_has_continuity_scope_but_not_ab_eligibility():
    candidate = _candidate(alignment_mode="cached_dtw_fallback_equal", f0_coverage=.1)
    aggregate = candidate["product_score_v3_candidate"]
    assert aggregate["score_scope"] == "continuity_only"
    assert aggregate["evidence_coverage"] == pytest.approx(.20)
    assert not aggregate["diagnostic_candidate_eligible"]
    assert not aggregate["ab_candidate_eligible"]


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
    assert updated["product_score_v3_candidate"]["overall_product_score_candidate_eligible"]
    assert updated["user_facing"] is False


def test_ssl_confidence_is_not_alignment_confidence():
    candidate = _candidate(alignment_confidence=.01, ssl_pronunciation={"available": True, "candidate_score": 72.5, "ssl_pronunciation_confidence": .81})
    assert candidate["dimensions"]["pronunciation"]["confidence"] == pytest.approx(.81)


def test_codec_like_alignment_failure_can_keep_global_ssl_evidence_without_overall_eligibility():
    candidate = attach_ssl_pronunciation_evidence(
        _candidate(alignment_mode="cached_dtw_fallback_equal", f0_coverage=.1),
        evidence_index=1.4,
        ssl_pronunciation_confidence=.76,
        reference_count=4,
        reference_dispersion=.03,
        content_verified=True,
        audio_valid=True,
    )
    aggregate = candidate["product_score_v3_candidate"]
    assert candidate["dimensions"]["pronunciation"]["available"]
    assert candidate["dimensions"]["pronunciation"]["value"] is None
    assert aggregate["score_scope"] == "pronunciation_plus_delivery"
    assert aggregate["diagnostic_candidate_eligible"]
    assert not aggregate["overall_product_score_candidate_eligible"]
    assert aggregate["overall_product_score_candidate_eligibility_reason"] == "pronunciation_evidence_not_score_mapped"


def test_ssl_evidence_requires_verified_content_and_valid_audio():
    evidence = attach_ssl_pronunciation_evidence(
        _candidate(),
        evidence_index=.22,
        ssl_pronunciation_confidence=.8,
        reference_count=4,
        reference_dispersion=.02,
        content_verified=False,
        audio_valid=True,
    )
    assert not evidence["ssl_pronunciation_evidence"]["available"]
    assert evidence["ssl_pronunciation_evidence"]["reason"] == "ssl_requires_verified_content"
    assert not evidence["dimensions"]["pronunciation"]["available"]
