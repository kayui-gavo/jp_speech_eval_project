from __future__ import annotations

from jp_speech_eval.practice_dimensions import (
    score_coarse_frame_pitch_fallback,
    score_pronunciation_clarity_practice,
    score_rhythm_timing_practice,
    weighted_four_dimension_overall,
)
from jp_speech_eval.scoring import score_fluency
from jp_speech_eval.special_mora_scorer import (
    RuntimeSpecialMoraDecision,
    special_mora_score_from_decisions,
)


def _boundaries(durations: list[float]) -> list[tuple[float, float]]:
    cursor = 0.0
    rows = []
    for duration in durations:
        rows.append((cursor, cursor + duration))
        cursor += duration
    return rows


def _decision(alignment_method: str) -> RuntimeSpecialMoraDecision:
    return RuntimeSpecialMoraDecision(
        type="long_vowel",
        surface_mora="ー",
        mora_index=2,
        phone_sequence_for_mora="o",
        feature_name="long_vowel_ratio_to_avg_mora",
        feature_value=0.8,
        threshold_low=0.23,
        threshold_high=1.1,
        threshold_status="active",
        decision="ok",
        evidence_confidence=0.9,
        confidence="high",
        alignment_method=alignment_method,
        mapping_success=True,
        mapping_warning_flags=[],
        user_feedback_allowed=False,
        suppression_reason="no_correction_needed",
        feedback_candidate_text="",
    )


def test_pronunciation_clarity_does_not_reward_equal_fallback_as_perfect() -> None:
    clean, details = score_pronunciation_clarity_practice(
        recording_quality={"score": 1.0},
        mora_evidence_summary={
            "mora_count": 12,
            "judgement_available_count": 12,
            "mean_energy_coverage": 1.0,
        },
        alignment_mode="cached_dtw_fallback_equal",
        content_match={"status": "unknown"},
    )
    degraded, _ = score_pronunciation_clarity_practice(
        recording_quality={"score": 0.45},
        mora_evidence_summary={
            "mora_count": 12,
            "judgement_available_count": 5,
            "mean_energy_coverage": 0.45,
        },
        alignment_mode="cached_dtw_fallback_equal",
        content_match={"status": "unknown"},
    )
    assert 70 <= clean <= 82
    assert degraded < clean
    assert details["alignment_fallback"] is True
    assert details["interpretation"].endswith("not_phone_level_pronunciation_correctness")


def test_pronunciation_clarity_ignores_nonfinite_optional_similarity() -> None:
    score, details = score_pronunciation_clarity_practice(
        recording_quality={"score": 0.9},
        mora_evidence_summary={
            "mora_count": 8,
            "judgement_available_count": 7,
            "mean_energy_coverage": 0.85,
        },
        alignment_mode="cached_dtw",
        content_match={"status": "pass", "kana_similarity": float("nan")},
    )
    assert 0 <= score <= 100
    assert details["components"]["content_intelligibility_evidence"] == 82.0


def test_robust_rhythm_separates_native_like_and_jitter_without_equal_inflation() -> None:
    native = [0.11, 0.15, 0.13, 0.18, 0.12, 0.16, 0.14, 0.19, 0.13, 0.15]
    jitter = [duration * (0.35 if i % 3 == 0 else 1.75 if i % 3 == 1 else 0.90) for i, duration in enumerate(native)]
    native_score, native_details = score_rhythm_timing_practice(
        boundaries=_boundaries(native),
        alignment_mode="mfa_phone_lab",
        rate_score=94,
    )
    jitter_score, _ = score_rhythm_timing_practice(
        boundaries=_boundaries(jitter),
        alignment_mode="mfa_phone_lab",
        rate_score=94,
    )
    fallback_score, fallback_details = score_rhythm_timing_practice(
        boundaries=_boundaries([0.14] * len(native)),
        alignment_mode="cached_dtw_fallback_equal",
        rate_score=94,
    )
    assert native_score > jitter_score + 15
    assert fallback_score < 80
    assert fallback_details["timing_evidence_available"] is False
    assert native_details["special_mora_included"] is False


def test_fluency_is_continuous_and_keeps_natural_pause_allowance() -> None:
    fluent, _, fluent_details = score_fluency(
        mora_count=24,
        duration=4.0,
        pause_info={"pause_ratio": 0.12, "pause_count": 1},
    )
    fast, _, _ = score_fluency(
        mora_count=24,
        duration=2.2,
        pause_info={"pause_ratio": 0.02, "pause_count": 0},
    )
    hesitant, _, hesitant_details = score_fluency(
        mora_count=24,
        duration=6.0,
        pause_info={"pause_ratio": 0.42, "pause_count": 6},
    )
    assert 90 <= fluent < 100
    assert fast < fluent
    assert hesitant < fluent
    assert fluent_details["delivery_fluency_components"]["pause_count_excess"] == 0
    assert hesitant_details["delivery_fluency_components"]["pause_count_excess"] > 0


def test_overall_uses_all_four_displayed_dimensions() -> None:
    high, details = weighted_four_dimension_overall(
        pronunciation=85,
        rhythm=80,
        fluency=95,
        pitch=75,
    )
    low_fluency, _ = weighted_four_dimension_overall(
        pronunciation=85,
        rhythm=80,
        fluency=35,
        pitch=75,
    )
    assert high is not None and low_fluency is not None
    assert high > low_fluency + 10
    assert details["available_dimensions"] == ["pronunciation", "rhythm", "fluency", "pitch"]


def test_special_mora_summary_rejects_fallback_boundaries() -> None:
    assert special_mora_score_from_decisions([_decision("cached_dtw_fallback_equal")]) is None
    assert special_mora_score_from_decisions([_decision("mfa_phone_lab")]) == 100.0


def test_frame_pitch_fallback_requires_real_f0_and_stays_conservative() -> None:
    contour = [float("nan")] * 20 + [140 + (index % 12) * 2 for index in range(50)] + [float("nan")] * 30
    score, details = score_coarse_frame_pitch_fallback(contour)
    assert score is not None
    assert 45 <= score <= 78
    assert details["confidence"] == "low"
    unavailable, unavailable_details = score_coarse_frame_pitch_fallback([float("nan")] * 100)
    assert unavailable is None
    assert unavailable_details["available"] is False
