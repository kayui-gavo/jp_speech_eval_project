from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


def _clamp_score(value: float) -> int:
    return int(max(0, min(100, round(float(value)))))


def _clip01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not np.isfinite(number):
        number = default
    return float(max(0.0, min(1.0, number)))


def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _is_fallback_alignment(alignment_mode: str) -> bool:
    mode = str(alignment_mode or "").lower()
    return "fallback" in mode or mode in {"equal", "equal_fallback", "none"}


def score_pronunciation_clarity_practice(
    *,
    recording_quality: Mapping[str, Any] | None,
    mora_evidence_summary: Mapping[str, Any] | None,
    alignment_mode: str,
    content_match: Mapping[str, Any] | None = None,
) -> Tuple[int, dict[str, Any]]:
    """Return a conservative clarity proxy without pretending to be GOP.

    The old implementation used mora-duration variation as pronunciation.
    Natural Japanese duration variation then pushed native speech toward zero,
    while equal-time fallback produced 100 by construction. This score keeps
    timing out of pronunciation and combines only judgeability/intelligibility
    evidence that is available in the current product.
    """

    quality = recording_quality or {}
    evidence = mora_evidence_summary or {}
    content = content_match or {}
    fallback = _is_fallback_alignment(alignment_mode)

    recording_component = 100.0 * _clip01(quality.get("score"), 0.70)
    energy_component = 100.0 * _clip01(evidence.get("mean_energy_coverage"), 0.70)
    mora_count = int(evidence.get("mora_count") or 0)
    judged = int(evidence.get("judgement_available_count") or 0)
    judgement_ratio = judged / max(mora_count, 1) if mora_count else 0.0
    if fallback:
        # Equal boundaries contain no segmental timing information. Keep a
        # neutral evidence contribution instead of rewarding perfect equality.
        evidence_component = 65.0
        alignment_component = 55.0
    else:
        evidence_component = 55.0 + 45.0 * _clip01(judgement_ratio)
        alignment_component = 78.0 if "dtw" in str(alignment_mode).lower() else 90.0

    content_status = str(content.get("status") or "unknown")
    kana_similarity = _finite_float_or_none(content.get("kana_similarity"))
    if content_status == "fail":
        content_component = 20.0
    elif content_status == "uncertain":
        content_component = 55.0
    elif kana_similarity is not None and kana_similarity > 0:
        content_component = 60.0 + 40.0 * _clip01(kana_similarity)
    elif content_status == "pass":
        content_component = 82.0
    else:
        content_component = 75.0

    components = {
        "neutral_practice_prior": 75.0,
        "recording_clarity": recording_component,
        "mora_energy_coverage": energy_component,
        "mora_evidence_coverage": evidence_component,
        "alignment_evidence": alignment_component,
        "content_intelligibility_evidence": content_component,
    }
    weights = {
        "neutral_practice_prior": 0.10,
        "recording_clarity": 0.25,
        "mora_energy_coverage": 0.20,
        "mora_evidence_coverage": 0.15,
        "alignment_evidence": 0.15,
        "content_intelligibility_evidence": 0.15,
    }
    score = sum(components[key] * weights[key] for key in weights)
    adjustments: list[str] = []
    if fallback:
        score = min(score, 82.0)
        adjustments.append("fallback_alignment_caps_clarity_proxy")
    if recording_component < 55.0:
        score = min(score, 68.0)
        adjustments.append("low_recording_quality_caps_clarity_proxy")
    if mora_count and judgement_ratio < 0.40 and not fallback:
        score = min(score, 72.0)
        adjustments.append("low_mora_evidence_caps_clarity_proxy")

    confidence = "low" if fallback else "medium"
    if not fallback and recording_component >= 80.0 and judgement_ratio >= 0.70:
        confidence = "high"
    return _clamp_score(score), {
        "score_type": "pronunciation_clarity_practice_proxy_v2",
        "score": _clamp_score(score),
        "components": {key: round(float(value), 4) for key, value in components.items()},
        "weights": weights,
        "alignment_fallback": fallback,
        "judgement_available_ratio": round(float(judgement_ratio), 4),
        "confidence": confidence,
        "adjustments": adjustments,
        "interpretation": "recording_and_intelligibility_clarity_proxy_not_phone_level_pronunciation_correctness",
        "known_limit": "segment_substitutions_are_not_directly_scored_without_phone_posteriors_or_human_labels",
    }


def _duration_regularity_components(boundaries: Sequence[Tuple[float, float]]) -> dict[str, float]:
    durations = np.asarray([max(0.001, float(end) - float(start)) for start, end in boundaries], dtype=float)
    if durations.size < 3:
        return {
            "log_duration_mad": 0.0,
            "adjacent_log_duration_median": 0.0,
            "dispersion_score": 65.0,
            "transition_score": 65.0,
        }
    log_duration = np.log(durations)
    log_mad = float(np.median(np.abs(log_duration - np.median(log_duration))))
    adjacent = float(np.median(np.abs(np.diff(log_duration))))

    # Transparent robust bands derived from the JVS phone-label audit. They
    # preserve natural duration variation and separate the paired timing-jitter
    # control without treating equal durations as automatically perfect.
    dispersion_score = np.clip(
        96.0
        - 110.0 * max(0.0, log_mad - 0.30)
        - 30.0 * max(0.0, 0.18 - log_mad),
        35.0,
        98.0,
    )
    transition_score = np.clip(
        96.0
        - 55.0 * max(0.0, adjacent - 0.55)
        - 25.0 * max(0.0, 0.30 - adjacent),
        35.0,
        98.0,
    )
    return {
        "log_duration_mad": log_mad,
        "adjacent_log_duration_median": adjacent,
        "dispersion_score": float(dispersion_score),
        "transition_score": float(transition_score),
    }


def score_rhythm_timing_practice(
    *,
    boundaries: Sequence[Tuple[float, float]],
    alignment_mode: str,
    rate_score: Optional[float],
) -> Tuple[int, dict[str, Any]]:
    """Score broad mora timing while keeping fallback evidence honest."""

    fallback = _is_fallback_alignment(alignment_mode)
    tempo = float(rate_score) if rate_score is not None else 75.0
    tempo = max(0.0, min(100.0, tempo))
    if fallback:
        # There is no trustworthy within-utterance mora timing under equal
        # segmentation. Return a conservative numeric estimate for product UX,
        # but keep the confidence low and the range deliberately narrow.
        score = 65.0 + 0.18 * (tempo - 50.0)
        return _clamp_score(score), {
            "score_type": "rhythm_timing_practice_proxy_v2",
            "score": _clamp_score(score),
            "timing_evidence_available": False,
            "alignment_fallback": True,
            "confidence": "low",
            "tempo_score": round(float(tempo), 4),
            "interpretation": "coarse_tempo_estimate_without_mora_boundary_claims",
            "special_mora_included": False,
        }

    components = _duration_regularity_components(boundaries)
    regularity = 0.55 * components["dispersion_score"] + 0.45 * components["transition_score"]
    score = 0.85 * regularity + 0.15 * tempo
    confidence = "medium" if "dtw" in str(alignment_mode).lower() else "high"
    if confidence == "medium":
        score = min(score, 90.0)
    return _clamp_score(score), {
        "score_type": "rhythm_timing_practice_proxy_v2",
        "score": _clamp_score(score),
        "timing_evidence_available": True,
        "alignment_fallback": False,
        "confidence": confidence,
        "tempo_score": round(float(tempo), 4),
        "regularity_score": round(float(regularity), 4),
        "components": {key: round(float(value), 4) for key, value in components.items()},
        "interpretation": "robust_mora_timing_regularity_not_special_mora_correctness",
        "special_mora_included": False,
    }


def weighted_four_dimension_overall(
    *,
    pronunciation: Optional[float],
    rhythm: Optional[float],
    fluency: Optional[float],
    pitch: Optional[float],
) -> Tuple[Optional[int], dict[str, Any]]:
    """Aggregate exactly the four dimensions exposed by the practice UI."""

    values = {
        "pronunciation": pronunciation,
        "rhythm": rhythm,
        "fluency": fluency,
        "pitch": pitch,
    }
    weights = {
        "pronunciation": 0.30,
        "rhythm": 0.20,
        "fluency": 0.25,
        "pitch": 0.25,
    }
    total = 0.0
    denominator = 0.0
    available: list[str] = []
    for name, value in values.items():
        if value is None:
            continue
        total += max(0.0, min(100.0, float(value))) * weights[name]
        denominator += weights[name]
        available.append(name)
    score = _clamp_score(total / denominator) if denominator > 0 else None
    return score, {
        "score_type": "weak_reference_four_dimension_overall_v2",
        "weights": weights,
        "available_dimensions": available,
        "missing_dimensions": [name for name in values if name not in available],
        "renormalized_over_available_dimensions": len(available) < len(values),
    }


def score_coarse_frame_pitch_fallback(frame_f0_hz: Sequence[float]) -> Tuple[Optional[int], dict[str, Any]]:
    """Return a low-confidence pitch number when mora aggregation fails.

    This path is deliberately capped and never used as pitch-accent evidence.
    It exists for otherwise valid full sentences where frame-level pYIN still
    recovered enough F0, but uncertain mora boundaries left too few mora
    medians for the calibrated pitch-v2 model.
    """

    f0 = np.asarray(frame_f0_hz, dtype=float)
    valid = np.isfinite(f0) & (f0 > 0)
    valid_count = int(np.sum(valid))
    coverage = valid_count / max(len(f0), 1)
    if valid_count < 15 or coverage < 0.10:
        return None, {
            "available": False,
            "reason": "insufficient_frame_f0_for_coarse_fallback",
            "frame_f0_coverage": round(float(coverage), 4),
            "valid_frame_count": valid_count,
        }

    logf0 = np.log(f0[valid])
    pitch_range = float(np.percentile(logf0, 90) - np.percentile(logf0, 10))
    range_score = min(1.0, pitch_range / 0.20)
    if pitch_range > 0.95:
        range_score *= max(0.35, 1.0 - (pitch_range - 0.95) / 0.80)

    valid_indices = np.flatnonzero(valid)
    adjacent = np.diff(logf0)
    adjacent_frames = np.diff(valid_indices)
    local = adjacent[adjacent_frames <= 2]
    local_movement = float(np.median(np.abs(local))) if local.size else 0.0
    stability_score = max(0.0, min(1.0, 1.0 - max(0.0, local_movement - 0.10) * 3.0))
    coverage_score = min(1.0, coverage / 0.45)
    naturalness = 0.45 * range_score + 0.30 * stability_score + 0.25 * coverage_score
    score = _clamp_score(np.clip(45.0 + 38.0 * naturalness, 45.0, 78.0))
    return score, {
        "available": True,
        "score": score,
        "score_type": "coarse_frame_pitch_naturalness_fallback",
        "confidence": "low",
        "frame_f0_coverage": round(float(coverage), 4),
        "valid_frame_count": valid_count,
        "pitch_range_log_p90_p10": round(float(pitch_range), 4),
        "local_frame_movement": round(float(local_movement), 4),
        "components": {
            "range": round(float(range_score), 4),
            "stability": round(float(stability_score), 4),
            "coverage": round(float(coverage_score), 4),
        },
        "strict_pitch_accent_correctness": False,
        "interpretation": "low_confidence_frame_level_pitch_naturalness_not_mora_or_accent_correctness",
    }
