"""Evidence-aware ProductScore v3 candidate (shadow only).

This module deliberately does *not* alter ``display_score`` or the v2 policy.
It records continuous, reference-relative evidence and a candidate aggregate so
that a later A/B decision can be based on real distributions rather than
clipped v2 dimensions.  It has no model dependency and never loads WavLM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DimensionEvidence:
    """One v3 dimension with availability separated from its numeric value."""

    value: Optional[float]
    available: bool
    confidence: float
    source: str
    reason: str = ""
    # Some research evidence (for example an uncalibrated SSL distance) is
    # real evidence but deliberately has no /100 value yet.
    score_mapped: bool = True

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(float(self.confidence), 4)
        payload["value"] = None if self.value is None else round(float(self.value), 4)
        return payload


def _clip(value: float) -> float:
    return float(np.clip(float(value), 0.0, 100.0))


def _durations(boundaries: Sequence[Tuple[float, float]]) -> np.ndarray:
    return np.asarray([max(0.0, float(end) - float(start)) for start, end in boundaries], dtype=float)


def reference_relative_timing_features(
    boundaries: Sequence[Tuple[float, float]],
    reference_boundaries: Optional[Sequence[Tuple[float, float]]],
    *,
    user_duration_sec: float,
    reference_duration_sec: Optional[float],
    pause_info: Mapping[str, Any],
    alignment_mode: str,
) -> Dict[str, Any]:
    """Continuous fixed-reference timing measurements.

    Equal-time fallback boundaries are synthetic.  Returning explicit
    unavailable local measurements prevents their near-zero CV from looking
    like unusually good articulation or rhythm evidence.
    """
    ref_duration = float(reference_duration_sec or 0.0)
    user_duration = float(user_duration_sec or 0.0)
    global_rate_log_ratio = (
        float(np.log(max(user_duration, 1e-6) / max(ref_duration, 1e-6)))
        if user_duration > 0 and ref_duration > 0
        else None
    )
    mode = str(alignment_mode or "")
    if mode.endswith("fallback_equal") or "fallback_equal" in mode:
        return {
            "available": False,
            "evidence_source": "synthetic_equal_boundaries",
            "reason": "fallback_equal_has_no_local_timing_evidence",
            "global_timing_available": global_rate_log_ratio is not None,
            "local_timing_available": False,
            "global_rate_log_ratio": global_rate_log_ratio,
            # Compatibility alias. Consumers must use the explicit global key.
            "rate_log_ratio": global_rate_log_ratio,
            "warp_slope_cv": None,
            "warp_local_deviation": None,
            "pause_duration_difference": None,
            "pause_position_difference": None,
        }

    user = _durations(boundaries)
    ref = _durations(reference_boundaries or [])
    n = min(len(user), len(ref))
    if n < 2 or float(np.sum(user[:n])) <= 0 or float(np.sum(ref[:n])) <= 0:
        return {
            "available": False,
            "evidence_source": "reference_alignment_missing",
            "reason": "insufficient_matched_mora_boundaries",
            "global_timing_available": global_rate_log_ratio is not None,
            "local_timing_available": False,
            "global_rate_log_ratio": global_rate_log_ratio,
            "rate_log_ratio": global_rate_log_ratio,
            "warp_slope_cv": None,
            "warp_local_deviation": None,
            "pause_duration_difference": None,
            "pause_position_difference": None,
        }

    # Normalising each duration vector removes global tempo.  The resulting
    # ratios describe local compression/expansion, not absolute speaking rate.
    user_fraction = user[:n] / np.sum(user[:n])
    ref_fraction = ref[:n] / np.sum(ref[:n])
    slopes = user_fraction / np.maximum(ref_fraction, 1e-8)
    warp_slope_cv = float(np.std(slopes) / max(np.mean(slopes), 1e-8))
    warp_local_deviation = float(np.mean(np.abs(np.log(np.maximum(slopes, 1e-8)))))
    return {
        "available": True,
        "evidence_source": "reference_relative_mora_boundaries",
        "reason": "",
        "global_timing_available": global_rate_log_ratio is not None,
        "local_timing_available": True,
        "global_rate_log_ratio": global_rate_log_ratio,
        "rate_log_ratio": global_rate_log_ratio,
        "warp_slope_cv": warp_slope_cv,
        "warp_local_deviation": warp_local_deviation,
        # Reference pause segmentation is intentionally not invented from
        # mora boundaries.  These become available only when a verified
        # reference pause track is supplied.
        "pause_duration_difference": None,
        "pause_position_difference": None,
        "pause_ratio": float(pause_info.get("pause_ratio", 0.0) or 0.0),
        "pause_count": int(pause_info.get("pause_count", 0) or 0),
        "mean_pause_sec": (
            float(pause_info.get("pause_total", 0.0) or 0.0)
            / max(int(pause_info.get("pause_count", 0) or 0), 1)
        ),
        "speech_run_duration_sec": user_duration - float(pause_info.get("pause_total", 0.0) or 0.0),
    }


def _weighted_candidate(dimensions: Mapping[str, DimensionEvidence]) -> Dict[str, Any]:
    weights = {"pronunciation": 0.45, "rhythm": 0.25, "fluency": 0.20, "intonation": 0.10}
    evidence = {key: item for key, item in dimensions.items() if item.available}
    scored = {key: item for key, item in evidence.items() if item.value is not None and item.score_mapped}
    denominator = sum(weights[key] for key in scored)
    value = None if denominator <= 0 else sum(weights[key] * float(item.value) for key, item in scored.items()) / denominator
    confidence = 0.0 if not evidence else sum(weights[key] * item.confidence for key, item in evidence.items()) / sum(weights[key] for key in evidence)
    coverage = sum(weights[key] for key in evidence)
    names = list(evidence)
    pronunciation = evidence.get("pronunciation")
    other = [key for key in names if key != "pronunciation"]
    if not names:
        scope = "unavailable"
    elif pronunciation and not other:
        scope = "pronunciation_only"
    elif pronunciation:
        scope = "full" if len(names) == len(weights) else "pronunciation_plus_delivery"
    elif names == ["fluency"]:
        scope = "continuity_only"
    else:
        scope = "delivery_prosody"
    diagnostic_eligible = bool(len(names) >= 2 and coverage >= .45)
    overall_eligible = bool(
        pronunciation is not None
        and bool(other)
        and coverage >= .65
        and pronunciation.value is not None
        and pronunciation.score_mapped
    )
    if overall_eligible:
        overall_reason = "eligible"
    elif pronunciation is None:
        overall_reason = "missing_pronunciation_evidence"
    elif pronunciation.value is None or not pronunciation.score_mapped:
        overall_reason = "pronunciation_evidence_not_score_mapped"
    elif not other:
        overall_reason = "missing_independent_dimension"
    else:
        overall_reason = "insufficient_evidence_coverage"
    return {
        "value": None if value is None else round(_clip(value), 4),
        "available": bool(evidence),
        "confidence": round(float(confidence), 4),
        "weights_requested": weights,
        "weights_effective": {key: round(weights[key] / denominator, 4) for key in scored} if denominator else {},
        "unavailable_dimensions": [key for key in weights if key not in evidence],
        "dimensions_available": names,
        "dimensions_score_mapped": list(scored),
        "evidence_coverage": round(float(coverage), 4),
        "score_scope": scope,
        "diagnostic_candidate_eligible": diagnostic_eligible,
        "overall_product_score_candidate_eligible": overall_eligible,
        "overall_product_score_candidate_eligibility_reason": overall_reason,
        # Compatibility alias.  It deliberately follows the stricter overall
        # definition rather than the old two-dimension diagnostic rule.
        "ab_candidate_eligible": overall_eligible,
    }


def build_product_score_v3_candidate(
    *,
    alignment_mode: str,
    boundaries: Sequence[Tuple[float, float]],
    reference_boundaries: Optional[Sequence[Tuple[float, float]]],
    user_duration_sec: float,
    reference_duration_sec: Optional[float],
    pause_info: Mapping[str, Any],
    legacy_pronunciation_score: Optional[float],
    legacy_prosody_score: Optional[float],
    f0_coverage: float,
    alignment_confidence: float,
    ssl_pronunciation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the v3 candidate without altering any v2 score.

    Before WavLM has been run, pronunciation stays unavailable rather than
    silently treating the legacy mora-timing proxy as phone correctness.
    """
    timing = reference_relative_timing_features(
        boundaries,
        reference_boundaries,
        user_duration_sec=user_duration_sec,
        reference_duration_sec=reference_duration_sec,
        pause_info=pause_info,
        alignment_mode=alignment_mode,
    )
    alignment_ok = bool(timing.get("available"))
    ssl_value = None
    ssl_confidence = 0.0
    if ssl_pronunciation and bool(ssl_pronunciation.get("available")):
        ssl_value = ssl_pronunciation.get("candidate_score")
        ssl_confidence = float(ssl_pronunciation.get("ssl_pronunciation_confidence", ssl_pronunciation.get("confidence", 0.0)) or 0.0)
    pronunciation = DimensionEvidence(
        value=None if ssl_value is None else _clip(float(ssl_value)),
        available=ssl_value is not None,
        confidence=min(1.0, max(0.0, ssl_confidence)) if ssl_value is not None else 0.0,
        source="wavlm_multi_reference" if ssl_value is not None else "unavailable_without_ssl_candidate",
        reason="" if ssl_value is not None else "ssl_pronunciation_candidate_not_available",
    )
    if alignment_ok:
        local = float(timing["warp_local_deviation"])
        slope_cv = float(timing["warp_slope_cv"])
        # Global tempo is deliberately a modest term.  Local relative timing
        # carries the primary candidate signal.
        rhythm_value = _clip(100.0 - 16.0 * abs(float(timing.get("rate_log_ratio") or 0.0)) - 72.0 * local - 28.0 * slope_cv)
        rhythm = DimensionEvidence(rhythm_value, True, min(1.0, max(0.0, alignment_confidence)), "reference_relative_warp", "")
    else:
        rhythm = DimensionEvidence(None, False, 0.0, str(timing.get("evidence_source")), str(timing.get("reason")))

    rate_log = timing.get("global_rate_log_ratio")
    pause_ratio = float(pause_info.get("pause_ratio", 0.0) or 0.0)
    pause_count = int(pause_info.get("pause_count", 0) or 0)
    # Separate continuous delivery continuity from rhythm's local warp term.
    rate_penalty = None if rate_log is None else 12.0 * abs(float(rate_log))
    timing["fluency_global_rate_penalty"] = rate_penalty
    fluency_value = _clip(100.0 - (rate_penalty or 0.0) - 115.0 * pause_ratio - 2.5 * pause_count)
    fluency = DimensionEvidence(
        fluency_value,
        True,
        .85 if rate_penalty is not None else .65,
        "continuous_rate_pause_continuity" if rate_penalty is not None else "pause_continuity_without_global_rate",
        "" if rate_penalty is not None else "global_rate_unavailable_rate_component_omitted",
    )

    if f0_coverage >= 0.50 and legacy_prosody_score is not None and alignment_ok:
        intonation = DimensionEvidence(_clip(float(legacy_prosody_score)), True, min(1.0, f0_coverage), "existing_reference_relative_f0_shadow_input", "")
    else:
        intonation = DimensionEvidence(None, False, min(1.0, max(0.0, f0_coverage)), "f0_or_alignment_unavailable", "requires_f0_and_non_synthetic_boundaries")

    dimensions = {
        "pronunciation": pronunciation,
        "rhythm": rhythm,
        "fluency": fluency,
        "intonation": intonation,
    }
    aggregate = _weighted_candidate(dimensions)
    return {
        "version": "product_score_v3_candidate",
        "user_facing": False,
        "candidate_only": True,
        "legacy_pronunciation_timing_proxy": legacy_pronunciation_score,
        "timing_features": timing,
        "dimensions": {key: value.to_dict() for key, value in dimensions.items()},
        "product_score_v3_candidate": aggregate,
    }


def attach_ssl_pronunciation_candidate(
    candidate: Mapping[str, Any],
    ssl_candidate_score: Optional[float],
    *,
    confidence: float,
    source: str = "wavlm_multi_reference",
) -> Dict[str, Any]:
    """Return a copy with SSL pronunciation attached and weights recomputed."""
    out = dict(candidate)
    dimensions = {key: dict(value) for key, value in dict(candidate.get("dimensions", {})).items()}
    dimensions["pronunciation"] = DimensionEvidence(
        None if ssl_candidate_score is None else _clip(float(ssl_candidate_score)),
        ssl_candidate_score is not None,
        min(1.0, max(0.0, float(confidence))) if ssl_candidate_score is not None else 0.0,
        source,
        "" if ssl_candidate_score is not None else "ssl_candidate_unavailable",
    ).to_dict()
    typed = {key: DimensionEvidence(**value) for key, value in dimensions.items()}
    out["dimensions"] = dimensions
    out["product_score_v3_candidate"] = _weighted_candidate(typed)
    return out


def attach_ssl_pronunciation_evidence(
    candidate: Mapping[str, Any],
    *,
    evidence_index: Optional[float],
    ssl_pronunciation_confidence: float,
    reference_count: int,
    reference_dispersion: Optional[float],
    content_verified: bool,
    audio_valid: bool,
    reason: str = "uncalibrated_ssl_evidence",
) -> Dict[str, Any]:
    """Attach global SSL evidence without pretending it is a /100 score.

    Global WavLM comparisons do not use mora boundaries.  The resulting
    evidence may change *scope* telemetry but cannot make an overall numeric
    ProductScore candidate eligible until a separately validated mapping is
    supplied.
    """
    out = dict(candidate)
    dimensions = {key: dict(value) for key, value in dict(candidate.get("dimensions", {})).items()}
    available = (
        evidence_index is not None
        and int(reference_count) >= 3
        and bool(content_verified)
        and bool(audio_valid)
    )
    if not available and reason == "uncalibrated_ssl_evidence":
        if not content_verified:
            reason = "ssl_requires_verified_content"
        elif not audio_valid:
            reason = "ssl_requires_valid_audio"
        elif int(reference_count) < 3:
            reason = "ssl_requires_at_least_three_native_references"
        else:
            reason = "ssl_evidence_index_unavailable"
    dimensions["pronunciation"] = DimensionEvidence(
        value=None,
        available=available,
        confidence=min(1.0, max(0.0, float(ssl_pronunciation_confidence))) if available else 0.0,
        source="wavlm_multi_reference_global",
        reason="" if available else reason,
        score_mapped=False,
    ).to_dict()
    typed = {key: DimensionEvidence(**value) for key, value in dimensions.items()}
    out["dimensions"] = dimensions
    out["ssl_pronunciation_evidence"] = {
        "available": available,
        "ssl_pronunciation_evidence_index": None if evidence_index is None else round(float(evidence_index), 6),
        "ssl_pronunciation_confidence": round(float(dimensions["pronunciation"]["confidence"]), 4),
        "native_reference_count": int(reference_count),
        "native_reference_dispersion": None if reference_dispersion is None else round(float(reference_dispersion), 6),
        "content_verified": bool(content_verified),
        "audio_valid": bool(audio_valid),
        "reason": "" if available else reason,
    }
    out["product_score_v3_candidate"] = _weighted_candidate(typed)
    return out
