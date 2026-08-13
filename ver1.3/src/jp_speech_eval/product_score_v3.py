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
    mode = str(alignment_mode or "")
    if mode.endswith("fallback_equal") or "fallback_equal" in mode:
        return {
            "available": False,
            "evidence_source": "synthetic_equal_boundaries",
            "reason": "fallback_equal_has_no_local_timing_evidence",
            "rate_log_ratio": None,
            "warp_slope_cv": None,
            "warp_local_deviation": None,
            "pause_duration_difference": None,
            "pause_position_difference": None,
        }

    ref_duration = float(reference_duration_sec or 0.0)
    user_duration = float(user_duration_sec or 0.0)
    rate_log_ratio = (
        float(np.log(max(user_duration, 1e-6) / max(ref_duration, 1e-6)))
        if user_duration > 0 and ref_duration > 0
        else None
    )
    user = _durations(boundaries)
    ref = _durations(reference_boundaries or [])
    n = min(len(user), len(ref))
    if n < 2 or float(np.sum(user[:n])) <= 0 or float(np.sum(ref[:n])) <= 0:
        return {
            "available": False,
            "evidence_source": "reference_alignment_missing",
            "reason": "insufficient_matched_mora_boundaries",
            "rate_log_ratio": rate_log_ratio,
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
        "rate_log_ratio": rate_log_ratio,
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
    available = {key: item for key, item in dimensions.items() if item.available and item.value is not None}
    denominator = sum(weights[key] for key in available)
    value = None if denominator <= 0 else sum(weights[key] * float(item.value) for key, item in available.items()) / denominator
    confidence = 0.0 if denominator <= 0 else sum(weights[key] * item.confidence for key, item in available.items()) / denominator
    return {
        "value": None if value is None else round(_clip(value), 4),
        "available": bool(available),
        "confidence": round(float(confidence), 4),
        "weights_requested": weights,
        "weights_effective": {key: round(weights[key] / denominator, 4) for key in available} if denominator else {},
        "unavailable_dimensions": [key for key in weights if key not in available],
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
    if ssl_pronunciation and bool(ssl_pronunciation.get("available")):
        ssl_value = ssl_pronunciation.get("candidate_score")
    pronunciation = DimensionEvidence(
        value=None if ssl_value is None else _clip(float(ssl_value)),
        available=ssl_value is not None,
        confidence=min(1.0, max(0.0, float(alignment_confidence))) if ssl_value is not None else 0.0,
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

    rate_log = timing.get("rate_log_ratio")
    pause_ratio = float(pause_info.get("pause_ratio", 0.0) or 0.0)
    pause_count = int(pause_info.get("pause_count", 0) or 0)
    # Separate continuous delivery continuity from rhythm's local warp term.
    fluency_value = _clip(100.0 - 12.0 * abs(float(rate_log or 0.0)) - 115.0 * pause_ratio - 2.5 * pause_count)
    fluency = DimensionEvidence(fluency_value, True, 0.85, "continuous_rate_pause_continuity", "")

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
