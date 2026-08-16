"""Consumer-facing separation of recording analyzability and score evidence.

A clean recording is not the same thing as a well-supported four-dimensional
score.  This module summarises those two questions separately without changing
ProductScore or treating any reliability value as learner ability.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .score_contract import PRODUCT_COMPONENT_WEIGHTS


SCORE_EVIDENCE_QUALITY_SCHEMA = "score_evidence_quality_v1"

# These are evidence-coverage strengths for UI semantics, not probabilities.
_EVIDENCE_STRENGTH = {
    "measured_proxy": 1.0,
    "broad_proxy": 0.60,
    "neutral_prior": 0.0,
    "unavailable": 0.0,
}

_DIMENSION_TO_COMPONENT = {
    "clarity": "clarity",
    "mora_timing": "mora_timing",
    "delivery_fluency": "delivery_fluency",
    "intonation": "intonation",
}


def _clip01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _level(value: float, *, high: float, medium: float) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def build_score_evidence_quality(
    result: Mapping[str, Any],
    score_dimensions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return two independent consumer semantics.

    ``recording_analyzability`` answers whether the waveform could be analysed.
    ``score_evidence`` answers how much of the four-dimensional score is backed
    by measured/broad evidence rather than a neutral placeholder.

    Neither value is a probability that the score is correct.
    """

    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    recording = details.get("recording_quality") if isinstance(details.get("recording_quality"), Mapping) else {}

    recording_score = _clip01(recording.get("score", reliability.get("recording_quality", 1.0)), 1.0)
    endpoint_score = _clip01(reliability.get("endpointing", 1.0), 1.0)
    analyzability_index = 0.70 * recording_score + 0.30 * endpoint_score
    analyzability_level = _level(analyzability_index, high=0.80, medium=0.50)

    weighted = 0.0
    weight_total = float(sum(PRODUCT_COMPONENT_WEIGHTS.values()))
    dimension_rows: Dict[str, Dict[str, Any]] = {}
    measured_count = 0
    broad_count = 0
    neutral_count = 0
    unavailable_count = 0

    for raw in score_dimensions:
        key = str(raw.get("key") or "")
        component_key = _DIMENSION_TO_COMPONENT.get(key)
        if component_key is None:
            continue
        state = str(raw.get("evidence_state") or "unavailable")
        strength = float(_EVIDENCE_STRENGTH.get(state, 0.0))
        base_weight = float(PRODUCT_COMPONENT_WEIGHTS.get(component_key, 0.0))
        weighted += base_weight * strength
        dimension_rows[key] = {
            "evidence_state": state,
            "strength": strength,
            "base_weight": base_weight,
            "confidence": str(raw.get("confidence") or "unknown"),
        }
        if state == "measured_proxy":
            measured_count += 1
        elif state == "broad_proxy":
            broad_count += 1
        elif state == "neutral_prior":
            neutral_count += 1
        else:
            unavailable_count += 1

    coverage = weighted / weight_total if weight_total > 0 else 0.0
    evidence_level = _level(coverage, high=0.75, medium=0.40)

    if neutral_count or unavailable_count:
        caveat = "some_dimensions_are_neutral_or_unavailable"
    elif broad_count:
        caveat = "some_dimensions_use_broad_proxy_evidence"
    else:
        caveat = "all_dimensions_have_measured_proxy_evidence"

    return {
        "schema_version": SCORE_EVIDENCE_QUALITY_SCHEMA,
        "recording_analyzability": {
            "level": analyzability_level,
            "index": round(analyzability_index, 4),
            "recording_quality": round(recording_score, 4),
            "endpointing": round(endpoint_score, 4),
            "interpretation": "can_the_recording_be_analyzed_not_learner_ability",
        },
        "score_evidence": {
            "level": evidence_level,
            "coverage_index": round(coverage, 4),
            "measured_dimension_count": measured_count,
            "broad_dimension_count": broad_count,
            "neutral_prior_dimension_count": neutral_count,
            "unavailable_dimension_count": unavailable_count,
            "dimensions": dimension_rows,
            "caveat": caveat,
            "interpretation": "evidence_coverage_not_probability_score_is_correct",
        },
        "product_score_changed": False,
        "score_mapped": False,
    }
