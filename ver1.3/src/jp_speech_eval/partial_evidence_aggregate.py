"""Shadow-only aggregate that does not treat neutral priors as measurements.

The current product contract deliberately keeps a neutral numeric anchor when a
component lacks validated evidence. That is safe for display continuity, but a
neutral placeholder must not silently receive the same aggregate weight as a
measured component when we study future score semantics.

This module therefore builds an *experimental* headline candidate from the
already-authorised product components and their evidence states. It does not
promote any new acoustic model, does not alter ProductScore, and is not
user-facing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from .score_contract import DISPLAY_ANCHOR, PRODUCT_COMPONENT_WEIGHTS, apply_display_transform


PARTIAL_EVIDENCE_AGGREGATE_SCHEMA = "partial_evidence_aggregate_shadow_v1"
PARTIAL_EVIDENCE_AGGREGATE_POLICY = "neutral_prior_excluded_coverage_shrunk_v1"

# These are experimental aggregation strengths, not confidence probabilities.
# A measured proxy keeps its contract weight. A broad proxy contributes less.
# Neutral/unavailable values carry zero measurement weight.
EVIDENCE_STATE_STRENGTH: Dict[str, float] = {
    "measured_proxy": 1.0,
    "broad_proxy": 0.75,
    "neutral_prior": 0.0,
    "unavailable": 0.0,
}


def _finite_score(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(0.0, min(100.0, number))


def build_partial_evidence_aggregate_candidate(
    component_scores: Mapping[str, Mapping[str, Any]],
    *,
    anchor: float = DISPLAY_ANCHOR,
) -> Dict[str, Any]:
    """Build a future headline-score candidate from real evidence only.

    Procedure:

    1. Start from the frozen ProductScore component weights.
    2. Multiply each weight by an experimental evidence-state strength.
    3. Exclude ``neutral_prior`` and ``unavailable`` from the evidence mean.
    4. Shrink the evidence-only mean toward the neutral anchor by
       ``sqrt(effective_coverage)`` so sparse evidence cannot look as certain as
       a fully observed four-dimensional score.
    5. Apply the existing display transform only for apples-to-apples telemetry.

    The square-root shrinkage is an intentionally conservative C-end candidate,
    not a calibrated statistical confidence interval. It must be tested on held
    real audio before any score-contract change.
    """

    base_weight_total = float(sum(PRODUCT_COMPONENT_WEIGHTS.values()))
    weighted_sum = 0.0
    effective_weight_sum = 0.0
    rows: Dict[str, Dict[str, Any]] = {}

    for key, base_weight in PRODUCT_COMPONENT_WEIGHTS.items():
        raw = component_scores.get(key) if isinstance(component_scores.get(key), Mapping) else {}
        state = str(raw.get("evidence_state") or "unavailable")
        value = _finite_score(raw.get("value"))
        state_strength = float(EVIDENCE_STATE_STRENGTH.get(state, 0.0))
        effective_weight = float(base_weight) * state_strength if value is not None else 0.0
        included = effective_weight > 0.0
        if included:
            weighted_sum += effective_weight * float(value)
            effective_weight_sum += effective_weight
        rows[key] = {
            "value": value,
            "evidence_state": state,
            "base_weight": round(float(base_weight), 6),
            "state_strength": round(state_strength, 6),
            "effective_weight": round(effective_weight, 6),
            "included": included,
            "exclusion_reason": "" if included else (
                "neutral_prior_not_measurement"
                if state == "neutral_prior"
                else "evidence_unavailable_or_not_eligible"
            ),
        }

    effective_coverage = (
        effective_weight_sum / base_weight_total if base_weight_total > 0.0 else 0.0
    )
    available_count = sum(1 for item in rows.values() if item["included"])
    neutral_prior_count = sum(
        1 for item in rows.values() if item["evidence_state"] == "neutral_prior"
    )

    if effective_weight_sum <= 0.0:
        return {
            "schema": PARTIAL_EVIDENCE_AGGREGATE_SCHEMA,
            "policy_id": PARTIAL_EVIDENCE_AGGREGATE_POLICY,
            "available": False,
            "candidate_raw_score": None,
            "candidate_display_score": None,
            "fallback_numeric_anchor": round(float(anchor), 4),
            "evidence_only_mean": None,
            "effective_coverage": 0.0,
            "coverage_shrinkage": 0.0,
            "available_component_count": 0,
            "neutral_prior_count": neutral_prior_count,
            "components": rows,
            "score_mapped": False,
            "product_calibrated": False,
            "user_facing": False,
            "product_score_changed": False,
            "interpretation": "no_measured_or_broad_component_evidence",
        }

    evidence_mean = weighted_sum / effective_weight_sum
    shrinkage = math.sqrt(max(0.0, min(1.0, effective_coverage)))
    candidate_raw = float(anchor) + shrinkage * (float(evidence_mean) - float(anchor))
    candidate_display = apply_display_transform(candidate_raw)

    return {
        "schema": PARTIAL_EVIDENCE_AGGREGATE_SCHEMA,
        "policy_id": PARTIAL_EVIDENCE_AGGREGATE_POLICY,
        "available": True,
        "candidate_raw_score": round(candidate_raw, 4),
        "candidate_display_score": round(float(candidate_display), 4),
        "fallback_numeric_anchor": round(float(anchor), 4),
        "evidence_only_mean": round(float(evidence_mean), 4),
        "effective_coverage": round(float(effective_coverage), 6),
        "coverage_shrinkage": round(float(shrinkage), 6),
        "available_component_count": int(available_count),
        "neutral_prior_count": int(neutral_prior_count),
        "components": rows,
        "score_mapped": False,
        "product_calibrated": False,
        "user_facing": False,
        "product_score_changed": False,
        "interpretation": (
            "shadow headline candidate; neutral priors excluded from measurement weight; "
            "sparse evidence shrunk toward neutral anchor"
        ),
        "promotion_gate": (
            "held real-audio dispersion/channel-safety plus construct-matched human validation "
            "before any score-contract change"
        ),
    }
