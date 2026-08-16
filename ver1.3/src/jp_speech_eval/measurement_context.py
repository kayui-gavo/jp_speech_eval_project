"""Consumer-facing measurement context separate from learner performance.

A single scalar "confidence percentage" is misleading for this product because
recording analyzability, component evidence coverage, reference precision, and
local-detail eligibility are different things. This module exposes those facts
separately for any frontend while keeping raw reliability scalars in debug
telemetry.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


MEASUREMENT_CONTEXT_SCHEMA = "consumer_measurement_context_v1"
DIMENSION_STATES = ("measured_proxy", "broad_proxy", "neutral_prior", "unavailable")
LOCAL_DETAIL_CATEGORIES = {"pitch", "special_mora", "pronunciation", "pronunciation_detail"}


def _clip01(value: Any, default: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _recording_state(result: Mapping[str, Any]) -> str:
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    recording = details.get("recording_quality") if isinstance(details.get("recording_quality"), Mapping) else {}
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    score = _clip01(recording.get("score", reliability.get("recording_quality", 1.0)))
    if score < 0.20:
        return "retry"
    if score < 0.55:
        return "affected"
    if score < 0.75:
        return "usable"
    return "good"


def _dimension_evidence(score_dimensions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = {state: 0 for state in DIMENSION_STATES}
    keys: Dict[str, str] = {}
    for raw in score_dimensions:
        key = str(raw.get("key") or "")
        state = str(raw.get("evidence_state") or "unavailable")
        if state not in counts:
            state = "unavailable"
        counts[state] += 1
        if key:
            keys[key] = state

    total = int(sum(counts.values()))
    evidence_count = int(counts["measured_proxy"] + counts["broad_proxy"])
    if total <= 0:
        coverage_state = "unavailable"
    elif evidence_count == total and counts["neutral_prior"] == 0 and counts["unavailable"] == 0:
        coverage_state = "all_dimensions_have_evidence"
    elif evidence_count > 0:
        coverage_state = "partial_evidence"
    elif counts["neutral_prior"] > 0:
        coverage_state = "reference_values_only"
    else:
        coverage_state = "unavailable"

    return {
        "dimension_count": total,
        "evidence_backed_count": evidence_count,
        "measured_proxy_count": int(counts["measured_proxy"]),
        "broad_proxy_count": int(counts["broad_proxy"]),
        "neutral_prior_count": int(counts["neutral_prior"]),
        "unavailable_count": int(counts["unavailable"]),
        "coverage_state": coverage_state,
        "by_dimension": keys,
    }


def _local_detail_state(
    gate: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> str:
    if str(gate.get("reliability") or "") == "unscorable" or str(gate.get("practice_check_result") or "") == "retry":
        return "unavailable"
    if bool(policy.get("broad_mode")) or bool(policy.get("weak_reference")):
        return "broad_only"
    blocked = {str(item) for item in (gate.get("blocked_categories") or [])}
    if blocked.intersection(LOCAL_DETAIL_CATEGORIES):
        return "limited"
    return "available"


def build_measurement_context(
    result: Mapping[str, Any],
    score_dimensions: Sequence[Mapping[str, Any]],
    *,
    reliability_gate: Mapping[str, Any],
    scoring_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return facts a consumer UI may show without inventing one confidence %.

    Raw reliability scalars intentionally remain outside this public summary.
    They are useful engineering/debug telemetry but are not calibrated
    probabilities that all four public scores are correct.
    """

    evidence = _dimension_evidence(score_dimensions)
    messages = [str(item) for item in (reliability_gate.get("messages") or []) if str(item).strip()]
    return {
        "schema": MEASUREMENT_CONTEXT_SCHEMA,
        "recording_state": _recording_state(result),
        "dimension_evidence": evidence,
        "local_detail_state": _local_detail_state(reliability_gate, scoring_policy),
        "primary_limitation": messages[0] if messages else "",
        "single_confidence_percentage_allowed": False,
        "raw_reliability_numeric_user_facing": False,
        "interpretation": (
            "recording analyzability, dimension evidence coverage, and local-detail eligibility are separate facts; "
            "do not collapse them into one learner-facing confidence percentage"
        ),
    }
