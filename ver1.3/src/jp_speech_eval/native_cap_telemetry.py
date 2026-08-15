"""Exact reliability-cap A/B from evaluator-native telemetry.

New fixed-reference results contain ``details.reliability_cap_audit`` captured
inside the evaluator before and after the existing reliability caps.  This
module turns that snapshot into the same report shape as the historical replay
auditor, but without rerunning any acoustic model or scorer.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


POLICY_ID = "native_reliability_cap_telemetry_v1"
_SCORE_KEYS = ("pronunciation", "prosody", "fluency", "tone", "total")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _integer(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def has_native_cap_telemetry(result: Mapping[str, Any]) -> bool:
    details = _mapping(result.get("details"))
    audit = _mapping(details.get("reliability_cap_audit"))
    return str(audit.get("schema_version") or "") == "reliability_cap_audit_v1"


def report_from_native_cap_telemetry(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Build an exact cap counterfactual from one evaluator-native snapshot."""
    if not has_native_cap_telemetry(result):
        raise ValueError("result has no reliability_cap_audit_v1 telemetry")
    details = _mapping(result.get("details"))
    audit = _mapping(details.get("reliability_cap_audit"))
    pre = _mapping(audit.get("pre_cap_scores"))
    post_components = _mapping(audit.get("post_component_cap_scores"))
    triggers_raw = _mapping(audit.get("cap_triggers"))
    weights = dict(_mapping(audit.get("aggregate_weights")))

    observed = {
        "pronunciation": _integer(result.get("pronunciation_score")),
        "prosody": _integer(result.get("prosody_score")),
        "fluency": _integer(result.get("fluency_score")),
        "tone": _integer(result.get("tone_score")),
        "total": _integer(result.get("total_score")),
    }
    candidate = {
        "pronunciation": _integer(pre.get("pronunciation")),
        "prosody": _integer(pre.get("prosody")),
        "fluency": _integer(pre.get("fluency")),
        "tone": _integer(pre.get("tone")),
        "total": _integer(audit.get("pre_component_cap_total")),
    }
    expected_post = {
        "pronunciation": _integer(post_components.get("pronunciation")),
        "prosody": _integer(post_components.get("prosody")),
        "fluency": _integer(post_components.get("fluency")),
        "tone": _integer(post_components.get("tone")),
        "pre_overall_cap_total": _integer(audit.get("post_component_cap_total")),
        "total": _integer(audit.get("post_overall_cap_total")),
    }

    component_consistency: Dict[str, Any] = {}
    all_components_match = True
    for key in ("pronunciation", "prosody", "fluency", "tone"):
        matches = observed[key] is not None and expected_post[key] is not None and observed[key] == expected_post[key]
        all_components_match = all_components_match and matches
        component_consistency[key] = {
            "checked": True,
            "consistent": bool(matches),
            "reason": "native_post_cap_snapshot_matches_result" if matches else "native_post_cap_snapshot_mismatch",
            "identifiability": "native_pre_cap_exact" if matches else "native_telemetry_internal_mismatch",
        }
    total_matches = observed["total"] is not None and expected_post["total"] is not None and observed["total"] == expected_post["total"]
    internally_consistent = bool(all_components_match and total_matches)

    delta: Dict[str, Optional[int]] = {}
    for key in _SCORE_KEYS:
        left = candidate.get(key)
        right = observed.get(key)
        delta[key] = None if left is None or right is None else int(left - right)

    triggers = {
        "applicable": True,
        "alignment_equal_fallback": bool(triggers_raw.get("alignment_equal_fallback")),
        "mora_evidence_below_threshold": bool(triggers_raw.get("mora_evidence_below_threshold")),
        "f0_coverage_below_0_50": bool(triggers_raw.get("f0_coverage_below_0_50")),
        "overall_reliability_below_0_75": bool(triggers_raw.get("overall_reliability_below_0_75")),
        "judgement_count": None,
        "judgement_needed": None,
        "f0_coverage": None,
        "overall_reliability": None,
    }
    return {
        "policy_id": POLICY_ID,
        "available": all(value is not None for value in candidate.values()),
        "availability_reason": "native_pre_post_cap_snapshot",
        "same_run": True,
        "observed_legacy_evaluator_scores": observed,
        "candidate_pre_cap_scores_from_current_scorer": candidate,
        "candidate_pre_cap_minus_observed": delta,
        "expected_post_cap_scores_from_replay": expected_post,
        "replay_consistency": {
            **component_consistency,
            "weighted_components_compatible": internally_consistent,
            "total_formula_matches": bool(total_matches),
            "historical_replay_compatible": internally_consistent,
            "native_telemetry": True,
        },
        "counterfactual_trust_level": "native_pre_cap_telemetry_exact" if internally_consistent else "native_telemetry_internal_mismatch",
        "counterfactual_trustworthy": internally_consistent,
        "aggregate_weights": weights,
        "source_wav_available": None,
        "tone_replayed": False,
        "cap_triggers": triggers,
        "product_behavior_changed": False,
        "user_facing": False,
        "interpretation": "exact evaluator-native pre/post reliability-cap snapshot; no scorer replay required",
    }
