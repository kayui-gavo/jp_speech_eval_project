"""Audit legacy reliability caps without confusing scorer drift with cap effects.

Production behavior is unchanged.  The fixed-reference evaluator scores an
utterance, applies post-hoc reliability caps, aggregates the capped components,
and finally caps the aggregate when overall reliability is low.

Two audit situations are intentionally separated:

* ``same_run=True``: the result and scorer replay come from the same code/config
  execution context.  The pre-cap replay is suitable for an exact A/B audit.
* ``same_run=False``: historical stored results are replayed with today's code.
  The replay is first passed through the *historical cap equations* and must
  reproduce the stored post-cap scores.  A mismatch is scorer/config drift, not
  a cap effect.  Even a compatible historical sample can be censored when the
  stored score sits exactly on a cap ceiling, so the pre-cap magnitude may be
  unidentifiable from post-cap telemetry alone.

The current C-end ProductScore is a separate semantic layer.  This module audits
legacy evaluator mechanics only and never changes a learner-facing score.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .audio_features import load_audio
from .config import load_scoring_config
from .scoring import score_fluency, score_pronunciation_rhythm, score_prosody, score_tone_simple
from .text_frontend import is_question_sentence
from .vad import trim_to_speech


POLICY_ID = "legacy_reliability_caps_counterfactual_v3"
PRON_ALIGNMENT_CAP = 80
PRON_EVIDENCE_CAP = 60
PROSODY_F0_CAP = 55
OVERALL_RELIABILITY_CAP = 82
_TRIGGER_KEYS = (
    "alignment_equal_fallback",
    "mora_evidence_below_threshold",
    "f0_coverage_below_0_50",
    "overall_reliability_below_0_75",
)
_SCORE_KEYS = ("pronunciation", "prosody", "fluency", "tone", "total")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(round(number))


def _mora_inputs(result: Mapping[str, Any]) -> tuple[list[str], list[tuple[float, float]], list[float]]:
    moras = [str(item) for item in (result.get("moras") or [])]
    rows = result.get("mora_table") if isinstance(result.get("mora_table"), list) else []
    boundaries: list[tuple[float, float]] = []
    f0: list[float] = []
    for row in rows:
        item = _mapping(row)
        start = _number(item.get("start_sec"), 0.0)
        end = _number(item.get("end_sec"), start)
        boundaries.append((start, end))
        value = item.get("f0_hz")
        f0.append(float("nan") if value is None else _number(value, float("nan")))
    if not moras or len(boundaries) != len(moras):
        raise ValueError(
            "counterfactual requires one stored mora_table boundary per target mora; "
            f"got boundaries={len(boundaries)} moras={len(moras)}"
        )
    return moras, boundaries, f0


def _legacy_fixed_evaluator_applicable(result: Mapping[str, Any]) -> bool:
    mode = str(result.get("alignment_mode") or "")
    rows = result.get("mora_table")
    moras = result.get("moras")
    details = _mapping(result.get("details"))
    return bool(
        mode
        and mode != "none"
        and isinstance(rows, list)
        and rows
        and isinstance(moras, list)
        and moras
        and isinstance(details.get("reliability"), Mapping)
    )


def reliability_cap_triggers(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Mirror the current fixed-evaluator post-score cap predicates."""
    applicable = _legacy_fixed_evaluator_applicable(result)
    details = _mapping(result.get("details"))
    reliability = _mapping(details.get("reliability"))
    mora_summary = _mapping(details.get("mora_evidence_summary"))
    moras = result.get("moras") or []
    alignment_mode = str(result.get("alignment_mode") or "")
    judgement_count = int(_number(mora_summary.get("judgement_available_count"), 0.0))
    judgement_needed = max(3, int(len(moras) * 0.55)) if moras else 0
    f0_coverage = _number(reliability.get("f0_coverage"), 0.0)
    overall = _number(reliability.get("overall"), 0.0)
    return {
        "applicable": applicable,
        "alignment_equal_fallback": bool(applicable and alignment_mode.endswith("fallback_equal")),
        "mora_evidence_below_threshold": bool(applicable and judgement_count < judgement_needed),
        "f0_coverage_below_0_50": bool(applicable and f0_coverage < 0.50),
        "overall_reliability_below_0_75": bool(applicable and overall < 0.75),
        "judgement_count": judgement_count if applicable else None,
        "judgement_needed": judgement_needed if applicable else None,
        "f0_coverage": round(f0_coverage, 6) if applicable else None,
        "overall_reliability": round(overall, 6) if applicable else None,
    }


def _aggregate_weights(raw: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "pronunciation": _number(raw.get("pronunciation", raw.get("pronunciation_weight")), 0.35),
        "prosody": _number(raw.get("prosody", raw.get("prosody_weight")), 0.40),
        "fluency": _number(raw.get("fluency", raw.get("fluency_weight")), 0.25),
        "tone": _number(raw.get("tone", raw.get("tone_weight")), 0.0),
    }


def _aggregate_total(scores: Mapping[str, int | float | None], *, aggregate_weights: Mapping[str, Any]) -> int | None:
    weights = _aggregate_weights(aggregate_weights)
    denominator = sum(max(0.0, value) for value in weights.values())
    if denominator <= 0:
        raise ValueError("aggregate score weights must sum to a positive value")
    numerator = 0.0
    for key, weight in weights.items():
        weight = max(0.0, weight)
        if weight <= 0:
            continue
        score = scores.get(key)
        if score is None:
            return None
        numerator += weight * float(score)
    return int(round(max(0.0, min(100.0, numerator / denominator))))


def _apply_component_caps(pre_cap: Mapping[str, int | None], triggers: Mapping[str, Any]) -> Dict[str, int | None]:
    pronunciation = pre_cap.get("pronunciation")
    if pronunciation is not None and triggers.get("alignment_equal_fallback"):
        pronunciation = min(int(pronunciation), PRON_ALIGNMENT_CAP)
    if pronunciation is not None and triggers.get("mora_evidence_below_threshold"):
        pronunciation = min(int(pronunciation), PRON_EVIDENCE_CAP)
    prosody = pre_cap.get("prosody")
    if prosody is not None and triggers.get("f0_coverage_below_0_50"):
        prosody = min(int(prosody), PROSODY_F0_CAP)
    return {
        "pronunciation": None if pronunciation is None else int(pronunciation),
        "prosody": None if prosody is None else int(prosody),
        "fluency": pre_cap.get("fluency"),
        "tone": pre_cap.get("tone"),
    }


def _effective_component_cap(key: str, triggers: Mapping[str, Any]) -> int | None:
    if key == "pronunciation":
        caps: list[int] = []
        if triggers.get("alignment_equal_fallback"):
            caps.append(PRON_ALIGNMENT_CAP)
        if triggers.get("mora_evidence_below_threshold"):
            caps.append(PRON_EVIDENCE_CAP)
        return min(caps) if caps else None
    if key == "prosody" and triggers.get("f0_coverage_below_0_50"):
        return PROSODY_F0_CAP
    return None


def _component_consistency(
    *,
    key: str,
    observed: int | None,
    replayed_pre_cap: int | None,
    expected_post_cap: int | None,
    triggers: Mapping[str, Any],
    same_run: bool,
) -> Dict[str, Any]:
    if observed is None or replayed_pre_cap is None or expected_post_cap is None:
        return {
            "checked": False,
            "consistent": None,
            "reason": "component_replay_unavailable",
            "identifiability": "unavailable",
        }
    consistent = int(observed) == int(expected_post_cap)
    cap = _effective_component_cap(key, triggers)
    if same_run:
        identifiability = "same_run_exact" if consistent else "same_run_internal_mismatch"
    elif not consistent:
        identifiability = "historical_scorer_or_config_drift"
    elif cap is None:
        identifiability = "historical_uncapped_exact"
    elif int(observed) < int(cap):
        identifiability = "historical_cap_nonbinding_exact"
    elif int(observed) == int(cap):
        identifiability = "historical_censored_at_cap"
    else:
        identifiability = "historical_inconsistent_above_cap"
    return {
        "checked": True,
        "consistent": bool(consistent),
        "reason": "replayed_cap_equation_matches_stored_score" if consistent else "replayed_cap_equation_does_not_match_stored_score",
        "effective_cap": cap,
        "observed": int(observed),
        "replayed_pre_cap": int(replayed_pre_cap),
        "expected_post_cap": int(expected_post_cap),
        "candidate_pre_cap_minus_observed": int(replayed_pre_cap - observed),
        "identifiability": identifiability,
    }


def _all_weighted_components_compatible(consistency: Mapping[str, Any], weights: Mapping[str, float]) -> bool:
    for key in ("pronunciation", "prosody", "fluency"):
        if max(0.0, float(weights.get(key, 0.0))) <= 0:
            continue
        item = _mapping(consistency.get(key))
        if item.get("consistent") is not True:
            return False
    # Tone is never reliability-capped.  When no WAV is present, its stored
    # value is intentionally carried through unchanged rather than re-estimated.
    return True


def _historical_counterfactual_identifiable(consistency: Mapping[str, Any], weights: Mapping[str, float]) -> bool:
    for key in ("pronunciation", "prosody", "fluency"):
        if max(0.0, float(weights.get(key, 0.0))) <= 0:
            continue
        ident = str(_mapping(consistency.get(key)).get("identifiability") or "")
        if ident not in {"historical_uncapped_exact", "historical_cap_nonbinding_exact", "same_run_exact"}:
            return False
    return True


def rescore_without_reliability_caps(
    result: Mapping[str, Any],
    *,
    wav_path: str | Path | None = None,
    scoring_config_path: Optional[str | Path] = None,
    sample_rate: int = 16000,
    same_run: bool = False,
) -> Dict[str, Any]:
    """Replay pre-cap scorers and audit whether that replay is interpretable."""
    triggers = reliability_cap_triggers(result)
    if not triggers["applicable"]:
        return {
            "policy_id": POLICY_ID,
            "available": False,
            "availability_reason": "not_legacy_fixed_evaluator_result",
            "cap_triggers": triggers,
            "product_behavior_changed": False,
            "user_facing": False,
        }

    config = load_scoring_config(scoring_config_path)
    details = _mapping(result.get("details"))
    prosody_details = _mapping(details.get("prosody"))
    aggregate_details = _mapping(details.get("aggregate"))
    moras, boundaries, f0_by_mora = _mora_inputs(result)

    pronunciation_score, _pron_fb, _pron_details = score_pronunciation_rhythm(moras, boundaries, config=config)
    target_pattern = [str(item) for item in (result.get("target_pitch") or [])]
    reference_f0 = details.get("reference_f0_by_mora") if isinstance(details.get("reference_f0_by_mora"), list) else None
    target_text = str(result.get("target_text") or "")
    kana = str(result.get("kana") or "")
    accent_phrases = details.get("accent_phrases") if isinstance(details.get("accent_phrases"), list) else None
    prosody_score, _prosody_fb, _prosody_replay = score_prosody(
        moras=moras,
        target_pattern=target_pattern,
        f0_by_mora=f0_by_mora,
        reference_f0_by_mora=reference_f0,
        pitch_target_source=str(prosody_details.get("pitch_target_source") or "heuristic"),
        is_question=is_question_sentence(target_text, kana or None),
        accent_phrases=accent_phrases,
        config=config,
    )
    fluency_score, _fluency_fb, _fluency_details = score_fluency(
        mora_count=len(moras),
        duration=_number(result.get("duration_sec"), 0.0),
        pause_info=_mapping(result.get("pause_info")),
        config=config,
    )

    observed: Dict[str, int | None] = {
        "pronunciation": _optional_int(result.get("pronunciation_score")),
        "prosody": _optional_int(result.get("prosody_score")),
        "fluency": _optional_int(result.get("fluency_score")),
        "tone": _optional_int(result.get("tone_score")),
        "total": _optional_int(result.get("total_score")),
    }

    tone_score: int | None = observed["tone"]
    tone_replayed = False
    wav_exists = False
    if wav_path is not None:
        path = Path(wav_path)
        wav_exists = path.is_file()
        if wav_exists:
            audio = load_audio(str(path), sr=int(sample_rate))
            y_speech, _region = trim_to_speech(audio.y, audio.sr)
            replayed_tone, _tone_fb, _tone_details = score_tone_simple(
                f0_by_mora,
                y_speech,
                _mapping(result.get("pause_info")),
                config=config,
            )
            tone_score = int(replayed_tone)
            tone_replayed = True

    weights_raw = aggregate_details.get("weights") if isinstance(aggregate_details.get("weights"), Mapping) else config.get("aggregate", {})
    weights = _aggregate_weights(_mapping(weights_raw))
    pre_cap: Dict[str, int | None] = {
        "pronunciation": int(pronunciation_score),
        "prosody": int(prosody_score),
        "fluency": int(fluency_score),
        "tone": tone_score,
    }
    expected_post_cap = _apply_component_caps(pre_cap, triggers)
    pre_overall_cap_total = _aggregate_total(expected_post_cap, aggregate_weights=weights)
    expected_stored_total = pre_overall_cap_total
    if expected_stored_total is not None and triggers.get("overall_reliability_below_0_75"):
        expected_stored_total = min(expected_stored_total, OVERALL_RELIABILITY_CAP)

    consistency: Dict[str, Any] = {}
    for key in ("pronunciation", "prosody", "fluency"):
        consistency[key] = _component_consistency(
            key=key,
            observed=observed[key],
            replayed_pre_cap=pre_cap[key],
            expected_post_cap=expected_post_cap[key],
            triggers=triggers,
            same_run=same_run,
        )
    if tone_replayed:
        consistency["tone"] = _component_consistency(
            key="tone",
            observed=observed["tone"],
            replayed_pre_cap=pre_cap["tone"],
            expected_post_cap=pre_cap["tone"],
            triggers=triggers,
            same_run=same_run,
        )
    else:
        consistency["tone"] = {
            "checked": False,
            "consistent": None,
            "reason": "stored_tone_carried_through_unchanged_no_reliability_cap",
            "identifiability": "stored_unchanged_no_cap",
        }

    weighted_components_compatible = _all_weighted_components_compatible(consistency, weights)
    total_formula_matches = bool(
        expected_stored_total is not None
        and observed["total"] is not None
        and int(expected_stored_total) == int(observed["total"])
    )
    historical_replay_compatible = bool(weighted_components_compatible and total_formula_matches)

    candidate_counterfactual = dict(pre_cap)
    candidate_counterfactual["total"] = _aggregate_total(pre_cap, aggregate_weights=weights)
    delta: Dict[str, int | None] = {}
    for key in _SCORE_KEYS:
        left = candidate_counterfactual.get(key)
        right = observed.get(key)
        delta[key] = None if left is None or right is None else int(left - right)

    if same_run and historical_replay_compatible:
        trust_level = "same_run_exact"
        counterfactual_trustworthy = True
    elif not historical_replay_compatible:
        trust_level = "historical_scorer_or_config_drift"
        counterfactual_trustworthy = False
    elif _historical_counterfactual_identifiable(consistency, weights):
        trust_level = "historical_exact_for_uncensored_components"
        counterfactual_trustworthy = True
    else:
        trust_level = "historical_cap_compatible_but_pre_cap_censored"
        counterfactual_trustworthy = False

    return {
        "policy_id": POLICY_ID,
        "available": candidate_counterfactual["total"] is not None,
        "availability_reason": "formula_replay_available" if candidate_counterfactual["total"] is not None else "weighted_component_missing",
        "same_run": bool(same_run),
        "observed_legacy_evaluator_scores": observed,
        "candidate_pre_cap_scores_from_current_scorer": candidate_counterfactual,
        "candidate_pre_cap_minus_observed": delta,
        "expected_post_cap_scores_from_replay": {
            **expected_post_cap,
            "pre_overall_cap_total": pre_overall_cap_total,
            "total": expected_stored_total,
        },
        "replay_consistency": {
            **consistency,
            "weighted_components_compatible": weighted_components_compatible,
            "total_formula_matches": total_formula_matches,
            "historical_replay_compatible": historical_replay_compatible,
        },
        "counterfactual_trust_level": trust_level,
        "counterfactual_trustworthy": counterfactual_trustworthy,
        "aggregate_weights": weights,
        "source_wav_available": wav_exists,
        "tone_replayed": tone_replayed,
        "cap_triggers": triggers,
        "product_behavior_changed": False,
        "user_facing": False,
        "interpretation": (
            "candidate pre-cap replay is a product-neutral audit. Historical deltas are not cap effects "
            "unless cap equations reproduce the stored scores; cap-saturated post-cap telemetry can still "
            "leave the historical pre-cap magnitude censored."
        ),
    }


def summarize_counterfactual_reports(reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize compatibility first; cap-effect deltas only when trustworthy."""
    applicable = [report for report in reports if _mapping(report.get("cap_triggers")).get("applicable")]
    formula_available = [report for report in applicable if bool(report.get("available"))]
    compatible = [report for report in formula_available if bool(report.get("replay_consistency", {}).get("historical_replay_compatible"))]
    trustworthy = [report for report in formula_available if bool(report.get("counterfactual_trustworthy"))]
    drifted = [report for report in formula_available if str(report.get("counterfactual_trust_level")) == "historical_scorer_or_config_drift"]
    censored = [report for report in formula_available if str(report.get("counterfactual_trust_level")) == "historical_cap_compatible_but_pre_cap_censored"]

    trigger_counts = {key: 0 for key in _TRIGGER_KEYS}
    for report in applicable:
        triggers = _mapping(report.get("cap_triggers"))
        for key in trigger_counts:
            trigger_counts[key] += int(bool(triggers.get(key)))

    raw_candidate_deltas: Dict[str, list[int]] = {key: [] for key in _SCORE_KEYS}
    trusted_deltas: Dict[str, list[int]] = {key: [] for key in _SCORE_KEYS}
    for report in formula_available:
        delta = _mapping(report.get("candidate_pre_cap_minus_observed"))
        for key in _SCORE_KEYS:
            if delta.get(key) is not None:
                raw_candidate_deltas[key].append(int(delta[key]))
    for report in trustworthy:
        delta = _mapping(report.get("candidate_pre_cap_minus_observed"))
        for key in _SCORE_KEYS:
            if delta.get(key) is not None:
                trusted_deltas[key].append(int(delta[key]))

    def stats(values: Sequence[int]) -> Dict[str, Any]:
        if not values:
            return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "positive_count": 0}
        return {
            "n": len(values),
            "mean": round(float(mean(values)), 6),
            "median": round(float(median(values)), 6),
            "min": int(min(values)),
            "max": int(max(values)),
            "positive_count": sum(value > 0 for value in values),
        }

    return {
        "policy_id": POLICY_ID,
        "report_count": len(reports),
        "applicable_count": len(applicable),
        "formula_replay_available_count": len(formula_available),
        "historical_replay_compatible_count": len(compatible),
        "historical_scorer_or_config_drift_count": len(drifted),
        "historical_pre_cap_censored_count": len(censored),
        "counterfactual_trustworthy_count": len(trustworthy),
        "non_applicable_count": len(reports) - len(applicable),
        "trigger_counts": trigger_counts,
        "raw_candidate_delta_stats": {key: stats(values) for key, values in raw_candidate_deltas.items()},
        "trusted_counterfactual_delta_stats": {key: stats(values) for key, values in trusted_deltas.items()},
        "decision": "none",
        "note": (
            "historical scorer/config drift and cap censoring are excluded from trusted cap-effect statistics; "
            "fresh same-run A/B is required before removing runtime caps"
        ),
    }
