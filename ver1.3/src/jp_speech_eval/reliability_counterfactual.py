"""Offline A/B reconstruction for legacy reliability-driven score caps.

Production behavior is intentionally unchanged. The fixed-reference evaluator
applies several score caps after raw pronunciation/prosody/fluency evidence has
already been scored. This module replays the same scoring functions from a
completed result and reports what the legacy evaluator would have produced
without those post-hoc reliability caps.

The current C-end ProductScore is a separate semantic layer. Therefore names in
this module deliberately say ``legacy_evaluator`` rather than ``product``.

A source WAV is optional. Pronunciation, prosody and fluency can be replayed
from the stored raw-result evidence. Tone needs waveform energy, but when its
aggregate weight is zero the cap-free *total* remains exactly reconstructible
without a WAV. Missing evidence stays missing; it is never imputed as a learner
penalty.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .audio_features import load_audio
from .config import load_scoring_config
from .scoring import (
    score_fluency,
    score_pronunciation_rhythm,
    score_prosody,
    score_tone_simple,
)
from .text_frontend import is_question_sentence
from .vad import trim_to_speech


POLICY_ID = "legacy_reliability_caps_counterfactual_v2"
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
    """Avoid treating broad/free-mode missing fields as cap-trigger evidence."""
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
    """Mirror fixed-evaluator post-score cap predicates for auditing."""
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


def _aggregate_total(
    scores: Mapping[str, int | float | None],
    *,
    aggregate_weights: Mapping[str, Any],
) -> int | None:
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
    value = numerator / denominator
    return int(round(max(0.0, min(100.0, value))))


def rescore_without_reliability_caps(
    result: Mapping[str, Any],
    *,
    wav_path: str | Path | None = None,
    scoring_config_path: Optional[str | Path] = None,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    """Replay raw scorers and compare them with the capped legacy evaluator.

    ``wav_path=None`` is valid. In that case tone is unavailable, while total
    can still be exact if the stored aggregate gives tone zero weight.
    """
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

    pronunciation_score, _pron_fb, _pron_details = score_pronunciation_rhythm(
        moras,
        boundaries,
        config=config,
    )
    target_pattern = [str(item) for item in (result.get("target_pitch") or [])]
    reference_f0 = details.get("reference_f0_by_mora") if isinstance(details.get("reference_f0_by_mora"), list) else None
    target_text = str(result.get("target_text") or "")
    kana = str(result.get("kana") or "")
    is_question = is_question_sentence(target_text, kana or None)
    accent_phrases = details.get("accent_phrases") if isinstance(details.get("accent_phrases"), list) else None
    prosody_score, _prosody_fb, _prosody_replay = score_prosody(
        moras=moras,
        target_pattern=target_pattern,
        f0_by_mora=f0_by_mora,
        reference_f0_by_mora=reference_f0,
        pitch_target_source=str(prosody_details.get("pitch_target_source") or "heuristic"),
        is_question=is_question,
        accent_phrases=accent_phrases,
        config=config,
    )
    fluency_score, _fluency_fb, _fluency_details = score_fluency(
        mora_count=len(moras),
        duration=_number(result.get("duration_sec"), 0.0),
        pause_info=_mapping(result.get("pause_info")),
        config=config,
    )

    tone_score: int | None = None
    tone_exact = False
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
            tone_exact = True

    weights_raw = (
        aggregate_details.get("weights")
        if isinstance(aggregate_details.get("weights"), Mapping)
        else config.get("aggregate", {})
    )
    weights = _aggregate_weights(_mapping(weights_raw))
    cap_free: Dict[str, int | None] = {
        "pronunciation": int(pronunciation_score),
        "prosody": int(prosody_score),
        "fluency": int(fluency_score),
        "tone": tone_score,
    }
    cap_free["total"] = _aggregate_total(cap_free, aggregate_weights=weights)

    observed: Dict[str, int | None] = {
        "pronunciation": _optional_int(result.get("pronunciation_score")),
        "prosody": _optional_int(result.get("prosody_score")),
        "fluency": _optional_int(result.get("fluency_score")),
        "tone": _optional_int(result.get("tone_score")),
        "total": _optional_int(result.get("total_score")),
    }
    delta: Dict[str, int | None] = {}
    for key in _SCORE_KEYS:
        left = cap_free.get(key)
        right = observed.get(key)
        delta[key] = None if left is None or right is None else int(left - right)

    total_exact = cap_free["total"] is not None
    exactness = {
        "pronunciation": True,
        "prosody": True,
        "fluency": True,
        "tone": tone_exact,
        "total": total_exact,
    }
    return {
        "policy_id": POLICY_ID,
        "available": bool(total_exact),
        "availability_reason": "exact_total_reconstructed" if total_exact else "weighted_component_missing",
        "observed_legacy_evaluator_scores": observed,
        "counterfactual_without_reliability_caps": cap_free,
        "counterfactual_minus_observed": delta,
        "counterfactual_exactness": exactness,
        "aggregate_weights": weights,
        "source_wav_available": wav_exists,
        "cap_triggers": triggers,
        "product_behavior_changed": False,
        "user_facing": False,
        "interpretation": (
            "legacy fixed-evaluator scorer replay for A/B; positive deltas identify post-hoc "
            "reliability-cap effects, not proven ProductScore improvements"
        ),
    }


def summarize_counterfactual_reports(reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize A/B deltas without selecting a winning product policy."""
    applicable = [report for report in reports if _mapping(report.get("cap_triggers")).get("applicable")]
    usable = [report for report in applicable if bool(report.get("available"))]
    dimension_deltas: Dict[str, list[int]] = {key: [] for key in _SCORE_KEYS}
    trigger_counts = {key: 0 for key in _TRIGGER_KEYS}
    for report in applicable:
        triggers = _mapping(report.get("cap_triggers"))
        for key in trigger_counts:
            trigger_counts[key] += int(bool(triggers.get(key)))
    for report in usable:
        delta = _mapping(report.get("counterfactual_minus_observed"))
        for key in dimension_deltas:
            value = delta.get(key)
            if value is not None:
                dimension_deltas[key].append(int(value))

    total_deltas = dimension_deltas["total"]

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
        "usable_exact_total_count": len(usable),
        "unavailable_exact_total_count": len(applicable) - len(usable),
        "non_applicable_count": len(reports) - len(applicable),
        "any_total_delta_count": sum(value != 0 for value in total_deltas),
        "positive_total_delta_count": sum(value > 0 for value in total_deltas),
        "trigger_counts": trigger_counts,
        "delta_stats": {key: stats(values) for key, values in dimension_deltas.items()},
        "decision": "none",
        "note": "descriptive A/B summary only; do not remove caps from these statistics alone",
    }
