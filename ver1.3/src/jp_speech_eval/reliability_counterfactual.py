"""Offline A/B reconstruction for legacy reliability-driven score caps.

Production behavior is intentionally unchanged.  The current evaluator applies
several score caps after raw pronunciation/prosody/fluency evidence has already
been scored.  This module replays the same scoring functions from a completed
result and reports what the scores would have been *without* those post-hoc
reliability caps.

It is an experiment/audit tool, not an alternative product policy.  Promotion
requires held acceptance plus human criterion evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

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


POLICY_ID = "legacy_reliability_caps_counterfactual_v1"


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
    if len(boundaries) != len(moras):
        raise ValueError(
            "counterfactual requires one stored mora_table boundary per target mora; "
            f"got boundaries={len(boundaries)} moras={len(moras)}"
        )
    return moras, boundaries, f0


def reliability_cap_triggers(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Mirror the current evaluator's post-score cap predicates for auditing."""
    details = _mapping(result.get("details"))
    reliability = _mapping(details.get("reliability"))
    mora_summary = _mapping(details.get("mora_evidence_summary"))
    moras = result.get("moras") or []
    alignment_mode = str(result.get("alignment_mode") or "")
    judgement_count = int(_number(mora_summary.get("judgement_available_count"), 0.0))
    judgement_needed = max(3, int(len(moras) * 0.55))
    f0_coverage = _number(reliability.get("f0_coverage"), 0.0)
    overall = _number(reliability.get("overall"), 0.0)
    return {
        "alignment_equal_fallback": alignment_mode.endswith("fallback_equal"),
        "mora_evidence_below_threshold": judgement_count < judgement_needed,
        "f0_coverage_below_0_50": f0_coverage < 0.50,
        "overall_reliability_below_0_75": overall < 0.75,
        "judgement_count": judgement_count,
        "judgement_needed": judgement_needed,
        "f0_coverage": round(f0_coverage, 6),
        "overall_reliability": round(overall, 6),
    }


def _aggregate_total(
    scores: Mapping[str, float],
    *,
    aggregate_weights: Mapping[str, Any],
) -> int:
    weights = {
        "pronunciation": _number(aggregate_weights.get("pronunciation", aggregate_weights.get("pronunciation_weight")), 0.35),
        "prosody": _number(aggregate_weights.get("prosody", aggregate_weights.get("prosody_weight")), 0.40),
        "fluency": _number(aggregate_weights.get("fluency", aggregate_weights.get("fluency_weight")), 0.25),
        "tone": _number(aggregate_weights.get("tone", aggregate_weights.get("tone_weight")), 0.0),
    }
    denominator = sum(max(0.0, value) for value in weights.values())
    if denominator <= 0:
        raise ValueError("aggregate score weights must sum to a positive value")
    value = sum(max(0.0, weights[key]) * float(scores[key]) for key in weights) / denominator
    return int(round(max(0.0, min(100.0, value))))


def rescore_without_reliability_caps(
    result: Mapping[str, Any],
    *,
    wav_path: str | Path,
    scoring_config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Replay raw scorers and compare them with the legacy capped result."""
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

    audio = load_audio(str(wav_path), sr=int(config.get("sample_rate", 16000) or 16000))
    y_speech, _region = trim_to_speech(audio.y, audio.sr)
    tone_score, _tone_fb, _tone_details = score_tone_simple(
        f0_by_mora,
        y_speech,
        _mapping(result.get("pause_info")),
        config=config,
    )

    cap_free = {
        "pronunciation": int(pronunciation_score),
        "prosody": int(prosody_score),
        "fluency": int(fluency_score),
        "tone": int(tone_score),
    }
    weights = aggregate_details.get("weights") if isinstance(aggregate_details.get("weights"), Mapping) else config.get("aggregate", {})
    cap_free["total"] = _aggregate_total(cap_free, aggregate_weights=_mapping(weights))

    observed = {
        "pronunciation": int(_number(result.get("pronunciation_score"), 0.0)),
        "prosody": int(_number(result.get("prosody_score"), 0.0)),
        "fluency": int(_number(result.get("fluency_score"), 0.0)),
        "tone": int(_number(result.get("tone_score"), 0.0)),
        "total": int(_number(result.get("total_score"), 0.0)),
    }
    delta = {key: int(cap_free[key] - observed[key]) for key in cap_free}
    triggers = reliability_cap_triggers(result)
    return {
        "policy_id": POLICY_ID,
        "available": True,
        "observed_legacy_product_scores": observed,
        "counterfactual_without_reliability_caps": cap_free,
        "counterfactual_minus_observed": delta,
        "cap_triggers": triggers,
        "product_behavior_changed": False,
        "user_facing": False,
        "interpretation": (
            "offline exact scorer replay for A/B; a positive delta means the current post-hoc "
            "reliability cap lowered the stored legacy score"
        ),
    }
