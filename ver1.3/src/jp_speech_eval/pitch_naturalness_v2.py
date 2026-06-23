from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


FEATURES = (
    "f0_coverage",
    "utterance_f0_range_log",
    "local_pitch_movement",
    "transition_smoothness",
    "flatness_penalty",
    "instability_penalty",
)


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "pitch_naturalness_v2.json"


@lru_cache(maxsize=4)
def load_pitch_naturalness_v2_config(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else default_config_path()
    if not target.exists():
        return {"active": False}
    return json.loads(target.read_text(encoding="utf-8"))


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def predict_continuous_naturalness(
    weak_details: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Optional[float]:
    feature_names = tuple(config.get("features") or FEATURES)
    values = [_finite(weak_details.get(name)) for name in feature_names]
    if any(value is None for value in values):
        return None
    mean = np.asarray(config.get("scaler_mean") or [], dtype=float)
    scale = np.asarray(config.get("scaler_scale") or [], dtype=float)
    coefficients = np.asarray(config.get("coefficients") or [], dtype=float)
    x = np.asarray(values, dtype=float)
    if not (x.size == mean.size == scale.size == coefficients.size):
        return None
    scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)
    score = float(config.get("intercept", 0.0)) + float(np.dot((x - mean) / scale, coefficients))
    lower = float(config.get("output_floor", 10.0))
    upper = float(config.get("output_ceiling", 98.0))
    return float(np.clip(score, lower, upper))


def _normalized_log_f0(f0_by_mora: Sequence[float]) -> np.ndarray:
    f0 = np.asarray(f0_by_mora, dtype=float)
    valid = np.isfinite(f0) & (f0 > 0)
    out = np.full(f0.shape, np.nan, dtype=float)
    if int(np.sum(valid)) < 3:
        return out
    values = np.log(f0[valid])
    center = float(np.median(values))
    scale = float(np.median(np.abs(values - center))) * 1.4826
    if scale < 1e-4:
        scale = float(np.std(values))
    if scale < 1e-4:
        scale = 1.0
    out[valid] = (values - center) / scale
    return out


def soft_accent_hint_score(
    f0_by_mora: Sequence[float],
    target_pattern: Sequence[str],
    accent_phrases: Sequence[Mapping[str, Any]] | None = None,
    *,
    is_question: bool = False,
) -> tuple[Optional[float], dict[str, Any]]:
    """Score only salient accent events, not a complete sentence F0 curve.

    The final boundary region is excluded because sentence type, focus and
    affect can legitimately change it. A +/-1 mora window allows acoustic and
    mora-alignment timing variation. This remains a soft Tokyo-style hint.
    """
    z = _normalized_log_f0(f0_by_mora)
    n = min(len(z), len(target_pattern))
    if n < 4:
        return None, {"available": False, "reason": "too_few_moras"}
    slopes = np.diff(z[:n])
    excluded_final_moras = 3 if is_question else 2
    last_transition = max(0, n - excluded_final_moras - 1)

    spans: list[tuple[int, int, int]] = []
    offset = 0
    for phrase in accent_phrases or []:
        length = len(phrase.get("moras") or [])
        if length <= 0:
            continue
        spans.append((offset, min(n, offset + length), int(phrase.get("accent_position", 0) or 0)))
        offset += length
    if not spans:
        spans = [(0, n, 0)]

    events: list[dict[str, Any]] = []
    for start, end, accent_position in spans:
        if end - start >= 2 and start < last_transition:
            first = str(target_pattern[start]).upper()
            second = str(target_pattern[start + 1]).upper()
            if first == "L" and second == "H":
                events.append({"index": start, "direction": "rise", "weight": 0.65, "role": "phrase_initial_rise", "start": start, "end": end})
        if accent_position > 0:
            index = start + accent_position - 1
            if start <= index < end - 1 and index < last_transition:
                events.append({"index": index, "direction": "drop", "weight": 1.0, "role": "accent_nucleus_drop", "start": start, "end": end})

    if not events:
        return None, {
            "available": False,
            "reason": "no_salient_nonfinal_accent_event",
            "excluded_final_moras": excluded_final_moras,
        }

    weighted = 0.0
    denom = 0.0
    event_rows: list[dict[str, Any]] = []
    for event in events:
        index = int(event["index"])
        candidates = [
            i for i in range(max(int(event["start"]), index - 1), min(int(event["end"]) - 1, index + 2, last_transition))
            if 0 <= i < len(slopes) and np.isfinite(slopes[i])
        ]
        if not candidates:
            continue
        directional = [float(slopes[i]) if event["direction"] == "rise" else -float(slopes[i]) for i in candidates]
        evidence = max(directional)
        agreement = float(np.clip((evidence + 0.05) / 0.55, 0.0, 1.0))
        weight = float(event["weight"])
        weighted += agreement * weight
        denom += weight
        event_rows.append({
            "expected_index": index,
            "direction": event["direction"],
            "role": event["role"],
            "best_directional_slope": round(evidence, 4),
            "agreement": round(agreement, 4),
        })
    if denom <= 0:
        return None, {"available": False, "reason": "accent_events_have_no_f0"}
    agreement = weighted / denom
    # Keep an automatic symbolic target deliberately soft: even zero event
    # agreement is not equivalent to an incorrect utterance.
    score = 55.0 + 40.0 * agreement
    return round(score, 4), {
        "available": True,
        "score": round(score, 4),
        "event_agreement": round(agreement, 4),
        "event_count": len(event_rows),
        "events": event_rows,
        "lag_tolerance_mora": 1,
        "excluded_final_moras": excluded_final_moras,
        "interpretation": "soft_tokyo_accent_hint_not_strict_correctness",
    }


def combine_naturalness_with_hint(naturalness: float, hint: Optional[float], weight: float) -> float:
    """Use the accent hint only as a bounded mismatch penalty.

    Automatic symbolic targets must never boost a flat or unstable contour.
    They can only reduce an otherwise natural score when the hint is lower.
    """
    if hint is None or weight <= 0:
        return float(naturalness)
    return float(naturalness) - float(weight) * max(0.0, float(naturalness) - float(hint))


def score_pitch_naturalness_v2(
    f0_by_mora: Sequence[float],
    weak_details: Mapping[str, Any],
    target_pattern: Sequence[str],
    accent_phrases: Sequence[Mapping[str, Any]] | None,
    *,
    pitch_target_source: str,
    is_question: bool,
    config: Mapping[str, Any],
) -> tuple[Optional[int], dict[str, Any]]:
    if weak_details.get("available") is False:
        return None, {"available": False, "reason": weak_details.get("unavailable_reason") or "weak_features_unavailable"}
    naturalness = predict_continuous_naturalness(weak_details, config)
    if naturalness is None:
        return None, {"available": False, "reason": "calibrated_features_unavailable"}
    hint, hint_details = soft_accent_hint_score(
        f0_by_mora,
        target_pattern,
        accent_phrases,
        is_question=is_question,
    )
    source = str(pitch_target_source or "")
    verified_sources = {"ojad_checked", "human_checked", "verified_manual", "reference_audio_f0_cache"}
    auto_sources = {"openjtalk_accent_phrase_chain", "auto_pyopenjtalk"}
    if hint is None:
        hint_weight = 0.0
    elif source in verified_sources:
        hint_weight = float(config.get("verified_accent_hint_weight", 0.20))
    elif source in auto_sources:
        hint_weight = float(config.get("automatic_accent_hint_weight", 0.08))
    else:
        hint_weight = 0.0
    combined = combine_naturalness_with_hint(naturalness, hint, hint_weight)
    combined = float(np.clip(combined, float(config.get("output_floor", 10.0)), float(config.get("output_ceiling", 98.0))))
    return int(round(combined)), {
        "available": True,
        "score_type": "weak_reference_pitch_naturalness_v2",
        "score": int(round(combined)),
        "data_calibrated_naturalness_score": round(naturalness, 4),
        "accent_hint_score": hint,
        "accent_hint_weight": round(hint_weight, 4),
        "accent_hint_source": source,
        "accent_hint_details": hint_details,
        "strict_pitch_accent_correctness": False,
        "interpretation": "continuous_native_likeness_with_soft_accent_hint",
        "accent_hint_policy": "mismatch_penalty_only_never_boosts_naturalness",
    }
