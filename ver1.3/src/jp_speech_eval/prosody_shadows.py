"""Audit-only phrase intonation and lexical accent-nucleus candidates."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


def _normalized_semitones(result: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rows = result.get("mora_table") or []
    f0 = np.array([
        float(row.get("f0_hz")) if isinstance(row, Mapping) and row.get("f0_hz") not in {None, ""} else np.nan
        for row in rows
    ], dtype=float)
    valid = np.isfinite(f0) & (f0 > 0)
    values = np.full_like(f0, np.nan)
    if np.any(valid):
        log_values = 12.0 * np.log2(f0[valid])
        values[valid] = log_values - np.median(log_values)
    return values, valid


def _adjacent_transitions(contour: np.ndarray, valid: np.ndarray) -> list[tuple[int, float]]:
    """Only retain real mora i -> i+1 transitions.

    Compressing valid F0 values first accidentally treats an F0 gap as an
    adjacent transition.  That manufactured a direction change in v1.
    """
    return [
        (i, float(contour[i + 1] - contour[i]))
        for i in range(max(0, len(contour) - 1))
        if bool(valid[i]) and bool(valid[i + 1])
    ]


def _reference_contour(result: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    values = details.get("reference_f0_by_mora") or result.get("reference_f0_by_mora") or []
    f0 = np.asarray([float(x) if x not in {None, ""} else np.nan for x in values], dtype=float)
    valid = np.isfinite(f0) & (f0 > 0)
    contour = np.full_like(f0, np.nan)
    if np.any(valid):
        contour[valid] = 12.0 * np.log2(f0[valid]) - np.median(12.0 * np.log2(f0[valid]))
    return contour, valid


def _corr(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 3 or len(b) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def compute_phrase_intonation_shadow(result: Mapping[str, Any]) -> Dict[str, Any]:
    contour, valid = _normalized_semitones(result)
    coverage = float(np.mean(valid)) if len(valid) else 0.0
    valid_values = contour[valid]
    transitions_with_index = _adjacent_transitions(contour, valid)
    transitions = np.asarray([value for _index, value in transitions_with_index], dtype=float)
    rise = float(np.mean(transitions > 0.5)) if len(transitions) else None
    fall = float(np.mean(transitions < -0.5)) if len(transitions) else None
    final = float(transitions_with_index[-1][1]) if transitions_with_index else None
    movement = float(np.mean(np.abs(transitions))) if len(transitions) else None
    reference, reference_valid = _reference_contour(result)
    common = [i for i in range(min(len(contour), len(reference))) if valid[i] and reference_valid[i]]
    contour_corr = _corr([float(contour[i]) for i in common], [float(reference[i]) for i in common])
    contour_rmse = (
        float(np.sqrt(np.mean([(contour[i] - reference[i]) ** 2 for i in common])))
        if common else None
    )
    user_transitions = {i: value for i, value in transitions_with_index}
    reference_transitions = dict(_adjacent_transitions(reference, reference_valid))
    common_transition_indices = sorted(set(user_transitions) & set(reference_transitions))
    informative_agreement = [
        np.sign(user_transitions[i]) == np.sign(reference_transitions[i])
        for i in common_transition_indices
        if abs(user_transitions[i]) >= 0.25 or abs(reference_transitions[i]) >= 0.25
    ]
    agreement = float(np.mean(informative_agreement)) if informative_agreement else None
    # "Final" must name the same transition in both contours.  Comparing a
    # user transition before an F0 gap with reference's real final transition
    # creates a fabricated difference.
    final_common_index = max(common_transition_indices) if common_transition_indices else None
    final_difference = (
        float(user_transitions[final_common_index] - reference_transitions[final_common_index])
        if final_common_index is not None else None
    )
    return {
        "available": len(valid_values) >= 3,
        "backend": "mora_log_f0_semitone_reference_relative_v2",
        "f0_coverage": coverage,
        "contour_movement_semitones": movement,
        "phrase_rise_ratio": rise,
        "phrase_fall_ratio": fall,
        "final_movement_semitones": final,
        "adjacent_transition_count": len(transitions_with_index),
        "contour_corr": contour_corr,
        "contour_rmse_semitone": contour_rmse,
        "adjacent_transition_agreement": agreement,
        "rise_fall_agreement": agreement,
        "final_movement_difference": final_difference,
        "final_movement_transition_index": final_common_index,
        "phrase_intonation_score": None,
        "interpretation": "reference_relative_raw_shadow_features_not_lexical_correctness_or_score",
        "user_facing": False,
    }


def _target_nucleus(target: List[str]) -> Optional[int]:
    for index in range(len(target) - 1):
        if target[index] == "H" and target[index + 1] == "L":
            return index + 1
    return None


def compute_accent_nucleus_shadow(result: Mapping[str, Any]) -> Dict[str, Any]:
    contour, valid = _normalized_semitones(result)
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    metrics = result.get("prosody_metrics") if isinstance(result.get("prosody_metrics"), Mapping) else {}
    source = str(
        details.get("pitch_target_source")
        or metrics.get("hl_target_source")
        or metrics.get("pitch_target_source")
        or "unknown"
    )
    normalized_source = source.lower()
    strong_sources = {"human_checked", "ojad_checked", "ojad_reviewed", "manual_verified"}
    weak_target = normalized_source not in strong_sources
    phrases = details.get("accent_phrases") if isinstance(details.get("accent_phrases"), list) else []
    if not phrases:
        phrases = [{"start_mora_index": 1, "end_mora_index": len(contour), "accent_position": 0, "moras": result.get("moras") or []}]
    phrase_results = []
    inferred_start = 0
    for phrase_index, phrase in enumerate(phrases):
        phrase_length = len(phrase.get("moras") or [])
        start = max(0, int(phrase.get("start_mora_index", inferred_start + 1) or inferred_start + 1) - 1)
        default_end = start + phrase_length if phrase_length else len(contour)
        end = min(len(contour), int(phrase.get("end_mora_index", default_end) or default_end))
        inferred_start = end
        accent_position = int(phrase.get("accent_position", 0) or 0)
        local_drops = []
        for i in range(start, max(start, end - 1)):
            if valid[i] and valid[i + 1]:
                local_drops.append((i + 1, float(contour[i] - contour[i + 1])))
        ranked = sorted(local_drops, key=lambda item: item[1], reverse=True)
        predicted = ranked[0][0] if ranked and ranked[0][1] > 0 else None
        strongest = ranked[0][1] if ranked else None
        second = ranked[1][1] if len(ranked) > 1 else None
        target_position = (start + accent_position) if accent_position > 0 else None
        coverage = float(np.mean(valid[start:end])) if end > start else 0.0
        margin = None if strongest is None or second is None else float(strongest - second)
        phrase_results.append({
            "phrase_index": phrase_index,
            "start_mora": start + 1,
            "end_mora": end,
            "target_accent_position": accent_position,
            "target_type": "heiban" if accent_position == 0 else "drop_after_accent",
            "predicted_drop_position": predicted,
            "strongest_drop": strongest,
            "second_best_drop": second,
            "drop_margin": margin,
            "f0_coverage": coverage,
            "confidence": min(1.0, coverage * max(0.0, (margin if margin is not None else strongest or 0.0) / 3.0)),
            "target_correctness_candidate": (predicted == target_position) if (not weak_target and target_position is not None) else None,
        })
    return {
        "available": bool(phrase_results and np.sum(valid) >= 3),
        "phrases": phrase_results,
        "target_source": source,
        "weak_target": weak_target,
        "alignment_confidence": reliability.get("alignment"),
        "interpretation": "per_accent_phrase_shadow_candidate_not_learner_facing_pitch_correctness",
        "user_facing": False,
    }
