"""Audit-only phrase intonation and lexical accent-nucleus candidates."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

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


def compute_phrase_intonation_shadow(result: Mapping[str, Any]) -> Dict[str, Any]:
    contour, valid = _normalized_semitones(result)
    coverage = float(np.mean(valid)) if len(valid) else 0.0
    valid_values = contour[valid]
    transitions = np.diff(valid_values) if len(valid_values) > 1 else np.array([], dtype=float)
    rise = float(np.mean(transitions > 0.5)) if len(transitions) else None
    fall = float(np.mean(transitions < -0.5)) if len(transitions) else None
    final = float(valid_values[-1] - valid_values[-2]) if len(valid_values) > 1 else None
    movement = float(np.mean(np.abs(transitions))) if len(transitions) else None
    score = None
    if movement is not None:
        # Candidate mapping only; never consumed by ProductScorePolicy.
        score = max(0.0, min(100.0, 100.0 - 8.0 * abs(movement - 1.5)))
    return {
        "available": len(valid_values) >= 3,
        "backend": "mora_log_f0_semitone_v1",
        "f0_coverage": coverage,
        "contour_movement_semitones": movement,
        "phrase_rise_ratio": rise,
        "phrase_fall_ratio": fall,
        "final_movement_semitones": final,
        "phrase_intonation_score": score,
        "interpretation": "broad_phrase_intonation_candidate_not_lexical_correctness",
        "user_facing": False,
    }


def _target_nucleus(target: List[str]) -> Optional[int]:
    for index in range(len(target) - 1):
        if target[index] == "H" and target[index + 1] == "L":
            return index + 1
    return None


def compute_accent_nucleus_shadow(result: Mapping[str, Any]) -> Dict[str, Any]:
    contour, valid = _normalized_semitones(result)
    target = [str(value) for value in (result.get("target_pitch") or [])]
    drops = np.where(np.isfinite(contour[:-1]) & np.isfinite(contour[1:]), contour[:-1] - contour[1:], -np.inf)
    predicted = int(np.argmax(drops) + 1) if len(drops) and np.max(drops) > 0 else None
    strength = float(np.max(drops)) if predicted is not None else None
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    source = str(details.get("pitch_target_source") or details.get("reference_source") or "unknown")
    weak_target = source in {"auto_pyopenjtalk", "pyopenjtalk", "unknown"}
    return {
        "available": bool(np.sum(valid) >= 3 and predicted is not None),
        "phrase": result.get("target_text"),
        "target_nucleus": _target_nucleus(target),
        "predicted_nucleus": predicted,
        "local_drop_strength": strength,
        "nucleus_confidence": min(1.0, max(0.0, (strength or 0.0) / 4.0)),
        "target_source": source,
        "weak_target": weak_target,
        "alignment_confidence": reliability.get("alignment"),
        "interpretation": "shadow_candidate_not_learner_facing_pitch_correctness",
        "user_facing": False,
    }
