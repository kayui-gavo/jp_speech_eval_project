from __future__ import annotations

from typing import Dict, List

import numpy as np

from .phonology import classify_mora_sequence


def mora_structure_features(moras: List[str], speech_duration_sec: float) -> Dict[str, float | int | None]:
    mora_count = len(moras)
    phonology = classify_mora_sequence(moras)
    long_count = sum(1 for ph in phonology if ph.mora_type == "explicit_long_vowel")
    weak_long_count = sum(1 for ph in phonology if ph.mora_type == "vowel_lengthening_candidate")
    sokuon_count = sum(1 for ph in phonology if ph.mora_type == "sokuon")
    nasal_count = sum(1 for ph in phonology if ph.mora_type == "nasal")
    strong_special_count = long_count + sokuon_count + nasal_count
    special_count = strong_special_count + weak_long_count
    duration = max(float(speech_duration_sec), 1e-6)
    return {
        "mora_count": mora_count,
        "long_vowel_count": long_count,
        "weak_long_vowel_candidate_count": weak_long_count,
        "sokuon_count": sokuon_count,
        "nasal_count": nasal_count,
        "special_mora_count": special_count,
        "strong_special_mora_count": strong_special_count,
        "special_mora_density": special_count / max(mora_count, 1),
        "strong_special_mora_density": strong_special_count / max(mora_count, 1),
        "mora_rate": mora_count / duration if mora_count else None,
        "avg_mora_duration_sec": duration / mora_count if mora_count else None,
    }


def f0_structure_features(f0_hz: np.ndarray) -> Dict[str, float | int | None]:
    f0 = np.asarray(f0_hz, dtype=float)
    valid = np.isfinite(f0) & (f0 > 0)
    if int(np.sum(valid)) < 4:
        return {
            "f0_structure_valid": 0,
            "voiced_frame_count": int(np.sum(valid)),
            "normalized_f0_range": None,
            "normalized_f0_slope": None,
            "f0_direction_change_rate": None,
            "f0_rise_ratio": None,
            "f0_fall_ratio": None,
            "f0_flat_ratio": None,
            "final_f0_movement": None,
        }

    log_f0 = np.log(f0[valid])
    z = (log_f0 - float(np.mean(log_f0))) / (float(np.std(log_f0)) + 1e-8)
    diffs = np.diff(z)
    threshold = 0.12
    dirs = np.where(diffs > threshold, 1, np.where(diffs < -threshold, -1, 0))
    dir_pairs = np.sum(dirs[1:] != dirs[:-1]) if dirs.size >= 2 else 0
    x = np.linspace(0.0, 1.0, num=z.size)
    slope = float(np.polyfit(x, z, deg=1)[0]) if z.size >= 2 else 0.0
    third = max(1, z.size // 3)
    final_movement = float(np.mean(z[-third:]) - np.mean(z[:third]))
    return {
        "f0_structure_valid": 1,
        "voiced_frame_count": int(np.sum(valid)),
        "normalized_f0_range": float(np.max(z) - np.min(z)),
        "normalized_f0_slope": slope,
        "f0_direction_change_rate": float(dir_pairs / max(dirs.size - 1, 1)),
        "f0_rise_ratio": float(np.mean(dirs == 1)) if dirs.size else None,
        "f0_fall_ratio": float(np.mean(dirs == -1)) if dirs.size else None,
        "f0_flat_ratio": float(np.mean(dirs == 0)) if dirs.size else None,
        "final_f0_movement": final_movement,
    }


def light_pronunciation_risk_features(
    moras: List[str],
    speech_duration_sec: float,
    voiced_ratio: float,
    pause_ratio: float,
) -> Dict[str, float | int | None]:
    mora_features = mora_structure_features(moras, speech_duration_sec)
    mora_rate = mora_features["mora_rate"]
    avg_mora = mora_features["avg_mora_duration_sec"]
    special_density = float(mora_features["strong_special_mora_density"] or 0.0)
    too_fast_for_special_mora = bool(
        mora_rate is not None
        and mora_rate > 7.5
        and special_density > 0.12
    )
    compressed_mora_risk = bool(avg_mora is not None and avg_mora < 0.115)
    return {
        "too_fast_for_special_mora": int(too_fast_for_special_mora),
        "compressed_mora_risk": int(compressed_mora_risk),
        "low_voicing_risk": int(float(voiced_ratio) < 0.25),
        "high_pause_risk": int(float(pause_ratio) > 0.25),
    }
