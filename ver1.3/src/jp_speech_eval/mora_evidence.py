from __future__ import annotations

from typing import Dict, List, Tuple

import librosa
import numpy as np

from .phonology import classify_mora_sequence


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _duration_expectedness(mora: str, duration: float, avg_duration: float) -> float:
    if avg_duration <= 0:
        return 0.0
    ratio = duration / avg_duration
    if mora == "ー":
        return _clamp01(1.0 - abs(ratio - 1.05) / 0.75)
    if mora == "ッ":
        return _clamp01(1.0 - abs(ratio - 0.75) / 0.70)
    if mora == "ン":
        return _clamp01(1.0 - abs(ratio - 0.90) / 0.75)
    return _clamp01(1.0 - abs(ratio - 1.00) / 0.85)


def build_mora_evidence(
    moras: List[str],
    boundaries: List[Tuple[float, float]],
    f0_times: np.ndarray,
    f0_hz: np.ndarray,
    y_speech: np.ndarray,
    sr: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Build per-mora evidence gates for reference-based scoring.

    This layer answers "is there enough acoustic evidence to judge this mora?"
    It does not decide whether the mora was pronounced correctly.
    """
    y = np.asarray(y_speech, dtype=float).reshape(-1)
    f0_times = np.asarray(f0_times, dtype=float)
    f0_hz = np.asarray(f0_hz, dtype=float)
    durations = np.asarray([max(0.0, e - s) for s, e in boundaries], dtype=float)
    avg_duration = float(np.mean(durations)) if durations.size else 0.0

    hop = 256
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0] if y.size else np.asarray([])
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop) if rms.size else np.asarray([])
    energy_threshold = max(float(np.percentile(rms, 25)), float(np.max(rms)) * 0.10, 1e-6) if rms.size else 1e-6

    phonology = classify_mora_sequence(moras)
    rows: List[Dict[str, object]] = []
    for i, mora in enumerate(moras):
        ph = phonology[i] if i < len(phonology) else None
        start, end = boundaries[i] if i < len(boundaries) else (0.0, 0.0)
        duration = max(0.0, float(end - start))

        energy_mask = (rms_times >= start) & (rms_times <= end)
        if np.any(energy_mask):
            energy_coverage = float(np.mean(rms[energy_mask] >= energy_threshold))
            energy_mean = float(np.mean(rms[energy_mask]))
        else:
            energy_coverage = 0.0
            energy_mean = 0.0

        f0_mask = (f0_times >= start) & (f0_times <= end)
        if np.any(f0_mask):
            f0_coverage = float(np.mean(np.isfinite(f0_hz[f0_mask]) & (f0_hz[f0_mask] > 0)))
        else:
            f0_coverage = 0.0

        expectedness = _duration_expectedness(mora, duration, avg_duration)
        too_short = bool(duration < max(0.055, 0.45 * avg_duration))
        too_long = bool(avg_duration > 0 and duration > 2.4 * avg_duration)
        boundary_confidence = _clamp01(0.55 * expectedness + 0.30 * energy_coverage + 0.15 * min(1.0, duration / max(avg_duration, 1e-6)))
        judgement_available = bool(boundary_confidence >= 0.45 and energy_coverage >= 0.20 and not too_short and not too_long)
        prosody_available = bool(judgement_available and f0_coverage >= 0.25)

        warnings: List[str] = []
        if too_short:
            warnings.append("too_short_for_stable_mora_judgement")
        if too_long:
            warnings.append("too_long_or_boundary_may_include_neighbor")
        if energy_coverage < 0.20:
            warnings.append("low_energy_coverage")
        if f0_coverage < 0.25:
            warnings.append("low_f0_coverage_for_prosody")

        rows.append({
            "index": i + 1,
            "mora": mora,
            "special_type": ph.mora_type if ph else "normal",
            "special_strength": ph.strength if ph else "none",
            "duration_role": ph.duration_role if ph else "plain_mora",
            "vowel": ph.vowel if ph else None,
            "start_sec": round(float(start), 4),
            "end_sec": round(float(end), 4),
            "duration_sec": round(float(duration), 4),
            "duration_expectedness": round(float(expectedness), 4),
            "energy_coverage": round(float(energy_coverage), 4),
            "energy_mean": energy_mean,
            "f0_coverage": round(float(f0_coverage), 4),
            "boundary_confidence": round(float(boundary_confidence), 4),
            "judgement_available": judgement_available,
            "prosody_available": prosody_available,
            "warnings": warnings,
        })

    reliable = [r for r in rows if bool(r["judgement_available"])]
    prosody = [r for r in rows if bool(r["prosody_available"])]
    special = [r for r in rows if r["special_type"] != "normal"]
    strong_special = [r for r in special if r.get("special_strength") == "strong"]
    weak_special = [r for r in special if r.get("special_strength") == "weak"]
    low = [r for r in rows if not bool(r["judgement_available"])]
    summary = {
        "mora_count": len(rows),
        "judgement_available_count": len(reliable),
        "prosody_available_count": len(prosody),
        "special_mora_count": len(special),
        "strong_special_mora_count": len(strong_special),
        "weak_special_mora_count": len(weak_special),
        "special_mora_judgement_available_count": sum(1 for r in special if bool(r["judgement_available"])),
        "strong_special_mora_judgement_available_count": sum(1 for r in strong_special if bool(r["judgement_available"])),
        "weak_special_mora_judgement_available_count": sum(1 for r in weak_special if bool(r["judgement_available"])),
        "phonology_note": "weak vowel_lengthening_candidate labels are diagnostics, not hard phoneme correctness",
        "mean_boundary_confidence": round(float(np.mean([float(r["boundary_confidence"]) for r in rows])) if rows else 0.0, 4),
        "mean_energy_coverage": round(float(np.mean([float(r["energy_coverage"]) for r in rows])) if rows else 0.0, 4),
        "mean_f0_coverage": round(float(np.mean([float(r["f0_coverage"]) for r in rows])) if rows else 0.0, 4),
        "low_evidence_mora_indices": [int(r["index"]) for r in low],
        "interpretation": "evidence_gate_not_pronunciation_correctness",
    }
    return rows, summary
