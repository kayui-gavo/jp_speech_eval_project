"""Audit-only local acoustic evidence for Japanese special morae."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np


SPECIAL_TYPES = {"ッ": "sokuon", "っ": "sokuon", "ー": "long_vowel", "ン": "moraic_nasal", "ん": "moraic_nasal"}


def _roi_stats(y: np.ndarray, sr: int, start: float, end: float) -> Dict[str, float]:
    lo = max(0, int(round(start * sr)))
    hi = min(len(y), max(lo + 1, int(round(end * sr))))
    roi = np.asarray(y[lo:hi], dtype=np.float32)
    if not roi.size:
        return {
            "duration_sec": 0.0,
            "rms": 0.0,
            "zero_crossing_rate": 0.0,
            "spectral_centroid_hz": 0.0,
            "low_energy_fraction": 0.0,
            "voicing_autocorrelation": 0.0,
        }
    rms = float(np.sqrt(np.mean(roi * roi) + 1e-12))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(roi))))) if len(roi) > 1 else 0.0
    spectrum = np.abs(np.fft.rfft(roi))
    freqs = np.fft.rfftfreq(len(roi), 1.0 / sr)
    centroid = float(np.sum(freqs * spectrum) / max(float(np.sum(spectrum)), 1e-12))
    frame_size = max(16, int(round(0.02 * sr)))
    frame_rms = [
        float(np.sqrt(np.mean(roi[i : i + frame_size] ** 2) + 1e-12))
        for i in range(0, len(roi), frame_size)
        if len(roi[i : i + frame_size])
    ]
    low_energy_fraction = float(np.mean(np.asarray(frame_rms) < max(rms * 0.5, 1e-5))) if frame_rms else 0.0
    lag = max(1, int(round(sr / 200.0)))
    if len(roi) > lag:
        numerator = float(np.dot(roi[:-lag], roi[lag:]))
        denominator = float(np.linalg.norm(roi[:-lag]) * np.linalg.norm(roi[lag:]))
        autocorrelation = numerator / max(denominator, 1e-12)
    else:
        autocorrelation = 0.0
    return {
        "duration_sec": float(len(roi) / sr),
        "rms": rms,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": centroid,
        "low_energy_fraction": low_energy_fraction,
        "voicing_autocorrelation": autocorrelation,
    }


def compute_special_mora_v2_shadow(
    result: Mapping[str, Any],
    waveform: np.ndarray,
    sample_rate: int,
) -> Dict[str, Any]:
    rows = result.get("mora_table") or []
    alignment_mode = str(result.get("alignment_mode") or "")
    alignment_reliable = not alignment_mode.endswith("fallback_equal")
    audio_duration = len(waveform) / max(sample_rate, 1)
    evidence: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        surface = str(row.get("mora") or "")
        special_type = SPECIAL_TYPES.get(surface)
        if not special_type:
            continue
        roi_start = float(row.get("start_sec") or 0.0)
        roi_end = float(row.get("end_sec") or 0.0)
        roi_valid = 0.0 <= roi_start < roi_end <= audio_duration + (1.0 / max(sample_rate, 1))
        stats = _roi_stats(
            waveform,
            sample_rate,
            roi_start,
            roi_end,
        )
        previous_duration = None
        previous_stats = None
        next_stats = None
        if index > 0 and isinstance(rows[index - 1], Mapping):
            previous_duration = max(
                0.0,
                float(rows[index - 1].get("end_sec") or 0.0)
                - float(rows[index - 1].get("start_sec") or 0.0),
            )
            previous_stats = _roi_stats(
                waveform,
                sample_rate,
                float(rows[index - 1].get("start_sec") or 0.0),
                float(rows[index - 1].get("end_sec") or 0.0),
            )
        if index + 1 < len(rows) and isinstance(rows[index + 1], Mapping):
            next_stats = _roi_stats(
                waveform,
                sample_rate,
                float(rows[index + 1].get("start_sec") or 0.0),
                float(rows[index + 1].get("end_sec") or 0.0),
            )
        neighbor_rms = [item["rms"] for item in (previous_stats, next_stats) if item is not None]
        neighbor_centroids = [item["spectral_centroid_hz"] for item in (previous_stats, next_stats) if item is not None]
        evidence.append({
            "type": special_type,
            "mora_index": index,
            "surface_mora": surface,
            "roi_start": roi_start,
            "roi_end": roi_end,
            "roi_valid": roi_valid,
            "alignment_mode": alignment_mode,
            "features": {
                **stats,
                "duration_to_previous_ratio": None if not previous_duration else stats["duration_sec"] / previous_duration,
                "energy_to_neighbor_ratio": None if not neighbor_rms else stats["rms"] / max(float(np.mean(neighbor_rms)), 1e-8),
                "spectral_change_from_neighbors_hz": None if not neighbor_centroids else stats["spectral_centroid_hz"] - float(np.mean(neighbor_centroids)),
                "boundary_displacement_sec": None,
                "local_ssl_reference_distance": None,
            },
            "context": {
                "previous_mora": None if index == 0 else str(rows[index - 1].get("mora") or ""),
                "next_mora": None if index + 1 >= len(rows) else str(rows[index + 1].get("mora") or ""),
            },
            "evidence_confidence": "medium" if roi_valid and alignment_reliable else "low",
            "shadow_decision": (
                "evidence_only"
                if roi_valid and alignment_reliable
                else "unavailable_alignment_or_roi_unreliable"
            ),
            "interpretation": "local_acoustic_evidence_not_phone_correctness",
            "user_facing": False,
        })
    return {
        "available": bool(evidence),
        "decision_available": bool(evidence) and alignment_reliable and all(item["roi_valid"] for item in evidence),
        "alignment_mode": alignment_mode,
        "backend": "numpy_target_local_roi_v2",
        "evidence": evidence,
        "user_facing": False,
    }
