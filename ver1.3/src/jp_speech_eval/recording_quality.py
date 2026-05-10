from __future__ import annotations

from typing import Dict, List, Optional

import librosa
import numpy as np


def _safe_db_ratio(signal: float, noise: float) -> float:
    return float(20.0 * np.log10((signal + 1e-8) / (noise + 1e-8)))


def _level(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def assess_recording_quality(
    y: np.ndarray,
    sr: int,
    speech_region: Optional[object] = None,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> Dict[str, object]:
    """Estimate recording/channel reliability from one wav.

    The output is a quality gate. It should reduce confidence when the input
    is noisy or clipped, but it should not be interpreted as pronunciation
    correctness.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size == 0:
        return {
            "score": 0.0,
            "level": "low",
            "reliability_factor": 0.25,
            "warnings": ["Empty audio."],
            "interpretation": "recording_quality_not_pronunciation",
        }

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size == 0:
        rms = np.asarray([float(np.sqrt(np.mean(y * y)))])

    peak = float(np.max(np.abs(y)))
    clipping_ratio = float(np.mean(np.abs(y) >= 0.995))
    rms_mean = float(np.mean(rms))
    rms_p10 = float(np.percentile(rms, 10))
    rms_p20 = float(np.percentile(rms, 20))
    rms_p95 = float(np.percentile(rms, 95))

    noise_samples: List[np.ndarray] = []
    if speech_region is not None and getattr(speech_region, "detected", False):
        start = int(getattr(speech_region, "start_sample", 0))
        end = int(getattr(speech_region, "end_sample", 0))
        if start > int(0.05 * sr):
            noise_samples.append(y[:start])
        if end < y.size - int(0.05 * sr):
            noise_samples.append(y[end:])
    if noise_samples:
        noise_y = np.concatenate(noise_samples)
        noise_rms = float(np.sqrt(np.mean(noise_y * noise_y))) if noise_y.size else rms_p10
    else:
        noise_rms = rms_p10

    signal_rms = max(rms_p95, rms_mean)
    snr_db = _safe_db_ratio(signal_rms, noise_rms)
    dynamic_range_db = _safe_db_ratio(max(rms_p95, 1e-8), max(rms_p20, 1e-8))

    score = 1.0
    warnings: List[str] = []
    if signal_rms < 0.015:
        score *= 0.75
        warnings.append("Low speech level; pronunciation feedback should be conservative.")
    if snr_db < 8.0:
        score *= 0.55
        warnings.append("Low estimated SNR; background noise may affect F0/alignment.")
    elif snr_db < 14.0:
        score *= 0.78
        warnings.append("Moderate background noise; detailed mora feedback may be less stable.")
    if clipping_ratio > 0.01:
        score *= 0.60
        warnings.append("Possible clipping; reduce microphone gain.")
    if dynamic_range_db < 4.0 and signal_rms > 0.02:
        score *= 0.82
        warnings.append("Narrow dynamic range; compression/noise suppression may affect acoustic cues.")

    score = float(max(0.0, min(1.0, score)))
    return {
        "score": round(score, 4),
        "level": _level(score),
        "reliability_factor": round(max(0.35, score), 4),
        "peak": peak,
        "rms_mean": rms_mean,
        "noise_rms": noise_rms,
        "snr_db": snr_db,
        "dynamic_range_db": dynamic_range_db,
        "clipping_ratio": clipping_ratio,
        "warnings": warnings,
        "interpretation": "recording_quality_not_pronunciation",
    }
