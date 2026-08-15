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


def _speech_sample_range(y: np.ndarray, sr: int, speech_region: Optional[object]) -> tuple[int, int]:
    """Map endpointing metadata onto this signal's sample-rate domain.

    The VAD may run on a resampled/normalized analysis copy while recording
    quality runs on the amplitude-preserved native-rate decode. Time-domain
    fields are therefore preferred over sample indices whenever available.
    """

    if speech_region is None or not getattr(speech_region, "detected", False):
        return 0, int(y.size)

    speech_start = getattr(speech_region, "speech_start", None)
    speech_end = getattr(speech_region, "speech_end", None)
    if speech_start is not None and speech_end is not None:
        start = int(round(max(0.0, float(speech_start)) * sr))
        end = int(round(max(0.0, float(speech_end)) * sr))
        return max(0, min(start, int(y.size))), max(0, min(end, int(y.size)))

    start = int(getattr(speech_region, "start_sample", 0))
    end = int(getattr(speech_region, "end_sample", int(y.size)))
    return max(0, min(start, int(y.size))), max(0, min(end, int(y.size)))


def assess_recording_quality(
    y: np.ndarray,
    sr: int,
    speech_region: Optional[object] = None,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> Dict[str, object]:
    """Estimate recording/channel reliability from one amplitude-preserved wav.

    ``load_audio`` historically returns a normalized ndarray. The current loader
    attaches the original decoded waveform as provenance, so this function can
    recover it before converting to a plain ndarray. Callers that pass an
    ordinary ndarray are still supported, but those values are then assumed to
    already be in the recording domain.

    The output remains a quality/reliability gate and must never be interpreted
    as pronunciation correctness.
    """
    raw_y = getattr(y, "raw_recording_y", None)
    raw_sr = getattr(y, "raw_recording_sr", None)
    normalization_gain = getattr(y, "analysis_normalization_gain", None)
    if raw_y is not None and raw_sr is not None:
        y = np.asarray(raw_y, dtype=float).reshape(-1)
        sr = int(raw_sr)
        input_domain = "amplitude_preserved_decode_from_analysis_provenance"
    else:
        y = np.asarray(y, dtype=float).reshape(-1)
        input_domain = "caller_supplied_recording_domain"

    if y.size == 0:
        return {
            "score": 0.0,
            "level": "low",
            "reliability_factor": 0.25,
            "warnings": ["Empty audio."],
            "interpretation": "recording_quality_not_pronunciation",
            "input_domain": input_domain,
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
        start, end = _speech_sample_range(y, sr, speech_region)
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
        "input_domain": input_domain,
        "analysis_normalization_gain": None if normalization_gain is None else round(float(normalization_gain), 6),
        "recording_sample_rate": int(sr),
    }