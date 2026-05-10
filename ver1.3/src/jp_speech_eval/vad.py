from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import librosa
import numpy as np


@dataclass(frozen=True)
class SpeechRegion:
    raw_duration: float
    speech_start: float
    speech_end: float
    speech_duration: float
    leading_silence: float
    trailing_silence: float
    speech_ratio: float
    start_sample: int
    end_sample: int
    detected: bool

    def to_dict(self) -> Dict[str, float | int | bool]:
        return asdict(self)


def _frame_voicing(
    y: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int,
    fmin: float = 70.0,
    fmax: float = 500.0,
    min_corr: float = 0.30,
) -> np.ndarray:
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    voiced = np.zeros(len(frames), dtype=bool)
    min_lag = max(1, int(sr / fmax))
    max_lag = min(frame_length - 1, int(sr / fmin))
    if max_lag <= min_lag:
        return voiced

    for i, frame in enumerate(frames):
        frame = np.asarray(frame, dtype=float)
        frame = frame - float(np.mean(frame))
        energy = float(np.dot(frame, frame))
        if energy <= 1e-10:
            continue
        frame = frame * np.hanning(frame.size)
        corr = np.correlate(frame, frame, mode="full")[frame.size - 1 :]
        if corr.size <= max_lag or corr[0] <= 1e-10:
            continue
        voiced[i] = bool(np.max(corr[min_lag : max_lag + 1]) / corr[0] >= min_corr)
    return voiced


def _bridge_short_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    if max_gap_frames <= 0 or mask.size == 0:
        return mask
    out = mask.copy()
    false_idx = np.where(~out)[0]
    if false_idx.size == 0:
        return out

    start = None
    for i, value in enumerate(out):
        if not value and start is None:
            start = i
        if (value or i == len(out) - 1) and start is not None:
            end = i if value else i + 1
            left_on = start > 0 and out[start - 1]
            right_on = end < len(out) and out[end]
            if left_on and right_on and end - start <= max_gap_frames:
                out[start:end] = True
            start = None
    return out


def detect_speech_region(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    pad_ms: float = 140.0,
    max_gap_ms: float = 120.0,
    min_speech_ms: float = 120.0,
) -> SpeechRegion:
    """
    Detect the core spoken region in a fixed-duration recording.

    The detector combines an adaptive RMS threshold with a simple autocorrelation
    voicing check. It is intentionally lightweight and deterministic; thresholds
    are engineering endpointing defaults, not pronunciation theory claims.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    raw_duration = float(len(y) / max(sr, 1))
    if y.size == 0 or raw_duration <= 0:
        return SpeechRegion(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, False)

    frame_length = max(1, int(sr * frame_ms / 1000.0))
    hop_length = max(1, int(sr * hop_ms / 1000.0))
    if y.size < frame_length:
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        detected = peak > 1e-4
        end = len(y) if detected else 0
        return SpeechRegion(
            raw_duration=raw_duration,
            speech_start=0.0,
            speech_end=float(end / sr),
            speech_duration=float(end / sr),
            leading_silence=0.0,
            trailing_silence=float((len(y) - end) / sr),
            speech_ratio=float(end / max(len(y), 1)),
            start_sample=0,
            end_sample=end,
            detected=detected,
        )

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=False)[0]
    if rms.size == 0:
        return SpeechRegion(raw_duration, 0.0, 0.0, 0.0, 0.0, raw_duration, 0.0, 0, 0, False)

    peak_rms = float(np.max(rms))
    noise_floor = float(np.percentile(rms, 20))
    adaptive_threshold = max(noise_floor * 2.8, peak_rms * 0.15, 1e-4)
    energy_mask = rms >= adaptive_threshold

    voiced_mask = _frame_voicing(y, sr, frame_length, hop_length)
    voiced_energy_floor = max(noise_floor * 2.0, peak_rms * 0.08, 5e-5)
    speech_mask = energy_mask | (voiced_mask & (rms >= voiced_energy_floor))
    speech_mask = _bridge_short_gaps(speech_mask, int(max_gap_ms / hop_ms))

    idx = np.where(speech_mask)[0]
    if idx.size == 0:
        return SpeechRegion(raw_duration, 0.0, 0.0, 0.0, 0.0, raw_duration, 0.0, 0, 0, False)

    start_frame = int(idx[0])
    end_frame = int(idx[-1])
    pad = int(sr * pad_ms / 1000.0)
    start_sample = max(0, start_frame * hop_length - pad)
    end_sample = min(len(y), end_frame * hop_length + frame_length + pad)

    if end_sample - start_sample < int(sr * min_speech_ms / 1000.0):
        return SpeechRegion(raw_duration, 0.0, 0.0, 0.0, 0.0, raw_duration, 0.0, 0, 0, False)

    speech_start = float(start_sample / sr)
    speech_end = float(end_sample / sr)
    speech_duration = max(0.0, speech_end - speech_start)
    return SpeechRegion(
        raw_duration=raw_duration,
        speech_start=speech_start,
        speech_end=speech_end,
        speech_duration=speech_duration,
        leading_silence=speech_start,
        trailing_silence=max(0.0, raw_duration - speech_end),
        speech_ratio=speech_duration / max(raw_duration, 1e-6),
        start_sample=int(start_sample),
        end_sample=int(end_sample),
        detected=True,
    )


def trim_to_speech(y: np.ndarray, sr: int) -> Tuple[np.ndarray, SpeechRegion]:
    region = detect_speech_region(y, sr)
    if not region.detected or region.end_sample <= region.start_sample:
        return np.asarray(y, dtype=float), region
    return np.asarray(y, dtype=float)[region.start_sample : region.end_sample], region
