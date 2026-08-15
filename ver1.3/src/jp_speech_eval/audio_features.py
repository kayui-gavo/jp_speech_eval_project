from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import librosa
import numpy as np


class AnalysisSignal(np.ndarray):
    """Normalized analysis ndarray carrying non-scoring recording provenance.

    This ndarray subclass is a compatibility bridge for legacy callers that pass
    ``audio.y`` directly into recording-quality code. Numerical operations see a
    normal ndarray; recording-quality can recover the amplitude-preserved decode
    before converting the input with ``np.asarray``. New code should prefer
    :func:`load_audio_views` instead of depending on these attributes.
    """

    raw_recording_y: np.ndarray
    raw_recording_sr: int
    analysis_normalization_gain: float


def _analysis_signal(
    y: np.ndarray,
    *,
    raw_y: np.ndarray,
    raw_sr: int,
    normalization_gain: float,
) -> AnalysisSignal:
    signal = np.asarray(y, dtype=np.float64).view(AnalysisSignal)
    signal.raw_recording_y = np.asarray(raw_y, dtype=np.float64)
    signal.raw_recording_sr = int(raw_sr)
    signal.analysis_normalization_gain = float(normalization_gain)
    return signal


@dataclass(frozen=True)
class AudioData:
    y: np.ndarray
    sr: int
    duration: float


@dataclass(frozen=True)
class AudioViews:
    """Keep recording diagnostics separate from normalized analysis audio.

    ``raw_y`` is the amplitude-preserved mono decode at the file's native sample
    rate. It is intended for channel/recording-quality diagnostics such as
    clipping and input level. ``analysis`` is resampled and peak-normalized for
    VAD, ASR, alignment, F0 and representation extraction.

    Keeping both views prevents analysis normalization from erasing information
    that is meaningful only in the original recording domain.
    """

    raw_y: np.ndarray
    raw_sr: int
    analysis: AudioData
    analysis_normalization_gain: float


def load_audio_views(path: str, sr: int = 16000) -> AudioViews:
    if int(sr) <= 0:
        raise ValueError(f"Invalid target sample rate: {sr}")

    raw_y, raw_sr = librosa.load(path, sr=None, mono=True, dtype=np.float64)
    raw_y = np.asarray(raw_y, dtype=np.float64).reshape(-1)
    raw_sr = int(raw_sr)
    if raw_y.size == 0:
        raise ValueError(f"Empty audio: {path}")
    if raw_sr <= 0:
        raise ValueError(f"Invalid decoded sample rate for {path}: {raw_sr}")

    if raw_sr == int(sr):
        analysis_y = raw_y.copy()
    else:
        analysis_y = librosa.resample(
            raw_y,
            orig_sr=raw_sr,
            target_sr=int(sr),
            res_type="soxr_hq",
        )
        analysis_y = np.asarray(analysis_y, dtype=np.float64)

    analysis_peak = float(np.max(np.abs(analysis_y))) if analysis_y.size else 0.0
    normalization_gain = 1.0
    if analysis_peak > 0:
        normalization_gain = 1.0 / (analysis_peak + 1e-9)
        analysis_y = analysis_y * normalization_gain

    tagged_y = _analysis_signal(
        analysis_y,
        raw_y=raw_y,
        raw_sr=raw_sr,
        normalization_gain=normalization_gain,
    )
    analysis = AudioData(
        y=tagged_y,
        sr=int(sr),
        duration=len(tagged_y) / int(sr),
    )
    return AudioViews(
        raw_y=raw_y,
        raw_sr=raw_sr,
        analysis=analysis,
        analysis_normalization_gain=float(normalization_gain),
    )


def load_audio(path: str, sr: int = 16000) -> AudioData:
    """Compatibility loader returning the normalized analysis-domain signal.

    New sentence-final evaluation code should prefer :func:`load_audio_views`
    when recording quality is also needed.
    """

    return load_audio_views(path, sr=sr).analysis


def trim_silence(y: np.ndarray, top_db: float = 30.0) -> Tuple[np.ndarray, Tuple[int, int]]:
    yt, idx = librosa.effects.trim(y, top_db=top_db)
    if yt.size == 0:
        return y, (0, len(y))
    return yt, (int(idx[0]), int(idx[1]))


def rms_energy(y: np.ndarray, frame_length: int = 1024, hop_length: int = 256) -> np.ndarray:
    return librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]


def _extract_f0_pyin(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, str]:
    fmin = librosa.note_to_hz("E2")
    hop_length = int(round(sr * 0.010))
    f0, _voiced_flag, _voiced_prob = librosa.pyin(
        y.astype(np.float32),
        fmin=fmin,
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=1024,
        hop_length=hop_length,
    )
    if _voiced_prob is not None:
        floor_artifact = np.isfinite(f0) & (f0 <= fmin * 1.03) & (_voiced_prob <= 0.02)
        f0[floor_artifact] = np.nan
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    f0 = np.nan_to_num(f0, nan=0.0)
    return times, f0, "librosa.pyin"


def extract_f0(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Extract F0. Prefer pyworld, fallback to librosa.pyin.

    Returns:
      times: seconds
      f0: Hz, unvoiced = 0
      method: string
    """
    try:
        import pyworld as pw

        _f0, t = pw.dio(y.astype(np.float64), sr)
        f0 = pw.stonemask(y.astype(np.float64), _f0, t, sr)
        voiced_ratio = float(np.mean(np.asarray(f0) > 0)) if len(f0) else 0.0
        if len(y) / max(sr, 1) >= 0.5 and voiced_ratio < 0.25:
            times, f0_pyin, method = _extract_f0_pyin(y, sr)
            return times, f0_pyin, f"{method} (pyworld_low_coverage)"
        return np.asarray(t), np.asarray(f0), "pyworld.dio+stonemask"
    except Exception:
        return _extract_f0_pyin(y, sr)


def log_f0_normalize(f0_values: np.ndarray) -> np.ndarray:
    x = np.array(f0_values, dtype=float)
    valid = np.isfinite(x) & (x > 0)
    out = np.full_like(x, np.nan, dtype=float)
    if valid.sum() < 2:
        return out
    logx = np.log(x[valid])
    out[valid] = (np.log(x[valid]) - np.mean(logx)) / (np.std(logx) + 1e-8)
    return out


def median_f0_by_mora(times: np.ndarray, f0: np.ndarray, boundaries: List[Tuple[float, float]]) -> List[float]:
    values: List[float] = []
    for start, end in boundaries:
        # Use center part to avoid consonant/transition noise at boundaries.
        duration = end - start
        if duration < 0.14:
            cs = start
            ce = end
        else:
            cs = start + 0.25 * duration
            ce = end - 0.25 * duration
        mask = (times >= cs) & (times <= ce) & (f0 > 0)
        if np.sum(mask) == 0 and duration < 0.16:
            # Short onset morae can have only one or two voiced frames after the
            # handoff from the consonant. Look slightly to the right instead of
            # treating a clear vowel onset as missing F0.
            right_margin = min(0.08, max(0.04, duration * 0.75))
            mask = (times >= start) & (times <= end + right_margin) & (f0 > 0)
        if np.sum(mask) == 0:
            values.append(float("nan"))
        else:
            values.append(float(np.median(f0[mask])))
    return values


def detect_pauses(y: np.ndarray, sr: int, min_pause_sec: float = 0.30) -> Dict:
    """Detect long pauses inside the analysis clip.

    Callers should pass endpointed speech, not the fixed-duration raw recording,
    when pause ratio is used for fluency or expression/style scoring.
    """
    hop = 256
    frame_length = 1024
    rms = rms_energy(y, frame_length=frame_length, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    if len(rms) == 0:
        return {"pause_count": 0, "pause_total": 0.0, "pause_ratio": 0.0, "pause_segments": []}

    threshold = max(float(np.percentile(rms, 20)), float(np.max(rms)) * 0.08)
    silent = rms < threshold

    segments = []
    start = None
    for i, is_silent in enumerate(silent):
        t = float(times[i])
        if is_silent and start is None:
            start = t
        if (not is_silent or i == len(silent) - 1) and start is not None:
            end = t
            if end - start >= min_pause_sec:
                segments.append((start, end))
            start = None

    pause_total = float(sum(e - s for s, e in segments))
    duration = len(y) / sr
    return {
        "pause_count": len(segments),
        "pause_total": pause_total,
        "pause_ratio": pause_total / max(duration, 1e-6),
        "pause_segments": segments,
        "analysis_duration_sec": duration,
    }


def basic_energy_stats(y: np.ndarray) -> Dict[str, float]:
    rms = rms_energy(y)
    if len(rms) == 0:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0}
    mean = float(np.mean(rms))
    std = float(np.std(rms))
    return {"mean": mean, "std": std, "cv": std / (mean + 1e-8)}