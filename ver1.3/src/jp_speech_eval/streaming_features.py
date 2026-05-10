from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np

from .config import DEFAULT_SCORING_CONFIG
from .schemas import FrameFeatures


def _dbfs(rms: float) -> float:
    return 20.0 * float(np.log10(max(rms, 1e-8)))


def autocorr_f0(
    y: np.ndarray,
    sr: int,
    fmin: float = 70.0,
    fmax: float = 500.0,
    min_corr: float = 0.28,
) -> Optional[float]:
    """Small CPU-only F0 estimator for realtime coarse feedback."""
    y = np.asarray(y, dtype=float)
    if y.size < int(sr * 0.025):
        return None
    y = y - np.mean(y)
    rms = float(np.sqrt(np.mean(y * y)))
    if rms < 1e-5:
        return None
    y = y * np.hanning(len(y))
    corr = np.correlate(y, y, mode="full")[len(y) - 1 :]
    if corr.size == 0 or corr[0] <= 1e-8:
        return None
    min_lag = max(1, int(sr / fmax))
    max_lag = min(len(corr) - 1, int(sr / fmin))
    if max_lag <= min_lag:
        return None
    segment = corr[min_lag : max_lag + 1]
    lag = int(np.argmax(segment) + min_lag)
    corr_ratio = float(corr[lag] / corr[0])
    if corr_ratio < min_corr:
        return None
    return float(sr / lag)


class StreamingFeatureExtractor:
    """
    Frame-level acoustic feature extractor for realtime feedback.

    It keeps a short rolling buffer and computes RMS + coarse F0 from that buffer.
    This is intentionally lightweight; sentence-final evaluation still uses pyworld.
    """

    def __init__(
        self,
        sr: int = 16000,
        analysis_window_ms: float = 60.0,
        min_rms_voiced: float = 0.012,
        f0_min_hz: float = 70.0,
        f0_max_hz: float = 500.0,
    ) -> None:
        self.sr = int(sr)
        self.window_samples = max(1, int(self.sr * analysis_window_ms / 1000.0))
        self.min_rms_voiced = float(min_rms_voiced)
        self.f0_min_hz = float(f0_min_hz)
        self.f0_max_hz = float(f0_max_hz)
        self.buffer: Deque[float] = deque(maxlen=self.window_samples)
        self.samples_seen = 0

    @classmethod
    def from_config(cls, config=None, sr: int = 16000) -> "StreamingFeatureExtractor":
        cfg = (config or DEFAULT_SCORING_CONFIG)["realtime"]
        return cls(
            sr=sr,
            analysis_window_ms=float(cfg["analysis_window_ms"]),
            min_rms_voiced=float(cfg["min_rms_voiced"]),
            f0_min_hz=float(cfg["f0_min_hz"]),
            f0_max_hz=float(cfg["f0_max_hz"]),
        )

    def process_chunk(self, chunk: np.ndarray) -> FrameFeatures:
        chunk = np.asarray(chunk, dtype=float).reshape(-1)
        if chunk.size == 0:
            t = self.samples_seen / self.sr
            return FrameFeatures(t, 0.0, -160.0, False, None, None)

        for v in chunk:
            self.buffer.append(float(v))
        self.samples_seen += int(chunk.size)
        t = self.samples_seen / self.sr

        buf = np.asarray(self.buffer, dtype=float)
        rms = float(np.sqrt(np.mean(buf * buf))) if buf.size else 0.0
        f0 = None
        if rms >= self.min_rms_voiced:
            f0 = autocorr_f0(buf, self.sr, fmin=self.f0_min_hz, fmax=self.f0_max_hz)
        is_voiced = f0 is not None
        return FrameFeatures(
            time_sec=round(float(t), 4),
            rms=rms,
            dbfs=_dbfs(rms),
            is_voiced=is_voiced,
            f0_hz=None if f0 is None else round(float(f0), 3),
            log_f0=None if f0 is None else float(np.log(f0)),
        )
