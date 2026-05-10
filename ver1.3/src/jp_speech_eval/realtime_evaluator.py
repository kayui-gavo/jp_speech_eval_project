from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import DEFAULT_SCORING_CONFIG
from .schemas import FrameFeatures, RealtimeFeedback
from .sentence_cache import SentenceCache


@dataclass
class RealtimeEvaluator:
    """
    Karaoke-like coarse feedback.

    This module is intentionally approximate. It estimates current mora progress
    from voiced elapsed time and pre-cached reference mora proportions. Detailed
    mora alignment should still be done after the utterance ends.
    """

    cache: SentenceCache
    expected_user_duration_sec: Optional[float] = None
    pitch_margin_log: float = 0.08
    speech_rms_threshold: float = 0.060
    maybe_end_silence_sec: float = 0.25
    end_silence_sec: float = 0.65
    speech_start_sec: float = 0.08
    voiced_elapsed_sec: float = 0.0
    last_time_sec: float = 0.0
    speech_started_at_sec: Optional[float] = None
    silence_elapsed_sec: float = 0.0
    speech_candidate_sec: float = 0.0
    endpoint_state: str = "waiting_for_speech"
    log_f0_history: List[float] = field(default_factory=list)

    @classmethod
    def from_config(
        cls,
        cache: SentenceCache,
        config=None,
        expected_user_duration_sec: Optional[float] = None,
    ) -> "RealtimeEvaluator":
        cfg = (config or DEFAULT_SCORING_CONFIG)["realtime"]
        return cls(
            cache=cache,
            expected_user_duration_sec=expected_user_duration_sec,
            pitch_margin_log=float(cfg["pitch_margin_log"]),
            speech_rms_threshold=float(cfg.get("speech_rms_threshold", 0.060)),
            maybe_end_silence_sec=float(cfg.get("maybe_end_silence_sec", 0.25)),
            end_silence_sec=float(cfg.get("end_silence_sec", 0.65)),
            speech_start_sec=float(cfg.get("speech_start_sec", 0.08)),
        )

    def _current_mora_index(self) -> int:
        n = self.cache.mora_count
        if n <= 0:
            return 0
        if self.endpoint_state == "ended":
            return n - 1
        ref_total = max(self.cache.meta.ref_duration_sec, 1e-6)
        expected_total = self.expected_user_duration_sec or ref_total
        # Realtime display should not let a long quiet tail make the mora cursor
        # crawl. Detailed duration scoring happens after endpointing offline.
        expected_total = min(max(float(expected_total), ref_total * 0.65), ref_total * 1.25)
        elapsed = self.voiced_elapsed_sec
        if self.speech_started_at_sec is not None:
            elapsed = max(elapsed, self.last_time_sec - self.speech_started_at_sec)
        progress_ref_time = min(elapsed / max(expected_total, 1e-6), 1.0) * ref_total
        for i, (_s, e) in enumerate(self.cache.meta.ref_mora_boundaries):
            if progress_ref_time <= e:
                return i
        return n - 1

    def _volume_state(self, features: FrameFeatures) -> str:
        if self.endpoint_state in {"waiting_for_speech", "maybe_ending", "ended"}:
            return self.endpoint_state
        if features.rms < self.speech_rms_threshold:
            return "too_quiet"
        if features.rms > 0.65:
            return "too_loud"
        return "ok"

    def _pitch_state(self, features: FrameFeatures, target_pitch: str) -> str:
        if self.endpoint_state in {"waiting_for_speech", "maybe_ending", "ended"}:
            return self.endpoint_state
        if features.log_f0 is None or not features.is_voiced:
            return "unvoiced"
        self.log_f0_history.append(float(features.log_f0))
        if len(self.log_f0_history) < 5:
            return "warming_up"
        median = float(np.median(self.log_f0_history[-30:]))
        diff = float(features.log_f0 - median)
        observed = "H" if diff >= self.pitch_margin_log else "L" if diff <= -self.pitch_margin_log else "MID"
        if target_pitch == "H" and observed == "L":
            return "too_low"
        if target_pitch == "L" and observed == "H":
            return "too_high"
        return "ok"

    def _update_endpoint_state(self, features: FrameFeatures, dt: float) -> None:
        has_energy = features.rms >= self.speech_rms_threshold
        has_voiced_energy = bool(features.is_voiced and features.rms >= self.speech_rms_threshold * 0.5)
        has_speech = bool(has_energy or has_voiced_energy)
        if self.endpoint_state == "ended":
            return
        if self.endpoint_state == "waiting_for_speech":
            self.speech_candidate_sec = self.speech_candidate_sec + dt if has_energy else 0.0
            if self.speech_candidate_sec < self.speech_start_sec:
                return
            self.endpoint_state = "speaking"
            self.speech_started_at_sec = max(0.0, float(features.time_sec) - self.speech_candidate_sec)

        if has_speech:
            self.silence_elapsed_sec = 0.0
            self.voiced_elapsed_sec += dt
            return

        self.silence_elapsed_sec += dt
        if self.silence_elapsed_sec >= self.end_silence_sec:
            self.endpoint_state = "ended"
        elif self.silence_elapsed_sec >= self.maybe_end_silence_sec:
            self.endpoint_state = "maybe_ending"

    def update(self, features: FrameFeatures) -> RealtimeFeedback:
        dt = max(0.0, float(features.time_sec) - self.last_time_sec)
        self.last_time_sec = float(features.time_sec)
        self._update_endpoint_state(features, dt)

        idx = self._current_mora_index()
        moras = self.cache.meta.moras
        target_pitch = self.cache.meta.target_pitch[idx] if idx < len(self.cache.meta.target_pitch) else "?"
        mora = moras[idx] if idx < len(moras) else "?"
        return RealtimeFeedback(
            time_sec=features.time_sec,
            endpoint_state=self.endpoint_state,
            mora_index=idx + 1,
            mora=mora,
            target_pitch=target_pitch,
            f0_hz=features.f0_hz,
            volume_state=self._volume_state(features),
            pitch_state=self._pitch_state(features, target_pitch),
            voiced_elapsed_sec=round(float(self.voiced_elapsed_sec), 4),
        )
