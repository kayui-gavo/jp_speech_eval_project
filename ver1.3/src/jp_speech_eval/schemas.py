from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class MoraRow:
    index: int
    mora: str
    start_sec: float
    end_sec: float
    f0_hz: Optional[float]
    target_pitch: str
    observed_pitch: str


@dataclass(frozen=True)
class FrameFeatures:
    time_sec: float
    rms: float
    dbfs: float
    is_voiced: bool
    f0_hz: Optional[float]
    log_f0: Optional[float]


@dataclass(frozen=True)
class RealtimeFeedback:
    time_sec: float
    endpoint_state: str
    mora_index: int
    mora: str
    target_pitch: str
    f0_hz: Optional[float]
    volume_state: str
    pitch_state: str
    voiced_elapsed_sec: float

    def to_dict(self) -> Dict:
        return {
            "time_sec": self.time_sec,
            "endpoint_state": self.endpoint_state,
            "mora_index": self.mora_index,
            "mora": self.mora,
            "target_pitch": self.target_pitch,
            "f0_hz": self.f0_hz,
            "volume_state": self.volume_state,
            "pitch_state": self.pitch_state,
            "voiced_elapsed_sec": self.voiced_elapsed_sec,
        }
