from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PROFILE_VERSION = "user_voice_profile_mvp_v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CalibrationSample:
    """One calibration utterance analyzed by the existing v1.3 evaluator."""

    text: str
    audio_path: str
    kana: str
    mora_count: int
    scores: Dict[str, float]
    features: Dict[str, Optional[float]]
    reliability: Dict[str, Any]
    feedback: List[str] = field(default_factory=list)


@dataclass
class UserVoiceProfile:
    """Lightweight user baseline for product feedback, not a new correctness target."""

    user_id: str
    calibration_samples: List[CalibrationSample] = field(default_factory=list)
    f0_median_hz: Optional[float] = None
    f0_range_log: Optional[float] = None
    mora_rate_avg: Optional[float] = None
    avg_mora_duration_sec: Optional[float] = None
    pause_ratio_avg: Optional[float] = None
    intensity_avg: Optional[float] = None
    baseline_scores: Dict[str, float] = field(default_factory=dict)
    common_issues: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    version: str = PROFILE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sample_from_dict(data: Dict[str, Any]) -> CalibrationSample:
    return CalibrationSample(
        text=str(data.get("text", "")),
        audio_path=str(data.get("audio_path", "")),
        kana=str(data.get("kana", "")),
        mora_count=int(data.get("mora_count", 0) or 0),
        scores=dict(data.get("scores") or {}),
        features=dict(data.get("features") or {}),
        reliability=dict(data.get("reliability") or {}),
        feedback=[str(x) for x in (data.get("feedback") or [])],
    )


def user_profile_from_dict(data: Dict[str, Any]) -> UserVoiceProfile:
    samples = [_sample_from_dict(row) for row in data.get("calibration_samples", [])]
    return UserVoiceProfile(
        user_id=str(data.get("user_id", "")),
        calibration_samples=samples,
        f0_median_hz=data.get("f0_median_hz"),
        f0_range_log=data.get("f0_range_log"),
        mora_rate_avg=data.get("mora_rate_avg"),
        avg_mora_duration_sec=data.get("avg_mora_duration_sec"),
        pause_ratio_avg=data.get("pause_ratio_avg"),
        intensity_avg=data.get("intensity_avg"),
        baseline_scores=dict(data.get("baseline_scores") or {}),
        common_issues=[str(x) for x in (data.get("common_issues") or [])],
        created_at=str(data.get("created_at") or utc_now_iso()),
        updated_at=str(data.get("updated_at") or utc_now_iso()),
        version=str(data.get("version") or PROFILE_VERSION),
    )


def save_user_profile(profile: UserVoiceProfile, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)


def load_user_profile(path: str | Path) -> UserVoiceProfile:
    with Path(path).open("r", encoding="utf-8") as f:
        return user_profile_from_dict(json.load(f))
