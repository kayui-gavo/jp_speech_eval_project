from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_CONFIG = ROOT / "configs" / "special_mora_threshold_profiles.json"


@dataclass(frozen=True)
class SpecialMoraThresholdProfile:
    profile_name: str
    threshold_file: Optional[str] = None
    allowed_user_facing_types: List[str] = field(default_factory=list)
    blocked_types: List[str] = field(default_factory=list)
    debug_only_types: List[str] = field(default_factory=list)
    user_facing_modes_allowed: List[str] = field(default_factory=list)
    require_high_evidence: bool = True
    suppress_near_boundary: bool = True
    too_long_user_facing_allowed: bool = False
    default_user_facing_enabled: bool = False
    weak_reference_hint_allowed: bool = False
    description: str = ""
    limitations: List[str] = field(default_factory=list)
    fallback_reason: str = ""

    def threshold_path(self) -> Optional[Path]:
        if not self.threshold_file:
            return None
        path = Path(self.threshold_file)
        return path if path.is_absolute() else ROOT / path

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        path = self.threshold_path()
        data["resolved_threshold_file"] = None if path is None else str(path)
        return data


def _default_safe(reason: str = "") -> SpecialMoraThresholdProfile:
    return SpecialMoraThresholdProfile(
        profile_name="default_safe",
        threshold_file="configs/special_mora_thresholds_v2.json",
        debug_only_types=["long_vowel", "moraic_nasal", "sokuon", "yoon"],
        blocked_types=["sokuon"],
        description="Fallback profile: debug only, no user-facing special mora feedback.",
        fallback_reason=reason,
    )


def load_threshold_profile(
    profile_name: str | None = None,
    *,
    config_path: str | Path | None = None,
) -> SpecialMoraThresholdProfile:
    path = Path(config_path) if config_path is not None else DEFAULT_PROFILE_CONFIG
    if not path.exists():
        return _default_safe("missing_threshold_profile_config")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_safe("invalid_threshold_profile_config")
    profiles = data.get("profiles", {})
    if not isinstance(profiles, Mapping):
        return _default_safe("invalid_threshold_profile_config")
    name = str(profile_name or data.get("default_profile") or "default_safe")
    raw = profiles.get(name)
    if not isinstance(raw, Mapping):
        fallback = profiles.get("default_safe")
        if isinstance(fallback, Mapping):
            return SpecialMoraThresholdProfile(**{**fallback, "fallback_reason": f"unknown_profile:{name}"})
        return _default_safe(f"unknown_profile:{name}")
    return SpecialMoraThresholdProfile(**raw)
