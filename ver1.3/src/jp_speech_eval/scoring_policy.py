from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from .target_specs import PITCH_FEEDBACK_LEVELS, verified_level_from_source


@dataclass(frozen=True)
class ScoringPolicy:
    mode: str
    target_source: str
    verified_level: str
    weak_reference: bool
    demo_only: bool
    allow_content_match_score: bool
    allow_pronunciation_feedback: str
    allow_pitch_feedback: bool
    allow_special_mora_feedback: bool
    allow_total_score_display: bool
    exclude_from_pronunciation_score: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def policy_from_result(result: Mapping[str, Any], *, mode: str | None = None) -> ScoringPolicy:
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    mode_name = str(mode or details.get("mode") or result.get("mode") or "reference")
    target_source = str(
        details.get("pitch_target_source")
        or details.get("reference_source")
        or (result.get("prosody_metrics") or {}).get("pitch_target_source")
        or "auto_pyopenjtalk"
    )
    verified_level = str(details.get("verified_level") or verified_level_from_source(target_source))
    weak_reference = bool(details.get("weak_reference")) or mode_name in {
        "asr_confirmed_weak_reference",
        "asr_pseudo_reference",
        "kanade_asr_voice_reference",
    }
    demo_only = bool(details.get("demo_only")) or mode_name.startswith("kanade")
    exclude = demo_only or bool(details.get("exclude_from_pronunciation_score"))
    allow_pitch = (
        verified_level in PITCH_FEEDBACK_LEVELS
        and not weak_reference
        and not demo_only
    )
    allow_pron = "limited" if weak_reference or demo_only else "standard"
    return ScoringPolicy(
        mode=mode_name,
        target_source=target_source,
        verified_level=verified_level,
        weak_reference=weak_reference,
        demo_only=demo_only,
        allow_content_match_score=not weak_reference and not demo_only,
        allow_pronunciation_feedback=allow_pron,
        allow_pitch_feedback=allow_pitch,
        allow_special_mora_feedback=not demo_only,
        allow_total_score_display=False,
        exclude_from_pronunciation_score=exclude,
    )
