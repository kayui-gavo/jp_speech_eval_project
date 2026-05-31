from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PhoneSegment:
    phone: str
    start: float
    end: float
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MoraSegment:
    mora: str
    start: float
    end: float
    phones: List[str] = field(default_factory=list)
    special_mora_type: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentEvidence:
    utterance_id: str
    target_text: str
    method: str
    phone_segments: List[PhoneSegment] = field(default_factory=list)
    mora_segments: List[MoraSegment] = field(default_factory=list)
    word_or_phrase_segments: List[Dict[str, Any]] = field(default_factory=list)
    alignment_confidence: float = 0.0
    fallback_used: bool = False
    usable_for_mora_feedback: bool = False
    usable_for_special_mora_feedback: bool = False
    usable_for_pitch_feedback: bool = False
    warning_flags: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None
    mapping_debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["phone_segments"] = [item.to_dict() for item in self.phone_segments]
        data["mora_segments"] = [item.to_dict() for item in self.mora_segments]
        return data
