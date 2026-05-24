from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping

from .scoring_policy import ScoringPolicy


@dataclass(frozen=True)
class ReliabilityGate:
    reliability: str
    practice_check_result: str
    blocked_categories: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    allow_special_mora_feedback: bool = True
    allow_pitch_feedback: bool = False
    allow_pronunciation_detail: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _pick(mapping: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def evaluate_reliability_gate(result: Mapping[str, Any], policy: ScoringPolicy) -> ReliabilityGate:
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    recording = details.get("recording_quality") if isinstance(details.get("recording_quality"), Mapping) else {}
    content = details.get("content_match") if isinstance(details.get("content_match"), Mapping) else {}
    alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}
    mora_count = len(result.get("moras") or [])

    level = str(reliability.get("level") or "medium")
    overall = float(reliability.get("overall", 0.0) or 0.0)
    f0_coverage = float(reliability.get("f0_coverage", 0.0) or 0.0)
    alignment_score = float(reliability.get("alignment", 1.0) or 0.0)
    recording_score = float(recording.get("score", 1.0) or 1.0)
    content_status = str(content.get("status") or "unknown")
    alignment_mode = str(result.get("alignment_mode") or alignment.get("mode") or "")

    messages: List[str] = []
    reasons: List[str] = []
    blocked: List[str] = []
    practice = "ok"
    allow_detail = True
    allow_special = policy.allow_special_mora_feedback
    allow_pitch = policy.allow_pitch_feedback

    if recording_score < 0.35 or "recording_quality" in str(reliability.get("warnings", [])):
        return ReliabilityGate(
            reliability="unscorable",
            practice_check_result="retry",
            blocked_categories=["content", "special_mora", "pitch", "pronunciation"],
            messages=["録音が聞き取りにくいため、もう一度録音してください。"],
            reasons=["recording_quality_bad"],
            allow_special_mora_feedback=False,
            allow_pitch_feedback=False,
            allow_pronunciation_detail=False,
        )

    if policy.allow_content_match_score and content_status == "fail":
        return ReliabilityGate(
            reliability="unscorable",
            practice_check_result="retry",
            blocked_categories=["special_mora", "pitch", "pronunciation"],
            messages=["目標文と違う内容に聞こえます。もう一度読んでください。"],
            reasons=["content_mismatch"],
            allow_special_mora_feedback=False,
            allow_pitch_feedback=False,
            allow_pronunciation_detail=False,
        )

    if level == "low" or overall < 0.45 or alignment_score < 0.45 or alignment_mode.endswith("fallback_equal"):
        practice = "retry"
        allow_detail = False
        allow_special = False
        allow_pitch = False
        blocked.extend(["special_mora", "pitch", "pronunciation"])
        messages.append("今回は細かい発音判定が難しいため、もう一度録音してください。")
        reasons.append("alignment_confidence_low")
    elif overall < 0.75:
        practice = "needs_attention"
        messages.append("今回は一部の判定だけ参考にしてください。")
        reasons.append("medium_reliability")

    if mora_count <= 3:
        allow_pitch = False
        blocked.append("pitch")
        reasons.append("short_utterance")
    if f0_coverage < 0.50:
        allow_pitch = False
        blocked.append("pitch")
        reasons.append("low_f0_coverage")

    if not allow_pitch and "pitch" not in blocked:
        blocked.append("pitch")
    reliability_label = "high" if overall >= 0.85 and level != "low" else "medium" if overall >= 0.45 else "low"
    if policy.weak_reference and practice == "ok":
        practice = "needs_attention"
    return ReliabilityGate(
        reliability=reliability_label,
        practice_check_result=practice,
        blocked_categories=sorted(set(blocked)),
        messages=messages,
        reasons=reasons,
        allow_special_mora_feedback=allow_special,
        allow_pitch_feedback=allow_pitch,
        allow_pronunciation_detail=allow_detail,
    )
