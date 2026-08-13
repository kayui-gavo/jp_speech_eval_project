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
    dimension_reliability: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clip01(value: Any, default: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def evaluate_reliability_gate(result: Mapping[str, Any], policy: ScoringPolicy) -> ReliabilityGate:
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    recording = details.get("recording_quality") if isinstance(details.get("recording_quality"), Mapping) else {}
    content = details.get("content_match") if isinstance(details.get("content_match"), Mapping) else {}
    alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}
    mora_count = len(result.get("moras") or [])

    endpoint_score = _clip01(reliability.get("endpointing", 1.0))
    alignment_score = _clip01(reliability.get("alignment", 1.0))
    evidence_score = _clip01(reliability.get("mora_evidence", alignment_score))
    f0_coverage = _clip01(reliability.get("f0_coverage", 0.0), default=0.0)
    recording_score = _clip01(recording.get("score", reliability.get("recording_quality", 1.0)))
    content_status = str(content.get("status") or "unknown")
    alignment_mode = str(result.get("alignment_mode") or alignment.get("mode") or "")
    is_fixed_reference = policy.fixed_reference

    if is_fixed_reference:
        core_reliability = 0.30 * endpoint_score + 0.35 * alignment_score + 0.20 * recording_score + 0.15 * evidence_score
    else:
        core_reliability = 0.55 * endpoint_score + 0.45 * recording_score

    messages: List[str] = []
    reasons: List[str] = []
    blocked: List[str] = []
    practice = "ok"
    allow_detail = is_fixed_reference and not policy.weak_reference and not policy.demo_only
    allow_special = policy.allow_special_mora_feedback
    allow_pitch = policy.allow_pitch_feedback and is_fixed_reference and not policy.weak_reference and not policy.demo_only

    if not is_fixed_reference:
        reasons.append("pitch_not_fixed_reference")

    if recording_score < 0.20:
        return ReliabilityGate(
            reliability="unscorable",
            practice_check_result="retry",
            blocked_categories=["content", "special_mora", "pitch", "pronunciation"],
            messages=["録音をうまく確認できませんでした。マイクに少し近づいて、もう一度録音してください。"],
            reasons=["recording_unusable"],
            allow_special_mora_feedback=False,
            allow_pitch_feedback=False,
            allow_pronunciation_detail=False,
            dimension_reliability={
                "audio": recording_score,
                "content": 0.0,
                "alignment": alignment_score,
                "pitch": f0_coverage,
                "special_mora": evidence_score,
            },
        )

    if recording_score < 0.55:
        practice = "needs_attention"
        allow_special = False
        allow_pitch = False
        blocked.extend(["special_mora", "pitch"])
        messages.append("録音条件の影響があるため、今回は全体的な目安を中心に表示します。")
        reasons.append("recording_quality_low_broad_only")

    if policy.allow_content_match_score and content_status in {"fail", "failed", "content_mismatch"}:
        practice = "needs_attention"
        allow_special = False
        allow_pitch = False
        blocked.extend(["special_mora", "pitch"])
        messages.append("目標文とは違う内容に聞こえますが、日本語としての全体的な話し方は評価します。")
        reasons.append("content_mismatch_broad_score")
        allow_detail = False

    if is_fixed_reference:
        if core_reliability < 0.40 or alignment_score < 0.35:
            practice = "needs_attention"
            allow_special = False
            allow_pitch = False
            blocked.extend(["special_mora", "pitch"])
            if not messages:
                messages.append("細かい位置合わせが不安定なため、今回は全体的な話し方を中心に評価します。")
            reasons.append("alignment_confidence_low_broad_only")
            allow_detail = False
        elif core_reliability < 0.75 or alignment_mode.endswith("fallback_equal"):
            practice = "needs_attention"
            if alignment_mode.endswith("fallback_equal"):
                allow_special = False
                allow_pitch = False
                blocked.extend(["special_mora", "pitch"])
                if not messages:
                    messages.append("細かい拍ごとの判定は不安定ですが、全体スコアは表示します。")
                reasons.append("fallback_alignment_broad_only")
                allow_detail = False
            elif not messages:
                messages.append("今回は一部の細かい判定だけ参考にしてください。")
                reasons.append("medium_reliability")

    if mora_count <= 3:
        allow_pitch = False
        blocked.append("pitch")
        reasons.append("short_utterance")
    if f0_coverage < 0.50:
        allow_pitch = False
        blocked.append("pitch")
        reasons.append("low_f0_coverage")

    if policy.weak_reference:
        allow_pitch = False
        allow_special = False
        blocked.extend(["pitch", "special_mora"])
        if practice == "ok":
            practice = "needs_attention"
        reasons.append("weak_reference_broad_only")
        allow_detail = False

    if not allow_pitch and "pitch" not in blocked:
        blocked.append("pitch")

    reliability_label = "high" if core_reliability >= 0.85 else "medium" if core_reliability >= 0.45 else "low"
    return ReliabilityGate(
        reliability=reliability_label,
        practice_check_result=practice,
        blocked_categories=sorted(set(blocked)),
        messages=messages,
        reasons=reasons,
        allow_special_mora_feedback=allow_special,
        allow_pitch_feedback=allow_pitch,
        allow_pronunciation_detail=allow_detail,
        dimension_reliability={
            "audio": recording_score,
            "content": 1.0 if content_status in {"pass", "unknown", "general_japanese"} else 0.5,
            "alignment": alignment_score if is_fixed_reference else 1.0,
            "pitch": f0_coverage,
            "special_mora": evidence_score if is_fixed_reference else 0.0,
        },
    )
