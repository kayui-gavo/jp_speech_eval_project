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


def evaluate_reliability_gate(result: Mapping[str, Any], policy: ScoringPolicy) -> ReliabilityGate:
    """Control feedback granularity without unnecessarily suppressing scores.

    C-end rule: normal Japanese speech should still receive a broad practice
    score when alignment, F0, or reference evidence is weak. These signals only
    decide how specific the feedback may be. Only truly unusable recordings are
    marked retry/unscorable here.
    """
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
    recording_score = float(recording.get("score", reliability.get("recording_quality", 1.0)) or 1.0)
    content_status = str(content.get("status") or "unknown")
    alignment_mode = str(result.get("alignment_mode") or alignment.get("mode") or "")
    is_fixed_reference = policy.mode in {"reference", "reference_based", "reference_fixed_sentence", "fixed_reference"}

    messages: List[str] = []
    reasons: List[str] = []
    blocked: List[str] = []
    practice = "ok"
    allow_detail = True
    allow_special = policy.allow_special_mora_feedback
    allow_pitch = policy.allow_pitch_feedback and is_fixed_reference and not policy.weak_reference and not policy.demo_only

    if not is_fixed_reference:
        reasons.append("pitch_not_fixed_reference")

    # Only extreme recording failure blocks the whole attempt. Mobile/noisy
    # recordings degrade confidence and local feedback instead of removing score.
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
        )

    if recording_score < 0.55:
        practice = "needs_attention"
        allow_detail = False
        allow_special = False
        allow_pitch = False
        blocked.extend(["special_mora", "pitch", "pronunciation"])
        messages.append("録音条件の影響があるため、今回は全体的な目安を中心に表示します。")
        reasons.append("recording_quality_low_broad_only")

    # Saying another valid Japanese sentence is task mismatch, not a reason to
    # call the speech unscorable. Hide target-local corrections only.
    if policy.allow_content_match_score and content_status in {"fail", "failed", "content_mismatch"}:
        practice = "needs_attention"
        allow_detail = False
        allow_special = False
        allow_pitch = False
        blocked.extend(["special_mora", "pitch", "pronunciation"])
        messages.append("目標文とは違う内容に聞こえますが、日本語としての全体的な話し方は評価します。")
        reasons.append("content_mismatch_broad_score")

    if level == "low" or overall < 0.40 or alignment_score < 0.35:
        practice = "needs_attention"
        allow_detail = False
        allow_special = False
        allow_pitch = False
        blocked.extend(["special_mora", "pitch", "pronunciation"])
        if not messages:
            messages.append("細かい位置合わせが不安定なため、今回は全体的な話し方を中心に評価します。")
        reasons.append("alignment_confidence_low_broad_only")
    elif overall < 0.75 or alignment_mode.endswith("fallback_equal"):
        practice = "needs_attention"
        if alignment_mode.endswith("fallback_equal"):
            allow_detail = False
            allow_special = False
            allow_pitch = False
            blocked.extend(["special_mora", "pronunciation", "pitch"])
            if not messages:
                messages.append("細かい拍ごとの判定は不安定ですが、全体スコアは表示します。")
            reasons.append("fallback_alignment_broad_only")
        elif not messages:
            messages.append("今回は一部の細かい判定だけ参考にしてください。")
            reasons.append("medium_reliability")

    # F0 reliability is dimension-local: it suppresses pitch only.
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
        allow_detail = False
        blocked.extend(["pitch", "pronunciation"])
        if practice == "ok":
            practice = "needs_attention"
        reasons.append("weak_reference_broad_only")

    if not allow_pitch and "pitch" not in blocked:
        blocked.append("pitch")

    reliability_label = "high" if overall >= 0.85 and level != "low" else "medium" if overall >= 0.45 else "low"
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
