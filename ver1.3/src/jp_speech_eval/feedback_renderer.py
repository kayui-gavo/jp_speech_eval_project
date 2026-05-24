from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional

from .reliability_gate import evaluate_reliability_gate
from .scoring_policy import ScoringPolicy, policy_from_result
from .special_mora_scorer import score_special_mora_timing


@dataclass(frozen=True)
class UserFacingResult:
    mode: str
    reliability: str
    practice_check_result: str
    user_messages: List[str]
    focus_feedback: Optional[Dict[str, Any]]
    display_total_score: bool
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _debug_payload(result: Mapping[str, Any], policy: ScoringPolicy, gate: Any) -> Dict[str, Any]:
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    pronunciation = details.get("pronunciation") if isinstance(details.get("pronunciation"), Mapping) else {}
    prosody = details.get("prosody") if isinstance(details.get("prosody"), Mapping) else {}
    alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}
    return {
        "debug_total_score": result.get("total_score"),
        "pronunciation_score": result.get("pronunciation_score"),
        "prosody_score": result.get("prosody_score"),
        "fluency_score": result.get("fluency_score"),
        "expression_proxy_score": result.get("tone_score"),
        "alignment_confidence": reliability.get("alignment"),
        "mora_duration_cv": pronunciation.get("mora_duration_cv"),
        "special_mora_ratios": pronunciation.get("special_mora_diagnostics"),
        "f0_voiced_coverage": reliability.get("f0_coverage"),
        "reference_source": details.get("reference_source"),
        "weak_reference": policy.weak_reference,
        "demo_only": policy.demo_only,
        "scoring_policy": policy.to_dict(),
        "reliability_gate": gate.to_dict(),
        "alignment": alignment,
        "prosody_debug": {
            "contour_corr": prosody.get("contour_corr"),
            "transition_agreement": prosody.get("transition_agreement"),
            "pitch_target_source": prosody.get("pitch_target_source"),
            "pitch_target_consistency": prosody.get("pitch_target_consistency"),
        },
    }


def render_user_facing_result(result: Mapping[str, Any], *, mode: str | None = None) -> Dict[str, Any]:
    policy = policy_from_result(result, mode=mode)
    gate = evaluate_reliability_gate(result, policy)
    messages: List[str] = list(gate.messages)
    focus: Optional[Dict[str, Any]] = None

    if policy.demo_only:
        messages.append("このモードは参考音のデモです。発音の正しさ判定には使いません。")
        focus = {"category": "demo", "message": messages[-1]}
    elif policy.weak_reference:
        messages.append("確認した文をもとにした練習用フィードバックです。厳密な発音採点ではありません。")
        focus = {"category": "weak_reference", "message": messages[-1]}

    if gate.allow_special_mora_feedback and policy.allow_special_mora_feedback:
        special = [item for item in score_special_mora_timing(result) if item.status in {"too_short", "too_long"}]
        if special:
            item = special[0]
            focus = {"category": "special_mora", **item.to_dict()}
            messages.append(item.message)

    if not messages:
        raw_feedback = [str(item) for item in (result.get("feedback") or [])]
        for item in raw_feedback:
            if not gate.allow_pitch_feedback and ("音高" in item or "語調" in item or "语调" in item):
                continue
            messages.append(item)
            break
    if not messages:
        messages.append("今回の練習は大きな問題なく確認できました。")

    if gate.practice_check_result == "ok" and any("もう少し" in msg or "注意" in msg for msg in messages):
        practice = "needs_attention"
    else:
        practice = gate.practice_check_result

    return UserFacingResult(
        mode=policy.mode,
        reliability=gate.reliability,
        practice_check_result=practice,
        user_messages=messages[:2],
        focus_feedback=focus,
        display_total_score=False,
        debug=_debug_payload(result, policy, gate),
    ).to_dict()
