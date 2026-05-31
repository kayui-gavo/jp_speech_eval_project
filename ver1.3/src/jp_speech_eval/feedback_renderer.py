from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional

from .reliability_gate import evaluate_reliability_gate
from .scoring_policy import ScoringPolicy, policy_from_result
from .special_mora_scorer import (
    decide_special_mora_runtime,
    select_special_mora_feedback_candidate,
    special_mora_score_from_decisions,
)


@dataclass(frozen=True)
class UserFacingResult:
    mode: str
    reliability: str
    practice_check_result: str
    display_score: Optional[int]
    user_messages: List[str]
    focus_feedback: Optional[Dict[str, Any]]
    display_total_score: bool
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _debug_payload(
    result: Mapping[str, Any],
    policy: ScoringPolicy,
    gate: Any,
    *,
    special_mora_decisions: List[Mapping[str, Any]],
    special_mora_score: Optional[float],
    special_mora_profile: Mapping[str, Any],
) -> Dict[str, Any]:
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    pronunciation = details.get("pronunciation") if isinstance(details.get("pronunciation"), Mapping) else {}
    prosody = details.get("prosody") if isinstance(details.get("prosody"), Mapping) else {}
    alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}
    fluency = details.get("fluency") if isinstance(details.get("fluency"), Mapping) else {}
    return {
        "debug_total_score": result.get("total_score"),
        "pronunciation_score": result.get("pronunciation_score"),
        "prosody_score": result.get("prosody_score"),
        "fluency_score": result.get("fluency_score"),
        "rhythm_timing_score": fluency.get("rhythm_timing_score"),
        "delivery_fluency_score": fluency.get("delivery_fluency_score"),
        "expression_proxy_score": result.get("tone_score"),
        "alignment_confidence": reliability.get("alignment"),
        "mora_duration_cv": pronunciation.get("mora_duration_cv"),
        "special_mora_ratios": pronunciation.get("special_mora_diagnostics"),
        "special_mora_decisions": special_mora_decisions,
        "special_mora_evidence_cards": [item.get("evidence_card") for item in special_mora_decisions if item.get("evidence_card")],
        "special_mora_threshold_profile": special_mora_profile,
        "special_mora_score": special_mora_score,
        "special_mora_score_available": special_mora_score is not None,
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


def _as_score(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(100.0, number))


def _weighted_available(scores: Mapping[str, Any], weights: Mapping[str, float]) -> Optional[float]:
    total = 0.0
    denom = 0.0
    for key, weight in weights.items():
        if weight <= 0:
            continue
        value = scores.get(key)
        if value is None or value == "":
            continue
        total += _as_score(value) * weight
        denom += weight
    if denom <= 0:
        return None
    return total / denom


def _display_score(
    result: Mapping[str, Any],
    policy: ScoringPolicy,
    gate: Any,
    *,
    special_mora_score: Optional[float] = None,
) -> Optional[int]:
    if gate.reliability == "unscorable" or gate.practice_check_result == "retry":
        return None
    raw_total = _as_score(result.get("total_score"))
    pronunciation = _as_score(result.get("pronunciation_score"))
    fluency = _as_score(result.get("fluency_score"))
    prosody = _as_score(result.get("prosody_score"))
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    content = details.get("content_match") if isinstance(details.get("content_match"), Mapping) else {}
    fluency_details = details.get("fluency") if isinstance(details.get("fluency"), Mapping) else {}
    pronunciation_details = details.get("pronunciation") if isinstance(details.get("pronunciation"), Mapping) else {}
    prosody_details = details.get("prosody") if isinstance(details.get("prosody"), Mapping) else {}
    content_score = 100.0 if str(content.get("status") or "unknown") in {"pass", "unknown"} else 35.0
    scores = {
        "content_score": content_score,
        "mora_clarity_score": pronunciation,
        "special_mora_score": special_mora_score,
        "rhythm_timing_score": fluency_details.get("rhythm_timing_score", fluency),
        "phrase_intonation_score": prosody_details.get("final_intonation_score"),
        "delivery_fluency_score": fluency_details.get("delivery_fluency_score", fluency),
    }
    if policy.demo_only:
        return None
    if policy.weak_reference:
        display = _weighted_available(scores, {
            "content_score": 0.25,
            "mora_clarity_score": 0.30,
            "special_mora_score": 0.25,
            "rhythm_timing_score": 0.15,
            "delivery_fluency_score": 0.05,
        })
        if display is None:
            display = 0.55 * pronunciation + 0.35 * fluency + 0.10 * prosody
        if gate.reliability == "high" and pronunciation >= 90 and fluency >= 60:
            display = max(display, 85.0)
        return int(round(max(raw_total, display)))
    if not gate.allow_pitch_feedback:
        display = _weighted_available(scores, {
            "content_score": 0.25,
            "mora_clarity_score": 0.30,
            "special_mora_score": 0.20,
            "rhythm_timing_score": 0.15,
            "delivery_fluency_score": 0.10,
        })
        if display is None:
            display = 0.55 * pronunciation + 0.35 * fluency + 0.10 * prosody
        if gate.reliability == "high" and pronunciation >= 90 and fluency >= 60:
            display = max(display, 85.0)
        return int(round(max(raw_total, display)))
    display = _weighted_available(scores, {
        "content_score": 0.25,
        "mora_clarity_score": 0.25,
        "special_mora_score": 0.20,
        "rhythm_timing_score": 0.15,
        "phrase_intonation_score": 0.10,
        "delivery_fluency_score": 0.05,
    })
    if display is not None:
        return int(round(max(raw_total, display)))
    return int(round(raw_total))


def _is_retry_message(message: str) -> bool:
    retry_terms = (
        "重录",
        "再录",
        "録り直",
        "もう一度録音",
        "record again",
        "try again",
    )
    lower = message.lower()
    return any(term.lower() in lower for term in retry_terms)


def render_user_facing_result(
    result: Mapping[str, Any],
    *,
    mode: str | None = None,
    enable_runtime_special_mora_shadow: bool = True,
    enable_user_facing_calibrated_special_mora: bool = False,
    special_mora_threshold_profile: str | None = "default_safe",
    enable_weak_reference_special_mora_hint: bool = False,
) -> Dict[str, Any]:
    policy = policy_from_result(result, mode=mode)
    gate = evaluate_reliability_gate(result, policy)
    decisions = decide_special_mora_runtime(
        result,
        threshold_profile=special_mora_threshold_profile,
        weak_reference=policy.weak_reference,
        mode_name=policy.mode,
        demo_only=policy.demo_only,
        enable_runtime_shadow=enable_runtime_special_mora_shadow,
        enable_user_facing=(
            enable_user_facing_calibrated_special_mora
            and gate.allow_special_mora_feedback
            and policy.allow_special_mora_feedback
            and not policy.demo_only
        ),
        enable_weak_reference_hint=enable_weak_reference_special_mora_hint,
    )
    decision_dicts = [item.to_dict() for item in decisions]
    special_mora_score = special_mora_score_from_decisions(decisions)
    messages: List[str] = list(gate.messages)
    focus: Optional[Dict[str, Any]] = None

    if policy.demo_only:
        messages.append("このモードは参考音のデモです。発音の正しさ判定には使いません。")
        focus = {"category": "demo", "message": messages[-1]}
    elif policy.weak_reference:
        messages.append("確認した文をもとにした練習用フィードバックです。厳密な発音採点ではありません。")
        focus = {"category": "weak_reference", "message": messages[-1]}

    if gate.allow_special_mora_feedback and policy.allow_special_mora_feedback:
        item = select_special_mora_feedback_candidate(decisions)
        if item:
            focus = {"category": "special_mora", **item.to_dict(), "message": item.feedback_candidate_text}
            messages.append(item.feedback_candidate_text)

    raw_feedback = [str(item) for item in (result.get("feedback") or [])]
    for item in raw_feedback:
        if len(messages) >= 2:
            break
        if gate.practice_check_result != "retry" and _is_retry_message(item):
            continue
        if not gate.allow_pitch_feedback and ("音高" in item or "語調" in item or "语调" in item):
            continue
        if item not in messages:
            messages.append(item)
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
        display_score=_display_score(result, policy, gate, special_mora_score=special_mora_score),
        user_messages=messages[:2],
        focus_feedback=focus,
        display_total_score=False,
        debug=_debug_payload(
            result,
            policy,
            gate,
            special_mora_decisions=decision_dicts,
            special_mora_score=special_mora_score,
            special_mora_profile=(decision_dicts[0].get("evidence_card", {}) if decision_dicts else {}),
        ),
    ).to_dict()
