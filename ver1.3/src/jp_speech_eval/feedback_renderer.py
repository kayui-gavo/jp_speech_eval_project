from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .consumer_dimension_policy import build_consumer_score_dimensions
from .reliability_gate import evaluate_reliability_gate
from .score_evidence_quality import build_score_evidence_quality
from .scoring_policy import ScoringPolicy, policy_from_result
from .special_mora_scorer import (
    decide_special_mora_runtime,
    select_special_mora_feedback_candidate,
    special_mora_score_from_decisions,
)
from .user_score_policy import apply_user_score_policy
from .user_facing_policy import (
    PracticeScore,
    UserFacingResult,
    practice_score_explanation,
    practice_score_label,
    user_message,
)


def _debug_payload(
    result: Mapping[str, Any],
    policy: ScoringPolicy,
    gate: Any,
    *,
    special_mora_decisions: List[Mapping[str, Any]],
    special_mora_score: Optional[float],
    special_mora_profile: Mapping[str, Any],
    user_score_policy: Mapping[str, Any],
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
        "user_score_policy": dict(user_score_policy),
        "reliability_gate": gate.to_dict(),
        "alignment": alignment,
        "prosody_debug": {
            "contour_corr": prosody.get("contour_corr"),
            "transition_agreement": prosody.get("transition_agreement"),
            "pitch_target_source": prosody.get("pitch_target_source"),
            "pitch_target_consistency": prosody.get("pitch_target_consistency"),
        },
    }


def _mode_notice(policy: ScoringPolicy, gate: Any) -> str:
    if policy.demo_only and "kanade" in policy.mode:
        return user_message("notice.kanade")
    if policy.weak_reference:
        return user_message("notice.weak_reference")
    if policy.mode == "reference_mismatch_general_japanese":
        return user_message("notice.general_japanese_fallback")
    if policy.broad_mode:
        return user_message("notice.general_japanese")
    if gate.allow_pitch_feedback:
        return user_message("notice.fixed_verified")
    return user_message("notice.fixed_limited")


def _policy_message(message_key: str) -> str:
    return user_message(f"score_policy.{message_key}") if message_key else ""


def _score_dimensions(
    result: Mapping[str, Any],
    gate: Any,
    user_score: Mapping[str, Any],
    *,
    mode: str,
) -> List[Dict[str, Any]]:
    """Expose the same four semantic components used by the product total.

    Lexical pitch accent remains conditional detail feedback. It is not the
    top-level intonation dimension. If the reliability gate makes the practice
    result truly unscorable/retry, all four dimensions become unavailable.
    """
    surface = dict(user_score)
    if gate.reliability == "unscorable" or gate.practice_check_result == "retry":
        surface["display_score"] = None
    return build_consumer_score_dimensions(result, surface, mode=mode)


def _status(policy: ScoringPolicy, gate: Any, focus: Optional[Dict[str, Any]]) -> str:
    if gate.practice_check_result == "retry":
        return "retry"
    if policy.demo_only:
        return "debug_only"
    if gate.practice_check_result == "needs_attention":
        return "practice_suggestion"
    if focus and focus.get("category") not in {"demo", "weak_reference"}:
        return "practice_suggestion"
    return "pass"


def _summary_text(status: str, messages: List[str]) -> str:
    if status == "retry":
        return user_message("status.retry")
    if status == "debug_only":
        return user_message("status.debug_only")
    if status == "practice_suggestion":
        return user_message("status.practice_suggestion")
    return user_message("status.pass")


def _suppressed_reasons(gate: Any, decision_dicts: List[Mapping[str, Any]]) -> List[str]:
    reasons = list(gate.reasons or [])
    for item in decision_dicts:
        reason = item.get("suppression_reason")
        if reason and reason not in reasons:
            reasons.append(str(reason))
    return reasons


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


def _is_pitch_or_prosody_message(message: str) -> bool:
    terms = (
        "音高",
        "語調",
        "语调",
        "韵律",
        "韻律",
        "抑揚",
        "アクセント",
        "イントネーション",
        "pitch",
        "prosody",
        "intonation",
        "accent",
    )
    lower = message.lower()
    return any(term.lower() in lower for term in terms)


def render_user_facing_result(
    result: Mapping[str, Any],
    *,
    mode: str | None = None,
    enable_runtime_special_mora_shadow: bool = True,
    enable_user_facing_calibrated_special_mora: bool = False,
    special_mora_threshold_profile: str | None = "v2_limited_candidate",
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
    user_score = apply_user_score_policy(
        result,
        mode=policy.mode,
        special_mora_decisions=decision_dicts,
    )
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
            focus = {
                "category": "special_mora",
                "type": item.type,
                "mora": item.surface_mora,
                "message": item.feedback_candidate_text,
            }
            messages.append(item.feedback_candidate_text)

    policy_message = _policy_message(str(user_score.get("main_message_key") or ""))
    if policy_message and policy_message not in messages:
        messages.append(policy_message)
        if focus is None and str(user_score.get("main_message_key")) not in {"weak_reference_practice_feedback"}:
            focus = {"category": "score_policy", "message": policy_message}

    raw_feedback = [str(item) for item in (result.get("feedback") or [])]
    for item in raw_feedback:
        if len(messages) >= 2:
            break
        if gate.practice_check_result != "retry" and _is_retry_message(item):
            continue
        if not gate.allow_pitch_feedback and _is_pitch_or_prosody_message(item):
            continue
        if item not in messages:
            messages.append(item)
    if not messages:
        messages.append("今回の練習は大きな問題なく確認できました。")

    if gate.practice_check_result == "ok" and any("もう少し" in msg or "注意" in msg for msg in messages):
        practice = "needs_attention"
    else:
        practice = gate.practice_check_result
    display_score = user_score.get("display_score") if user_score.get("score_available") else None
    if gate.reliability == "unscorable" or gate.practice_check_result == "retry":
        display_score = None
    status = _status(policy, gate, focus)
    mode_notice = _mode_notice(policy, gate)
    primary = None
    suggestion_type = "none"
    if focus and focus.get("category") not in {"demo", "weak_reference"}:
        primary = str(focus.get("message") or "")
        suggestion_type = str(focus.get("category") or "none")
    elif len(messages) > 1 and status == "practice_suggestion":
        primary = messages[1]
    practice_score = PracticeScore(
        value=display_score,
        label=practice_score_label(display_score, status),
        explanation=practice_score_explanation(mode_notice),
    )
    score_dimensions = _score_dimensions(result, gate, user_score, mode=policy.mode)
    evidence_quality = build_score_evidence_quality(result, score_dimensions)

    return UserFacingResult(
        mode=policy.mode,
        status=status,
        reliability=gate.reliability,
        confidence=gate.reliability,
        practice_check_result=practice,
        practice_score=practice_score,
        summary_text=_summary_text(status, messages),
        primary_suggestion_text=primary,
        suggestion_type=suggestion_type,
        mode_notice=mode_notice,
        debug_available=True,
        suppressed_reasons=_suppressed_reasons(gate, decision_dicts),
        display_score=display_score,
        pronunciation_clarity_score=user_score.get("pronunciation_clarity_score"),
        rhythm_fluency_score=user_score.get("rhythm_fluency_score"),
        practice_completion_score=user_score.get("practice_completion_score"),
        confidence_label=str(user_score.get("confidence_label") or gate.reliability),
        score_policy_warnings=list(user_score.get("score_policy_warnings") or []),
        score_caps=dict(user_score.get("score_caps") or {}),
        score_dimensions=score_dimensions,
        recording_analyzability=dict(evidence_quality.get("recording_analyzability") or {}),
        score_evidence=dict(evidence_quality.get("score_evidence") or {}),
        detail_feedback_allowed=bool(user_score.get("detail_feedback_allowed", True)),
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
            user_score_policy=user_score,
        ),
        score_contract_version=str(user_score.get("score_contract_version") or ""),
        evidence_schema_version=str(user_score.get("evidence_schema_version") or ""),
    ).to_dict()
