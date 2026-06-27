from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .feedback_candidates import build_feedback_candidates
from .reliability_gate import evaluate_reliability_gate
from .scoring_policy import ScoringPolicy, policy_from_result
from .special_mora_scorer import (
    decide_special_mora_runtime,
    special_mora_score_from_decisions,
)
from .user_facing_policy import (
    PracticeScore,
    UserFacingResult,
    practice_score_explanation,
    practice_score_label,
    user_message,
)


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _is_degraded_reference_practice(
    result: Mapping[str, Any],
    policy: ScoringPolicy,
    gate: Any,
) -> bool:
    """Return whether fixed-reference output must fall back to practice proxies.

    A failed alignment invalidates strict reference comparisons, but it does not
    automatically invalidate recording-level clarity, broad timing, fluency,
    and weak pitch-naturalness evidence. Content and recording vetoes remain
    authoritative through the reliability gate and weak-overall guardrail.
    """

    if policy.weak_reference or policy.demo_only or gate.reliability == "unscorable":
        return False
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}
    alignment_mode = str(result.get("alignment_mode") or alignment.get("mode") or "")
    reasons = {str(reason) for reason in (gate.reasons or [])}
    return (
        "fallback" in alignment_mode
        or alignment_mode in {"equal", "equal_fallback"}
        or bool({"fallback_alignment", "alignment_confidence_low"}.intersection(reasons))
    )


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
    weak = details.get("weak_reference_native_likeness") if isinstance(details.get("weak_reference_native_likeness"), Mapping) else {}
    weak_guardrail = weak.get("weak_overall_guardrail") if isinstance(weak.get("weak_overall_guardrail"), Mapping) else {}
    alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}
    fluency = details.get("fluency") if isinstance(details.get("fluency"), Mapping) else {}
    content = details.get("content_match") if isinstance(details.get("content_match"), Mapping) else {}
    raw_prosody_score = result.get("prosody_score")
    weak_prosody_score = _first_not_none(
        weak.get("weak_prosody_naturalness_score"),
        result.get("weak_prosody_naturalness_score"),
    )
    degraded_reference_practice = _is_degraded_reference_practice(result, policy, gate)
    if policy.weak_reference or degraded_reference_practice:
        guardrail_blocks = weak_guardrail.get("status") == "no_score"
        # In weak or degraded-reference practice, raw strict-reference prosody
        # is diagnostic only. Never backfill a missing weak pitch estimate with
        # the strict/reference score.
        visible_prosody_score = None if guardrail_blocks else weak_prosody_score
    else:
        visible_prosody_score = raw_prosody_score if gate.allow_pitch_feedback else None
    return {
        "debug_total_score": result.get("total_score"),
        "pronunciation_score": result.get("pronunciation_score"),
        "prosody_score": raw_prosody_score,
        "rhythm_score": result.get("rhythm_score"),
        "rhythm_practice": details.get("rhythm") if isinstance(details.get("rhythm"), Mapping) else {},
        "weak_pronunciation_naturalness_score": _first_not_none(weak.get("weak_pronunciation_naturalness_score"), result.get("weak_pronunciation_naturalness_score")),
        "weak_prosody_naturalness_score": weak_prosody_score,
        "weak_rhythm_naturalness_score": _first_not_none(weak.get("weak_rhythm_naturalness_score"), result.get("weak_rhythm_naturalness_score")),
        "weak_overall_practice_score": _first_not_none(weak.get("weak_overall_practice_score"), result.get("weak_overall_practice_score")),
        "weak_overall_guardrail": weak_guardrail,
        "score_type": details.get("score_type") or result.get("score_type"),
        "strict_reference_available": details.get("strict_reference_available") if "strict_reference_available" in details else result.get("strict_reference_available"),
        "visible_prosody_score": visible_prosody_score,
        "prosody_score_visible": visible_prosody_score is not None,
        "fluency_score": result.get("fluency_score"),
        "rhythm_timing_score": fluency.get("rhythm_timing_score"),
        "delivery_fluency_score": fluency.get("delivery_fluency_score"),
        "expression_proxy_score": result.get("tone_score"),
        "alignment_confidence": reliability.get("alignment"),
        "mora_duration_cv": _first_not_none(
            pronunciation.get("mora_duration_cv"),
            (pronunciation.get("legacy_timing_proxy_details") or {}).get("mora_duration_cv")
            if isinstance(pronunciation.get("legacy_timing_proxy_details"), Mapping)
            else None,
        ),
        "special_mora_ratios": (
            pronunciation.get("legacy_timing_proxy_details", {}).get("special_mora_diagnostics")
            if isinstance(pronunciation.get("legacy_timing_proxy_details"), Mapping)
            else pronunciation.get("special_mora_diagnostics")
        ),
        "special_mora_decisions": special_mora_decisions,
        "special_mora_evidence_cards": [item.get("evidence_card") for item in special_mora_decisions if item.get("evidence_card")],
        "special_mora_threshold_profile": special_mora_profile,
        "special_mora_score": special_mora_score,
        "special_mora_score_available": special_mora_score is not None,
        "f0_voiced_coverage": reliability.get("f0_coverage"),
        "reference_source": details.get("reference_source"),
        "weak_reference": policy.weak_reference,
        "degraded_reference_practice": degraded_reference_practice,
        "demo_only": policy.demo_only,
        "scoring_policy": policy.to_dict(),
        "reliability_gate": gate.to_dict(),
        "alignment": alignment,
        "content_match_visibility": {
            "status": content.get("status"),
            "method": content.get("method"),
            "note": content.get("note"),
            "asr_provider": content.get("asr_provider"),
            "transcript": content.get("transcript"),
            "transcript_kana": content.get("transcript_kana"),
            "target_kana": content.get("target_kana"),
            "kana_similarity": content.get("kana_similarity"),
            "content_mismatch_veto": "content_mismatch_veto" in list(gate.reasons or []),
            "hidden_reasons": list(gate.reasons or []),
        },
        "prosody_debug": {
            "contour_corr": prosody.get("contour_corr"),
            "transition_agreement": prosody.get("transition_agreement"),
            "pitch_target_source": prosody.get("pitch_target_source"),
            "pitch_target_reliability": prosody.get("pitch_target_reliability") or details.get("pitch_target_reliability"),
            "pitch_target_consistency": prosody.get("pitch_target_consistency"),
            "raw_prosody_score": raw_prosody_score,
            "weak_prosody_naturalness_score": weak_prosody_score,
            "score_type": details.get("score_type") or result.get("score_type"),
            "strict_pitch_accent_correctness": False
            if policy.weak_reference or degraded_reference_practice
            else gate.allow_pitch_feedback,
            "visible_prosody_score": visible_prosody_score,
            "visible": visible_prosody_score is not None,
            "hidden_reason": "pitch_blocked" if visible_prosody_score is None else None,
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
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    weak_details = details.get("weak_reference_native_likeness") if isinstance(details.get("weak_reference_native_likeness"), Mapping) else {}
    weak_guardrail = weak_details.get("weak_overall_guardrail") if isinstance(weak_details.get("weak_overall_guardrail"), Mapping) else {}
    weak_no_score = weak_guardrail.get("status") == "no_score"
    if gate.reliability == "unscorable" or weak_no_score:
        return None
    pronunciation = _as_score(result.get("pronunciation_score"))
    fluency = _as_score(result.get("fluency_score"))
    prosody = _as_score(result.get("prosody_score"))
    fluency_details = details.get("fluency") if isinstance(details.get("fluency"), Mapping) else {}
    rhythm = result.get("rhythm_score")
    degraded_reference_practice = _is_degraded_reference_practice(result, policy, gate)
    scores = {
        "mora_clarity_score": pronunciation,
        "rhythm_timing_score": rhythm,
        "delivery_fluency_score": fluency_details.get("delivery_fluency_score", fluency),
        "pitch_score": prosody,
    }
    if policy.demo_only:
        return None
    if policy.weak_reference or degraded_reference_practice:
        weak_overall = _first_not_none(
            result.get("weak_overall_practice_score"),
            weak_details.get("weak_overall_practice_score"),
        )
        if weak_overall is not None:
            return int(round(_as_score(weak_overall)))
        weak_pronunciation = _first_not_none(
            result.get("weak_pronunciation_naturalness_score"),
            weak_details.get("weak_pronunciation_naturalness_score"),
            result.get("pronunciation_score"),
        )
        weak_rhythm = _first_not_none(
            result.get("weak_rhythm_naturalness_score"),
            weak_details.get("weak_rhythm_naturalness_score"),
            rhythm,
        )
        weak_pitch = _first_not_none(
            result.get("weak_prosody_naturalness_score"),
            weak_details.get("weak_prosody_naturalness_score"),
        )
        scores = {
            "mora_clarity_score": weak_pronunciation,
            "rhythm_timing_score": weak_rhythm,
            "delivery_fluency_score": fluency_details.get("delivery_fluency_score", fluency),
            "pitch_score": weak_pitch,
        }
        display = _weighted_available(scores, {
            "mora_clarity_score": 0.30,
            "rhythm_timing_score": 0.20,
            "delivery_fluency_score": 0.25,
            "pitch_score": 0.25,
        })
        if display is None:
            return None
        if weak_guardrail.get("status") == "capped" and weak_guardrail.get("cap") is not None:
            display = min(display, _as_score(weak_guardrail.get("cap")))
        return int(round(display))
    if not gate.allow_pitch_feedback:
        display = _weighted_available(scores, {
            "mora_clarity_score": 0.40,
            "rhythm_timing_score": 0.30,
            "delivery_fluency_score": 0.30,
        })
        if display is None:
            display = 0.55 * pronunciation + 0.35 * fluency + 0.10 * prosody
        return int(round(display))
    display = _weighted_available(scores, {
        "mora_clarity_score": 0.30,
        "rhythm_timing_score": 0.20,
        "delivery_fluency_score": 0.25,
        "pitch_score": 0.25,
    })
    if display is not None:
        return int(round(display))
    return int(round(0.55 * pronunciation + 0.35 * fluency + 0.10 * prosody))


def _mode_notice(policy: ScoringPolicy, gate: Any) -> str:
    if policy.demo_only and "kanade" in policy.mode:
        return user_message("notice.kanade")
    if policy.weak_reference:
        return user_message("notice.weak_reference")
    if gate.allow_pitch_feedback:
        return user_message("notice.fixed_verified")
    return user_message("notice.fixed_limited")


def _status(policy: ScoringPolicy, gate: Any, focus: Optional[Dict[str, Any]]) -> str:
    if gate.practice_check_result == "retry":
        return "retry"
    if policy.demo_only:
        return "debug_only"
    if policy.weak_reference and "confirmed" not in policy.mode:
        return "debug_only"
    if gate.reliability == "low" and not policy.weak_reference:
        reasons = {str(reason) for reason in (gate.reasons or [])}
        if {"fallback_alignment", "alignment_confidence_low"}.intersection(reasons):
            return "practice_suggestion"
        return "debug_only"
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


def _visible_dimension_contract(
    result: Mapping[str, Any],
    policy: ScoringPolicy,
    gate: Any,
    *,
    display_score: Optional[int],
    special_mora_score: Optional[float],
    debug: Mapping[str, Any],
) -> tuple[Dict[str, Optional[int]], Dict[str, str]]:
    names = ("pronunciation", "rhythm", "fluency", "pitch")
    if display_score is None:
        return ({name: None for name in names}, {name: "unavailable" for name in names})

    def visible_int(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        return int(round(_as_score(value)))

    degraded_reference_practice = bool(debug.get("degraded_reference_practice"))
    weak = bool(policy.weak_reference or degraded_reference_practice)
    pronunciation = debug.get("weak_pronunciation_naturalness_score") if weak else debug.get("pronunciation_score")
    rhythm = debug.get("weak_rhythm_naturalness_score") if weak else debug.get("rhythm_score")
    if rhythm is None:
        rhythm = debug.get("rhythm_score")
    if rhythm is None:
        rhythm = debug.get("rhythm_timing_score")
    pitch = debug.get("visible_prosody_score")
    values = {
        "pronunciation": visible_int(pronunciation),
        "rhythm": visible_int(rhythm),
        "fluency": visible_int(debug.get("fluency_score")),
        "pitch": visible_int(pitch),
    }

    reasons = set(str(reason) for reason in (gate.reasons or []))
    base = "high" if gate.reliability == "high" else "medium" if gate.reliability == "medium" else "low"
    confidence = {name: base for name in names}
    alignment_mode = str(result.get("alignment_mode") or "")
    rhythm_practice = debug.get("rhythm_practice") if isinstance(debug.get("rhythm_practice"), Mapping) else {}
    if weak:
        # Weak-reference dimensions are practice proxies. High acoustic
        # reliability must not be presented as teacher-grade correctness.
        confidence["pronunciation"] = "medium" if values["pronunciation"] is not None else "unavailable"
        confidence["pitch"] = "medium" if values["pitch"] is not None else "unavailable"
        confidence["rhythm"] = str(rhythm_practice.get("confidence") or "medium")
    if degraded_reference_practice:
        confidence = {
            name: "low" if values[name] is not None else "unavailable"
            for name in names
        }
    if "fallback" in alignment_mode or alignment_mode in {"equal", "equal_fallback"}:
        confidence["pronunciation"] = "low"
        confidence["rhythm"] = "low"
    if {"fallback_alignment", "alignment_confidence_low"}.intersection(reasons):
        confidence["pronunciation"] = "low"
        confidence["rhythm"] = "low"
        confidence["pitch"] = "low"
    if "low_f0_coverage" in reasons:
        confidence["pitch"] = "low"
    if not gate.allow_special_mora_feedback:
        confidence["rhythm"] = "low" if "fallback_alignment" in reasons else "medium"
    for name, value in values.items():
        if value is None:
            confidence[name] = "unavailable"
    return values, confidence


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
    candidates = build_feedback_candidates(
        result,
        policy,
        gate,
        special_mora_decisions=decision_dicts,
        max_candidates=2,
    )
    candidate_dicts = [item.to_dict() for item in candidates]
    messages: List[str] = []
    focus: Optional[Dict[str, Any]] = None

    if policy.demo_only:
        messages.append("このモードは参考音のデモです。発音の正しさ判定には使いません。")
        focus = {"category": "demo", "message": messages[-1]}
    elif candidate_dicts:
        focus = candidate_dicts[0]
        messages.append(str(focus.get("user_message") or ""))
        tip = str(focus.get("practice_tip") or "")
        if tip:
            messages.append(f"下一次先练一个点：{tip}")

    raw_feedback = [str(item) for item in (result.get("feedback") or [])]
    if not candidate_dicts and not policy.demo_only:
        messages.extend(gate.messages)
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
    display_score = _display_score(result, policy, gate, special_mora_score=special_mora_score)
    status = _status(policy, gate, focus)
    mode_notice = _mode_notice(policy, gate)
    primary = None
    suggestion_type = "none"
    if focus and focus.get("category") not in {"demo", "weak_reference"}:
        primary = str(focus.get("practice_tip") or focus.get("message") or "")
        suggestion_type = str(focus.get("category") or "none")
    elif len(messages) > 1 and status == "practice_suggestion":
        primary = messages[1]
    practice_score = PracticeScore(
        value=display_score if status != "debug_only" else None,
        label=practice_score_label(display_score if status != "debug_only" else None, status),
        explanation=practice_score_explanation(mode_notice),
    )

    debug_payload = _debug_payload(
        result,
        policy,
        gate,
        special_mora_decisions=decision_dicts,
        special_mora_score=special_mora_score,
        special_mora_profile=(decision_dicts[0].get("evidence_card", {}) if decision_dicts else {}),
    )
    dimension_scores, dimension_confidence = _visible_dimension_contract(
        result,
        policy,
        gate,
        display_score=display_score,
        special_mora_score=special_mora_score,
        debug=debug_payload,
    )
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
        dimension_scores=dimension_scores,
        dimension_confidence=dimension_confidence,
        user_messages=messages[:2],
        focus_feedback=focus,
        feedback_candidates=candidate_dicts,
        display_total_score=False,
        debug=debug_payload,
    ).to_dict()
