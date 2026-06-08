from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def _pick(mapping: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _score(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(100.0, number))


def _cap(value: Optional[float], maximum: float) -> Optional[float]:
    if value is None:
        return None
    return min(value, maximum)


def _round_score(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    return int(round(max(0.0, min(100.0, value))))


def _confidence_at_most(current: str, maximum: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    labels = {0: "low", 1: "medium", 2: "high"}
    return labels[min(order.get(current, 1), order.get(maximum, 1))]


def _is_user_facing_special_mora(decision: Mapping[str, Any]) -> bool:
    if not bool(decision.get("user_feedback_allowed")):
        return False
    mora_type = str(decision.get("type") or "")
    return mora_type in {"long_vowel", "moraic_nasal"}


def apply_user_score_policy(
    raw_result: Mapping[str, Any],
    *,
    mode: str = "fixed_reference",
    special_mora_decisions: Optional[List[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute conservative learner-facing scores from proxy metrics.

    This layer does not change the acoustic evaluator. It only prevents the
    consumer UI from showing a high overall number when the pronunciation
    evidence itself is weak, fallback-based, or only weak-reference.
    """

    details = raw_result.get("details") if isinstance(raw_result.get("details"), Mapping) else {}
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    content = details.get("content_match") if isinstance(details.get("content_match"), Mapping) else {}
    alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}
    recording = details.get("recording_quality") if isinstance(details.get("recording_quality"), Mapping) else {}
    fluency_details = details.get("fluency") if isinstance(details.get("fluency"), Mapping) else {}

    warnings: List[str] = []
    score_caps: Dict[str, Any] = {}
    detail_feedback_allowed = True
    score_available = True
    gate_state: Dict[str, Any] = {
        "recording_ok": True,
        "target_match_ok": True,
        "alignment_ok": True,
        "pronunciation_evidence_ok": True,
        "score_available": True,
        "user_message_type": "",
        "reasons": [],
    }

    pronunciation = _score(raw_result.get("pronunciation_score"))
    rhythm = _score(fluency_details.get("rhythm_timing_score", raw_result.get("prosody_score")))
    fluency = _score(fluency_details.get("delivery_fluency_score", raw_result.get("fluency_score")))

    content_status = str(content.get("status") or "unknown")
    alignment_mode = str(raw_result.get("alignment_mode") or alignment.get("mode") or "")
    weak_reference = bool(details.get("weak_reference")) or mode in {
        "asr_confirmed_weak_reference",
        "asr_pseudo_reference",
        "kanade_asr_voice_reference",
    }
    demo_only = bool(details.get("demo_only")) or str(mode).startswith("kanade")
    mora_count = len(raw_result.get("moras") or [])

    reliability_level = str(reliability.get("level") or "medium")
    reliability_overall = float(reliability.get("overall", 0.0) or 0.0)
    alignment_confidence = float(reliability.get("alignment", 1.0) or 0.0)
    recording_score = float(recording.get("score", 1.0) or 1.0)
    confidence_label = "high" if reliability_overall >= 0.85 and reliability_level != "low" else "medium" if reliability_overall >= 0.45 else "low"

    if demo_only:
        warnings.append("demo_only_no_pronunciation_score")
        gate_state.update({
            "pronunciation_evidence_ok": False,
            "score_available": False,
            "user_message_type": "demo_only",
            "reasons": ["demo_only_no_pronunciation_score"],
        })
        return {
            "display_score": None,
            "pronunciation_clarity_score": None,
            "rhythm_fluency_score": _round_score((rhythm + fluency) / 2.0),
            "practice_completion_score": None,
            "confidence_label": "low",
            "main_message_key": "demo_only_no_pronunciation_score",
            "score_policy_warnings": warnings,
            "score_caps": score_caps,
            "score_available": False,
            "scoring_gate": gate_state,
            "detail_feedback_allowed": False,
            "inputs": {
                "mode": mode,
                "weak_reference": weak_reference,
                "demo_only": demo_only,
                "raw_pronunciation_score": raw_result.get("pronunciation_score"),
                "raw_total_score": raw_result.get("total_score"),
            },
        }

    if content_status in {"fail", "failed", "content_mismatch"}:
        warnings.append("content_match_failed_no_pronunciation_score")
        gate_state.update({
            "target_match_ok": False,
            "pronunciation_evidence_ok": False,
            "score_available": False,
            "user_message_type": "content_mismatch",
            "reasons": ["content_match_failed_no_pronunciation_score"],
        })
        return {
            "display_score": None,
            "pronunciation_clarity_score": None,
            "rhythm_fluency_score": None,
            "practice_completion_score": None,
            "confidence_label": "low",
            "main_message_key": "content_match_failed_no_pronunciation_score",
            "score_policy_warnings": warnings,
            "score_caps": score_caps,
            "score_available": False,
            "scoring_gate": gate_state,
            "detail_feedback_allowed": False,
            "inputs": {
                "mode": mode,
                "content_status": content_status,
                "raw_total_score": raw_result.get("total_score"),
            },
        }

    if special_mora_decisions and any(_is_user_facing_special_mora(item) for item in special_mora_decisions):
        penalty = 6.0
        pronunciation = max(0.0, pronunciation - penalty)
        warnings.append("special_mora_soft_penalty")
        score_caps["special_mora_soft_penalty"] = -penalty

    display = 0.70 * pronunciation + 0.20 * rhythm + 0.10 * fluency
    display = min(display, pronunciation + 8.0)
    pronunciation_clarity = pronunciation

    if pronunciation < 50:
        display = _cap(display, 60.0)
        warnings.append("pronunciation_under_50_display_cap")
        score_caps["pronunciation_under_50_display_cap"] = 60
    elif pronunciation < 60:
        display = _cap(display, 68.0)
        warnings.append("pronunciation_under_60_display_cap")
        score_caps["pronunciation_under_60_display_cap"] = 68
    elif pronunciation < 70:
        display = _cap(display, 78.0)
        warnings.append("pronunciation_under_70_display_cap")
        score_caps["pronunciation_under_70_display_cap"] = 78

    if alignment_mode.endswith("fallback_equal") or "fallback" in alignment_mode:
        pronunciation_clarity = None
        display = None
        detail_feedback_allowed = False
        score_available = False
        warnings.append("alignment_fallback_cap")
        warnings.append("alignment_fallback_no_display_score")
        score_caps["fallback_pronunciation_score"] = None
        score_caps["fallback_display_score"] = None
        confidence_label = _confidence_at_most(confidence_label, "medium")
        gate_state["alignment_ok"] = False
        gate_state["pronunciation_evidence_ok"] = False
        gate_state["score_available"] = False
        gate_state["user_message_type"] = "alignment_limited"
        gate_state["reasons"].append("alignment_fallback_no_display_score")

    if reliability_level == "low" or alignment_confidence < 0.45:
        pronunciation_clarity = None
        display = None
        detail_feedback_allowed = False
        score_available = False
        warnings.append("low_alignment_cap")
        warnings.append("low_alignment_no_display_score")
        score_caps["low_alignment_pronunciation_score"] = None
        score_caps["low_alignment_display_score"] = None
        confidence_label = _confidence_at_most(confidence_label, "low")
        gate_state["alignment_ok"] = False
        gate_state["pronunciation_evidence_ok"] = False
        gate_state["score_available"] = False
        gate_state["user_message_type"] = "alignment_low"
        gate_state["reasons"].append("low_alignment_no_display_score")

    if score_available and content_status in {"marginal", "partial"}:
        pronunciation_clarity = _cap(pronunciation_clarity, 70.0) or 0.0
        display = _cap(display, 75.0)
        warnings.append("content_marginal_cap")
        score_caps["content_marginal_pronunciation_cap"] = 70
        score_caps["content_marginal_display_cap"] = 75
        confidence_label = _confidence_at_most(confidence_label, "medium")

    if mora_count <= 4:
        display = _cap(display, 80.0)
        detail_feedback_allowed = False
        warnings.append("short_sentence_cap")
        score_caps["short_sentence_display_cap"] = 80

    if weak_reference:
        display = None
        pronunciation_clarity = _cap(pronunciation_clarity, 80.0) or 0.0
        detail_feedback_allowed = False
        score_available = False
        confidence_label = _confidence_at_most(confidence_label, "medium")
        warnings.append("weak_reference_no_display_score")
        score_caps["weak_reference_display_score"] = None
        score_caps["weak_reference_pronunciation_cap"] = 80
        gate_state["pronunciation_evidence_ok"] = False
        gate_state["score_available"] = False
        gate_state["user_message_type"] = "weak_reference"
        gate_state["reasons"].append("weak_reference_no_display_score")

    if recording_score < 0.55:
        display = None
        pronunciation_clarity = None
        detail_feedback_allowed = False
        score_available = False
        confidence_label = _confidence_at_most(confidence_label, "low")
        warnings.append("recording_quality_no_display_score")
        score_caps["recording_quality_display_score"] = None
        gate_state["recording_ok"] = False
        gate_state["pronunciation_evidence_ok"] = False
        gate_state["score_available"] = False
        gate_state["user_message_type"] = "recording_bad"
        gate_state["reasons"].append("recording_quality_no_display_score")

    practice_completion: Optional[float]
    if content_status in {"pass", "unknown"}:
        content_completion = 100.0
    elif content_status in {"marginal", "partial"}:
        content_completion = 70.0
    else:
        content_completion = 50.0
    practice_completion = 0.45 * content_completion + 0.35 * _score(recording_score * 100.0) + 0.20 * _score(reliability_overall * 100.0)

    main_message_key = ""
    if weak_reference:
        main_message_key = "weak_reference_practice_feedback"
    if "alignment_fallback_cap" in warnings:
        main_message_key = "alignment_limited_score_cap"
    if "short_sentence_cap" in warnings and not main_message_key:
        main_message_key = "short_sentence_overall_only"
    if recording_score >= 0.75 and pronunciation_clarity is not None and pronunciation_clarity < 70:
        main_message_key = "clear_recording_but_pronunciation_needs_practice"

    return {
        "display_score": _round_score(display),
        "pronunciation_clarity_score": _round_score(pronunciation_clarity),
        "rhythm_fluency_score": _round_score((rhythm + fluency) / 2.0),
        "practice_completion_score": _round_score(practice_completion),
        "confidence_label": confidence_label,
        "main_message_key": main_message_key,
        "score_policy_warnings": warnings,
        "score_caps": score_caps,
        "score_available": score_available,
        "scoring_gate": gate_state,
        "detail_feedback_allowed": detail_feedback_allowed,
        "inputs": {
            "mode": mode,
            "weak_reference": weak_reference,
            "demo_only": demo_only,
            "content_status": content_status,
            "alignment_mode": alignment_mode,
            "mora_count": mora_count,
            "recording_score": recording_score,
            "raw_total_score": raw_result.get("total_score"),
            "raw_pronunciation_score": raw_result.get("pronunciation_score"),
            "rhythm_source_score": rhythm,
            "fluency_source_score": fluency,
            "reliability_level": reliability_level,
            "reliability_overall": reliability_overall,
            "alignment_confidence": alignment_confidence,
        },
    }
