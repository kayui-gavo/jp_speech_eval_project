from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def _score(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(100.0, number))


def _unit_score(value: Any, default: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


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
    return str(decision.get("type") or "") in {"long_vowel", "moraic_nasal", "sokuon"}


def _smooth_product_score(pronunciation: float, rhythm: float, fluency: float) -> float:
    """Continuous consumer-facing score mapping without discrete caps."""
    raw = 0.58 * pronunciation + 0.22 * rhythm + 0.20 * fluency
    centered = 70.0 + 1.10 * (raw - 70.0)
    return max(0.0, min(100.0, centered))


def apply_user_score_policy(
    raw_result: Mapping[str, Any],
    *,
    mode: str = "fixed_reference",
    special_mora_decisions: Optional[List[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute product-facing practice scores with graceful degradation."""
    details = raw_result.get("details") if isinstance(raw_result.get("details"), Mapping) else {}
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    content = details.get("content_match") if isinstance(details.get("content_match"), Mapping) else {}
    alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}
    recording = details.get("recording_quality") if isinstance(details.get("recording_quality"), Mapping) else {}
    fluency_details = details.get("fluency") if isinstance(details.get("fluency"), Mapping) else {}
    sanity = details.get("transcript_sanity") if isinstance(details.get("transcript_sanity"), Mapping) else {}

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

    pronunciation = _score(raw_result.get("pronunciation_score"), 60.0)
    rhythm = _score(fluency_details.get("rhythm_timing_score", raw_result.get("prosody_score")), 65.0)
    fluency = _score(fluency_details.get("delivery_fluency_score", raw_result.get("fluency_score")), 65.0)

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
    reliability_overall = _unit_score(reliability.get("overall"), default=0.0)
    alignment_confidence = _unit_score(reliability.get("alignment"), default=1.0)
    recording_value = recording.get("score")
    if recording_value is None:
        recording_value = reliability.get("recording_quality")
    recording_score = _unit_score(recording_value, default=1.0)
    confidence_label = "high" if reliability_overall >= 0.85 and reliability_level != "low" else "medium" if reliability_overall >= 0.45 else "low"

    sanity_ok = sanity.get("ok")
    if sanity_ok is False:
        warnings.append("transcript_sanity_failed_no_score")
        gate_state.update({
            "target_match_ok": False,
            "pronunciation_evidence_ok": False,
            "score_available": False,
            "user_message_type": "invalid_or_non_japanese",
            "reasons": ["transcript_sanity_failed_no_score"],
        })
        return {
            "display_score": None,
            "display_score_before_cap": None,
            "display_score_after_cap": None,
            "display_cap_applied": False,
            "display_cap_reason": "",
            "pronunciation_clarity_score": None,
            "rhythm_fluency_score": None,
            "practice_completion_score": None,
            "confidence_label": "low",
            "main_message_key": "invalid_or_non_japanese",
            "score_policy_warnings": warnings,
            "score_caps": score_caps,
            "score_available": False,
            "scoring_gate": gate_state,
            "detail_feedback_allowed": False,
            "inputs": {"mode": mode, "content_status": content_status},
        }

    if demo_only:
        gate_state.update({
            "score_available": False,
            "user_message_type": "demo_only",
            "reasons": ["demo_only_no_practice_score"],
        })
        return {
            "display_score": None,
            "display_score_before_cap": None,
            "display_score_after_cap": None,
            "display_cap_applied": False,
            "display_cap_reason": "",
            "pronunciation_clarity_score": None,
            "rhythm_fluency_score": None,
            "practice_completion_score": None,
            "confidence_label": "low",
            "main_message_key": "",
            "score_policy_warnings": ["demo_only_no_practice_score"],
            "score_caps": score_caps,
            "score_available": False,
            "scoring_gate": gate_state,
            "detail_feedback_allowed": False,
            "inputs": {"mode": mode, "demo_only": True},
        }

    if recording_score < 0.20:
        warnings.append("recording_unusable_no_score")
        gate_state.update({
            "recording_ok": False,
            "pronunciation_evidence_ok": False,
            "score_available": False,
            "user_message_type": "recording_bad",
            "reasons": ["recording_unusable_no_score"],
        })
        return {
            "display_score": None,
            "display_score_before_cap": None,
            "display_score_after_cap": None,
            "display_cap_applied": False,
            "display_cap_reason": "",
            "pronunciation_clarity_score": None,
            "rhythm_fluency_score": None,
            "practice_completion_score": None,
            "confidence_label": "low",
            "main_message_key": "recording_unusable_no_score",
            "score_policy_warnings": warnings,
            "score_caps": score_caps,
            "score_available": False,
            "scoring_gate": gate_state,
            "detail_feedback_allowed": False,
            "inputs": {"mode": mode, "recording_score": recording_score},
        }

    if special_mora_decisions and any(_is_user_facing_special_mora(item) for item in special_mora_decisions):
        pronunciation = max(0.0, pronunciation - 4.0)
        warnings.append("special_mora_soft_penalty")

    display = _smooth_product_score(pronunciation, rhythm, fluency)
    display_score_before_cap = display
    pronunciation_clarity: Optional[float] = pronunciation

    if content_status in {"fail", "failed", "content_mismatch"}:
        warnings.append("target_content_mismatch_general_score")
        gate_state["target_match_ok"] = False
        gate_state["user_message_type"] = "content_mismatch_general_score"
        gate_state["reasons"].append("target_content_mismatch_general_score")
        detail_feedback_allowed = False
        confidence_label = _confidence_at_most(confidence_label, "medium")

    if alignment_mode.endswith("fallback_equal") or "fallback" in alignment_mode:
        warnings.append("alignment_fallback_broad_score_only")
        detail_feedback_allowed = False
        gate_state["alignment_ok"] = False
        gate_state["pronunciation_evidence_ok"] = False
        gate_state["reasons"].append("alignment_fallback_broad_score_only")
        confidence_label = _confidence_at_most(confidence_label, "medium")

    if reliability_level == "low" or alignment_confidence < 0.35:
        warnings.append("low_alignment_broad_score_only")
        detail_feedback_allowed = False
        gate_state["alignment_ok"] = False
        gate_state["pronunciation_evidence_ok"] = False
        gate_state["reasons"].append("low_alignment_broad_score_only")
        confidence_label = _confidence_at_most(confidence_label, "low")

    if weak_reference:
        warnings.append("weak_reference_broad_score_only")
        detail_feedback_allowed = False
        gate_state["pronunciation_evidence_ok"] = False
        gate_state["reasons"].append("weak_reference_broad_score_only")
        confidence_label = _confidence_at_most(confidence_label, "medium")

    if recording_score < 0.55:
        warnings.append("recording_quality_low_confidence_score")
        detail_feedback_allowed = False
        gate_state["recording_ok"] = False
        gate_state["reasons"].append("recording_quality_low_confidence_score")
        confidence_label = _confidence_at_most(confidence_label, "low")

    if mora_count <= 3:
        warnings.append("short_utterance_broad_score_only")
        detail_feedback_allowed = False

    if content_status in {"pass", "unknown", "general_japanese"}:
        content_completion = 100.0
    elif content_status in {"marginal", "partial", "uncertain"}:
        content_completion = 80.0
    else:
        content_completion = 65.0
    practice_completion = 0.50 * content_completion + 0.30 * _score(recording_score * 100.0) + 0.20 * _score(reliability_overall * 100.0)

    main_message_key = ""
    if content_status in {"fail", "failed", "content_mismatch"}:
        main_message_key = "content_mismatch_general_score"
    elif weak_reference:
        main_message_key = "weak_reference_practice_feedback"
    elif not detail_feedback_allowed:
        main_message_key = "broad_score_only"
    elif recording_score >= 0.75 and pronunciation_clarity is not None and pronunciation_clarity < 70:
        main_message_key = "clear_recording_but_pronunciation_needs_practice"

    return {
        "display_score": _round_score(display),
        "display_score_before_cap": _round_score(display_score_before_cap),
        "display_score_after_cap": _round_score(display),
        "display_cap_applied": False,
        "display_cap_reason": "",
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
