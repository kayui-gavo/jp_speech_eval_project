from __future__ import annotations

import math
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence


FIXED_REFERENCE_MODES = {
    "reference",
    "reference_based",
    "reference_fixed_sentence",
    "fixed_reference",
}

WEAK_REFERENCE_MODES = {
    "asr_confirmed_weak_reference",
    "asr_pseudo_reference",
    "kanade_asr_voice_reference",
    "kanade_asr_confirmed_voice_reference",
}

TARGET_MATCH_STATUSES = {"pass", "uncertain", "marginal", "partial"}
TARGET_MISMATCH_STATUSES = {"fail", "failed", "content_mismatch"}


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _clip01(value: Any, default: float = 0.0) -> float:
    number = _number(value)
    if number is None:
        number = default
    return max(0.0, min(1.0, number))


def _score(value: Any) -> Optional[int]:
    number = _number(value)
    if number is None:
        return None
    return int(round(max(0.0, min(100.0, number))))


def _blend(parts: Sequence[tuple[Optional[float], float]]) -> Optional[float]:
    usable = [(float(value), float(weight)) for value, weight in parts if value is not None and weight > 0]
    if not usable:
        return None
    denom = sum(weight for _value, weight in usable)
    return sum(value * weight for value, weight in usable) / max(denom, 1e-8)


def _duration_match_score(ratio: Any) -> Optional[float]:
    value = _number(ratio)
    if value is None or value <= 0:
        return None
    return 100.0 * math.exp(-0.80 * abs(math.log(value)))


def _dimension(
    key: str,
    label: str,
    value: Any,
    *,
    source_field: str,
    construct: str,
    available: bool,
    confidence: str = "unknown",
    evidence_tier: str = "unknown",
    note: str = "",
) -> Dict[str, Any]:
    score = _score(value) if available else None
    return {
        "key": key,
        "label": label,
        "value": score,
        "available": score is not None,
        "source_field": source_field,
        "construct": construct,
        "confidence": confidence,
        "evidence_tier": evidence_tier,
        "product_calibrated": False,
        "note": note,
    }


def _target_relative_allowed(details: Mapping[str, Any]) -> bool:
    content = details.get("content_match") if isinstance(details.get("content_match"), Mapping) else {}
    status = str(content.get("status") or "unknown")
    return status in TARGET_MATCH_STATUSES


def _alignment_state(
    result: Mapping[str, Any],
    details: Mapping[str, Any],
) -> tuple[bool, bool, str]:
    """Resolve alignment availability from both legacy and current fields.

    Some evaluators historically wrote the fallback state only to the top-level
    ``alignment_mode`` while ``details.alignment`` still looked nominal. A C-end
    dimension must not treat equal-segmentation fallback as trustworthy local
    alignment merely because the nested legacy object is stale.
    """
    alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}
    mode = str(result.get("alignment_mode") or alignment.get("mode") or "")
    mode_lower = mode.lower()
    fallback = bool(alignment.get("used_equal_fallback")) or "fallback" in mode_lower
    available = bool(alignment.get("available", True)) and not fallback
    return available, fallback, mode


def _mapped_clarity_evidence(details: Mapping[str, Any]) -> tuple[Optional[float], str, str]:
    shadow = details.get("shadow") if isinstance(details.get("shadow"), Mapping) else {}
    ssl = shadow.get("ssl_pronunciation") if isinstance(shadow.get("ssl_pronunciation"), Mapping) else {}
    if bool(ssl.get("score_mapped")):
        value = _number(ssl.get("mapped_score", ssl.get("value")))
        if value is not None:
            return value, "details.shadow.ssl_pronunciation.mapped_score", "mapped_ssl_phonetic_clarity_evidence"

    pronunciation_evidence = (
        details.get("pronunciation_evidence")
        if isinstance(details.get("pronunciation_evidence"), Mapping)
        else {}
    )
    if bool(pronunciation_evidence.get("score_mapped")):
        value = _number(pronunciation_evidence.get("mapped_score", pronunciation_evidence.get("value")))
        if value is not None:
            return value, "details.pronunciation_evidence.mapped_score", "mapped_pronunciation_clarity_evidence"

    return None, "", ""


def _clarity_proxy(
    result: Mapping[str, Any],
    details: Mapping[str, Any],
    *,
    allow_target_relative: bool,
) -> tuple[float, str, str, str, str]:
    mapped, mapped_source, mapped_construct = _mapped_clarity_evidence(details)
    if mapped is not None:
        return mapped, mapped_source, mapped_construct, "medium", "mapped_pronunciation_evidence"

    if not allow_target_relative:
        return (
            70.0,
            "product_prior",
            "broad_clarity_prior_without_target_independent_measurement",
            "low",
            "reference_independent_prior_fallback",
        )

    content = details.get("content_match") if isinstance(details.get("content_match"), Mapping) else {}
    kana_similarity = _number(content.get("kana_similarity"))
    acoustic_score = _number(content.get("score"))
    content_status = str(content.get("status") or "unknown")

    if kana_similarity is not None and kana_similarity > 0:
        asr_component = 45.0 + 50.0 * _clip01(kana_similarity)
        acoustic_component = None if acoustic_score is None else 55.0 + 35.0 * _clip01(acoustic_score)
        value = _blend([(asr_component, 0.82), (acoustic_component, 0.18)]) or asr_component
        confidence = "medium" if content_status in {"pass", "uncertain"} else "low"
        return (
            value,
            "details.content_match.kana_similarity",
            "asr_machine_intelligibility_plus_acoustic_support",
            confidence,
            "asr_target_agreement",
        )

    if acoustic_score is not None:
        value = 55.0 + 35.0 * _clip01(acoustic_score)
        return (
            value,
            "details.content_match.score",
            "reference_relative_acoustic_clarity_proxy",
            "low",
            "acoustic_content_match",
        )

    alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}
    dtw_cost = _number(alignment.get("normalized_dtw_cost"))
    alignment_available, _alignment_fallback, _alignment_mode = _alignment_state(result, details)
    if dtw_cost is not None and alignment_available:
        similarity = max(0.0, min(1.0, 1.0 - (dtw_cost - 3.4) / 1.8))
        value = 55.0 + 35.0 * similarity
        return (
            value,
            "details.alignment.normalized_dtw_cost",
            "reference_relative_mfcc_clarity_proxy",
            "low",
            "mfcc_reference_similarity",
        )

    if content_status == "pass":
        return (
            85.0,
            "details.content_match.status",
            "passed_target_content_gate_broad_machine_intelligibility_support",
            "low",
            "content_gate_pass_fallback",
        )
    if content_status in {"uncertain", "marginal", "partial"}:
        return (
            76.0,
            "details.content_match.status",
            "partial_target_content_gate_broad_machine_intelligibility_support",
            "low",
            "content_gate_partial_fallback",
        )

    return 70.0, "product_prior", "broad_clarity_prior_without_independent_measurement", "low", "prior_fallback"


def _corr(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 3 or len(b) < 3:
        return None
    ma = mean(a)
    mb = mean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 1e-10 or vb <= 1e-10:
        return None
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(va * vb)


def _z(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / max(len(values), 1)
    sd = math.sqrt(var)
    if sd <= 1e-8:
        return [0.0 for _ in values]
    return [(x - m) / sd for x in values]


def _f0_pair_fallback(result: Mapping[str, Any], details: Mapping[str, Any]) -> tuple[Optional[float], str, str]:
    rows = result.get("mora_table") if isinstance(result.get("mora_table"), list) else []
    ref = details.get("reference_f0_by_mora") if isinstance(details.get("reference_f0_by_mora"), list) else []
    user_values: list[float] = []
    ref_values: list[float] = []
    for index in range(min(len(rows), len(ref))):
        row = rows[index] if isinstance(rows[index], Mapping) else {}
        user_f0 = _number(row.get("f0_hz"))
        ref_f0 = _number(ref[index])
        if user_f0 is None or ref_f0 is None or user_f0 <= 0 or ref_f0 <= 0:
            continue
        user_values.append(math.log(user_f0))
        ref_values.append(math.log(ref_f0))

    if len(user_values) >= 3:
        uz = _z(user_values)
        rz = _z(ref_values)
        corr = _corr(uz, rz)
        rmse = math.sqrt(sum((u - r) ** 2 for u, r in zip(uz, rz)) / len(uz))
        corr_score = 0.5 if corr is None else (max(-1.0, min(1.0, corr)) + 1.0) / 2.0
        rmse_score = math.exp(-0.55 * rmse)
        value = 100.0 * (0.72 * corr_score + 0.28 * rmse_score)
        return value, "paired_mora_f0_fallback", "medium"

    if len(user_values) == 2:
        user_delta = user_values[1] - user_values[0]
        ref_delta = ref_values[1] - ref_values[0]
        if abs(ref_delta) < 0.04 and abs(user_delta) < 0.04:
            return 78.0, "two_point_f0_direction_fallback", "low"
        same_direction = user_delta * ref_delta > 0
        magnitude_gap = abs(abs(user_delta) - abs(ref_delta))
        value = (80.0 if same_direction else 58.0) - min(12.0, 20.0 * magnitude_gap)
        return value, "two_point_f0_direction_fallback", "low"

    return None, "", "low"


def _legacy_phrase_intonation_fallback(prosody: Mapping[str, Any]) -> tuple[Optional[float], str]:
    """Recover phrase-level evidence from older result schemas without lexical accent."""
    final_score = _number(prosody.get("final_intonation_score"))
    contour_corr = _number(prosody.get("contour_corr"))
    transition = _number(prosody.get("transition_agreement"))

    parts: list[tuple[Optional[float], float]] = []
    if contour_corr is not None:
        corr_score = 50.0 * (max(-1.0, min(1.0, contour_corr)) + 1.0)
        parts.append((corr_score, 0.72))
    if transition is not None:
        transition_score = 100.0 * _clip01(transition)
        parts.append((transition_score, 0.28))
    if parts:
        return _blend(parts), "legacy_phrase_contour_fields_fallback"
    if final_score is not None:
        return final_score, "legacy_final_intonation_score_fallback"
    return None, ""


def _intonation_proxy(
    result: Mapping[str, Any],
    details: Mapping[str, Any],
    *,
    mode: str,
    allow_reference_relative: bool,
) -> tuple[float, str, str, str, str]:
    prosody = details.get("prosody") if isinstance(details.get("prosody"), Mapping) else {}
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    weak_reference = bool(details.get("weak_reference")) or mode in WEAK_REFERENCE_MODES
    raw = _number(result.get("prosody_score"))
    note = str(prosody.get("note") or "")
    contour_corr = _number(prosody.get("contour_corr"))
    valid_mora = int(_number(prosody.get("contour_valid_mora_count", prosody.get("valid_mora_count", 0))) or 0)
    f0_coverage = _clip01(reliability.get("f0_coverage"), default=0.0)
    alignment_available, alignment_fallback, _alignment_mode = _alignment_state(result, details)

    if (
        allow_reference_relative
        and alignment_available
        and raw is not None
        and note not in {"no_valid_f0", "insufficient_valid_mora_f0"}
        and contour_corr is not None
        and valid_mora >= 3
    ):
        confidence = "high" if f0_coverage >= 0.65 and not weak_reference else "medium"
        value = raw
        if weak_reference:
            value = 72.0 + 0.60 * (value - 72.0)
            confidence = "low"
        return (
            value,
            "prosody_score",
            "reference_relative_normalized_f0_contour_similarity",
            confidence,
            "mora_contour_primary",
        )

    if allow_reference_relative and alignment_available:
        legacy_phrase, legacy_source = _legacy_phrase_intonation_fallback(prosody)
        if legacy_phrase is not None and note not in {"no_valid_f0"}:
            confidence = "low" if weak_reference else "medium"
            if weak_reference:
                legacy_phrase = 72.0 + 0.55 * (legacy_phrase - 72.0)
            return (
                legacy_phrase,
                f"details.prosody.{legacy_source}",
                "reference_relative_phrase_intonation_fallback_without_lexical_pitch_accent",
                confidence,
                "phrase_intonation_semantic_fallback",
            )

        fallback, fallback_source, fallback_conf = _f0_pair_fallback(result, details)
        if fallback is not None:
            if weak_reference:
                fallback = 72.0 + 0.55 * (fallback - 72.0)
                fallback_conf = "low"
            return (
                fallback,
                fallback_source,
                "partial_reference_relative_f0_contour_similarity",
                fallback_conf,
                "partial_f0_fallback",
            )

    tone = details.get("tone") if isinstance(details.get("tone"), Mapping) else {}
    pitch_range = _number(tone.get("pitch_range_log"))
    pitch_score = _number(tone.get("pitch_score"))
    if pitch_range is not None and pitch_score is not None:
        value = 72.0 + 0.55 * (pitch_score - 72.0)
        source = "details.tone.pitch_score"
        tier = "pitch_range_fallback"
        if alignment_fallback:
            source += "+alignment_fallback"
            tier = "alignment_fallback_pitch_range_proxy"
        return (
            value,
            source,
            "reference_independent_pitch_movement_naturalness_proxy",
            "low",
            tier,
        )

    tier = "alignment_fallback_prior" if alignment_fallback else "prior_fallback"
    return 70.0, "product_prior", "broad_intonation_prior_without_reliable_f0", "low", tier


def build_consumer_score_components(
    result: Mapping[str, Any],
    *,
    mode: str,
) -> List[Dict[str, Any]]:
    """Build the four semantic practice components independently of score gates.

    These are product heuristics, not calibrated educational measurements. The
    same component builder is consumed by both the C-end dimension renderer and
    the overall practice-score policy so their semantics cannot silently drift.
    """
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    fluency = details.get("fluency") if isinstance(details.get("fluency"), Mapping) else {}
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    content = details.get("content_match") if isinstance(details.get("content_match"), Mapping) else {}
    target_relative = _target_relative_allowed(details)

    rate_score = _number(fluency.get("rate_score"))
    pause_score = _number(fluency.get("pause_score"))
    delivery_legacy = _number(fluency.get("delivery_fluency_score", result.get("fluency_score")))
    fluency_value = _blend([(rate_score, 0.48), (pause_score, 0.52), (delivery_legacy, 0.0)])
    if fluency_value is None:
        fluency_value = delivery_legacy if delivery_legacy is not None else 70.0
    fluency_conf = "high" if _clip01(reliability.get("endpointing"), 1.0) >= 0.75 else "medium"

    clarity_value, clarity_source, clarity_construct, clarity_conf, clarity_tier = _clarity_proxy(
        result,
        details,
        allow_target_relative=target_relative,
    )

    legacy_timing = _number(result.get("pronunciation_score"))
    duration_ratio = _number(reliability.get("duration_ratio_to_reference"))
    if duration_ratio is None:
        duration_ratio = _number(content.get("duration_ratio"))
    duration_score = _duration_match_score(duration_ratio)
    alignment_available, alignment_fallback, _alignment_mode = _alignment_state(result, details)

    if target_relative and alignment_available and legacy_timing is not None:
        rhythm_value = _blend([(legacy_timing, 0.78), (duration_score, 0.22)]) or legacy_timing
        rhythm_conf = "medium"
        rhythm_tier = "local_mora_plus_global_duration"
        rhythm_source = "pronunciation_score+duration_ratio_to_reference"
    elif target_relative:
        rhythm_value = _blend([(rate_score, 0.62), (duration_score, 0.38)])
        if rhythm_value is None:
            rhythm_value = legacy_timing if legacy_timing is not None else 70.0
        rhythm_conf = "low"
        rhythm_tier = "alignment_fallback_broad_timing" if alignment_fallback else "broad_timing_fallback"
        rhythm_source = "rate_score+duration_ratio_to_reference"
    else:
        rhythm_value = rate_score if rate_score is not None else 70.0
        rhythm_conf = "low"
        rhythm_tier = "reference_independent_rate_fallback"
        rhythm_source = "details.fluency.rate_score"

    intonation_value, intonation_source, intonation_construct, intonation_conf, intonation_tier = _intonation_proxy(
        result,
        details,
        mode=str(mode or ""),
        allow_reference_relative=target_relative,
    )

    mismatch_note = " target-relative evidence disabled because the spoken Japanese did not match the fixed target." if not target_relative else ""
    alignment_note = " local alignment fell back, so target-local timing/F0 evidence was not used as if it were precise." if alignment_fallback else ""
    return [
        _dimension(
            "delivery_fluency",
            "流暢さ",
            fluency_value,
            available=True,
            source_field="details.fluency.rate_score+pause_score",
            construct="speed_and_breakdown_fluency",
            confidence=fluency_conf,
            evidence_tier="rate_plus_pause",
            note="speed and breakdown fluency are combined; repair fluency is not yet modeled",
        ),
        _dimension(
            "clarity",
            "明瞭さ",
            clarity_value,
            available=True,
            source_field=clarity_source,
            construct=clarity_construct,
            confidence=clarity_conf,
            evidence_tier=clarity_tier,
            note="broad machine-intelligibility/phonetic-clarity practice proxy; not recording quality and not a formal human comprehensibility score." + mismatch_note,
        ),
        _dimension(
            "mora_timing",
            "リズム",
            rhythm_value,
            available=True,
            source_field=rhythm_source,
            construct=(
                "japanese_timing_structure_with_mora_special_mora_and_global_tempo_evidence"
                if target_relative
                else "reference_independent_broad_japanese_timing_proxy"
            ),
            confidence=rhythm_conf,
            evidence_tier=rhythm_tier,
            note="Japanese rhythm is not assumed to be perfectly equal-mora timing." + mismatch_note + alignment_note,
        ),
        _dimension(
            "intonation",
            "抑揚",
            intonation_value,
            available=True,
            source_field=intonation_source,
            construct=intonation_construct,
            confidence=intonation_conf,
            evidence_tier=intonation_tier,
            note="phrase/sentence intonation practice proxy; not strict lexical pitch-accent correctness." + mismatch_note + alignment_note,
        ),
    ]


def build_consumer_score_dimensions(
    result: Mapping[str, Any],
    user_facing: Mapping[str, Any],
    *,
    mode: str,
) -> List[Dict[str, Any]]:
    """Build four always-display C-end practice dimensions."""
    score_available = user_facing.get("display_score") is not None
    if not score_available:
        return [
            _dimension(key, label, None, source_field="", construct=construct, available=False, confidence="unavailable")
            for key, label, construct in (
                ("delivery_fluency", "流暢さ", "speed_and_breakdown_fluency"),
                ("clarity", "明瞭さ", "machine_intelligibility_and_phonetic_clarity_proxy"),
                ("mora_timing", "リズム", "japanese_timing_structure_proxy"),
                ("intonation", "抑揚", "phrase_sentence_f0_movement_proxy"),
            )
        ]
    return build_consumer_score_components(result, mode=mode)
