from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


FIXED_REFERENCE_MODES = {
    "reference",
    "reference_based",
    "reference_fixed_sentence",
    "fixed_reference",
}


def _score(value: Any) -> Optional[int]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not (number == number):
        return None
    return int(round(max(0.0, min(100.0, number))))


def _dimension(
    key: str,
    label: str,
    value: Any,
    *,
    source_field: str,
    construct: str,
    available: bool,
    confidence: str = "unknown",
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
        "note": note,
    }


def _mapped_clarity_evidence(details: Mapping[str, Any]) -> tuple[Any, str, str]:
    """Return only pronunciation/clarity evidence that has an explicit score map.

    The old ``pronunciation_score`` is intentionally excluded here because it
    is mainly a mora-timing/special-mora proxy.  Recording quality and ASR text
    match are also excluded: analyzability/content verification are not speech
    clarity constructs.
    """
    shadow = details.get("shadow") if isinstance(details.get("shadow"), Mapping) else {}
    ssl = shadow.get("ssl_pronunciation") if isinstance(shadow.get("ssl_pronunciation"), Mapping) else {}
    if bool(ssl.get("score_mapped")):
        value = ssl.get("mapped_score", ssl.get("value"))
        if value is not None:
            return value, "details.shadow.ssl_pronunciation.mapped_score", "mapped_ssl_phonetic_clarity_evidence"

    pronunciation_evidence = (
        details.get("pronunciation_evidence")
        if isinstance(details.get("pronunciation_evidence"), Mapping)
        else {}
    )
    if bool(pronunciation_evidence.get("score_mapped")):
        value = pronunciation_evidence.get("mapped_score", pronunciation_evidence.get("value"))
        if value is not None:
            return value, "details.pronunciation_evidence.mapped_score", "mapped_pronunciation_clarity_evidence"

    return None, "", "pronunciation_clarity_evidence_not_yet_mapped"


def build_consumer_score_dimensions(
    result: Mapping[str, Any],
    user_facing: Mapping[str, Any],
    *,
    mode: str,
) -> List[Dict[str, Any]]:
    """Build the four C-end dimensions with non-overlapping semantics.

    Product labels are intentionally simple:

    * ``流暢さ`` = pauses / continuity / delivery fluency;
    * ``明瞭さ`` = pronunciation/phonetic clarity evidence, never recording
      quality and never the legacy mora-timing proxy;
    * ``リズム`` = Japanese timing structure, including mora timing and
      special-mora duration evidence;
    * ``抑揚`` = reference-relative phrase/sentence F0 contour, separate from
      strict lexical pitch-accent correctness.

    ``韻律`` is deliberately not a top-level label because prosody is an
    umbrella term that already includes rhythm and intonation.  Keeping both
    ``韻律`` and ``抑揚`` as peer scores would blur constructs.
    """
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    pronunciation = details.get("pronunciation") if isinstance(details.get("pronunciation"), Mapping) else {}
    fluency = details.get("fluency") if isinstance(details.get("fluency"), Mapping) else {}
    prosody = details.get("prosody") if isinstance(details.get("prosody"), Mapping) else {}
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}

    score_available = user_facing.get("display_score") is not None
    fixed_reference = str(mode or "").strip() in FIXED_REFERENCE_MODES
    alignment_available = bool(alignment.get("available", True)) and not bool(alignment.get("used_equal_fallback"))

    # 1) Fluency: continuity / pauses.  Speaking-rate effects remain supporting
    # evidence but the displayed source is the delivery-specific score.
    delivery_value = fluency.get("delivery_fluency_score", result.get("fluency_score"))
    delivery_available = score_available and delivery_value is not None
    delivery_confidence = "high" if score_available else "low"

    # 2) Clarity: never recycle the legacy timing proxy under a new name.
    clarity_value, clarity_source, clarity_construct = _mapped_clarity_evidence(details)
    clarity_available = score_available and clarity_value is not None

    # 3) Rhythm: the legacy pronunciation score is allowed here only because
    # its implementation explicitly describes itself as mora timing / special
    # mora duration rather than segmental pronunciation correctness.
    rhythm_value = result.get("pronunciation_score")
    rhythm_interpretation = str(pronunciation.get("score_interpretation") or "")
    rhythm_available = score_available and alignment_available and (
        "mora_timing_proxy" in rhythm_interpretation or rhythm_value is not None
    )

    # 4) Intonation: use the existing reference-relative F0 score whenever
    # there is enough real F0 evidence.  Lexical pitch accent remains a local
    # detail and is not a prerequisite for phrase-level intonation scoring.
    f0_coverage = float(reliability.get("f0_coverage", 0.0) or 0.0)
    valid_mora = int(prosody.get("contour_valid_mora_count", prosody.get("valid_mora_count", 0)) or 0)
    prosody_note = str(prosody.get("note") or "")
    contour_corr = prosody.get("contour_corr")
    hard_f0_failure = prosody_note in {"no_valid_f0", "insufficient_valid_mora_f0"}
    intonation_available = (
        score_available
        and fixed_reference
        and not hard_f0_failure
        and f0_coverage >= 0.35
        and valid_mora >= 3
        and contour_corr is not None
        and result.get("prosody_score") is not None
    )
    intonation_confidence = (
        "high"
        if intonation_available and alignment_available and f0_coverage >= 0.65
        else "medium"
        if intonation_available
        else "low"
    )

    return [
        _dimension(
            "delivery_fluency",
            "流暢さ",
            delivery_value,
            available=delivery_available,
            source_field="details.fluency.delivery_fluency_score",
            construct="pause_hesitation_and_delivery_continuity",
            confidence=delivery_confidence,
        ),
        _dimension(
            "clarity",
            "明瞭さ",
            clarity_value,
            available=clarity_available,
            source_field=clarity_source,
            construct=clarity_construct,
            confidence="medium" if clarity_available else "unavailable",
            note="not recording quality; not formal human intelligibility/comprehensibility until validated",
        ),
        _dimension(
            "mora_timing",
            "リズム",
            rhythm_value,
            available=rhythm_available,
            source_field="pronunciation_score",
            construct="reference_relative_mora_timing_and_special_mora_duration_proxy",
            confidence="medium" if rhythm_available else "low",
            note="legacy pronunciation_score is intentionally relabelled; Japanese rhythm is not assumed to be perfectly equal-mora timing",
        ),
        _dimension(
            "intonation",
            "抑揚",
            result.get("prosody_score"),
            available=intonation_available,
            source_field="prosody_score",
            construct="reference_relative_normalized_f0_contour_similarity",
            confidence=intonation_confidence,
            note="phrase/sentence intonation, not strict lexical pitch-accent correctness",
        ),
    ]
