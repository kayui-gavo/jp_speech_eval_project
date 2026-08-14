from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


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
        "note": note,
    }


def build_consumer_score_dimensions(
    result: Mapping[str, Any],
    user_facing: Mapping[str, Any],
    *,
    mode: str,
) -> List[Dict[str, Any]]:
    """Build scientifically narrower top-level dimensions for the C-end preview.

    Important semantic corrections:

    * the legacy ``pronunciation_score`` is a mora-timing/special-mora proxy,
      so it is shown as rhythm/timing rather than segmental pronunciation;
    * delivery fluency is the pause/continuity dimension;
    * reference-relative F0 contour similarity is shown as ``抑揚`` and is
      deliberately separate from lexical pitch-accent correctness;
    * no top-level numeric pronunciation score is invented until mapped SSL or
      another validated pronunciation backbone exists.
    """
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    pronunciation = details.get("pronunciation") if isinstance(details.get("pronunciation"), Mapping) else {}
    fluency = details.get("fluency") if isinstance(details.get("fluency"), Mapping) else {}
    prosody = details.get("prosody") if isinstance(details.get("prosody"), Mapping) else {}
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}
    shadow = details.get("shadow") if isinstance(details.get("shadow"), Mapping) else {}

    score_available = user_facing.get("display_score") is not None
    alignment_available = bool(alignment.get("available", True)) and not bool(alignment.get("used_equal_fallback"))

    rhythm_value = result.get("pronunciation_score")
    rhythm_interpretation = str(pronunciation.get("score_interpretation") or "")
    rhythm_available = score_available and alignment_available and (
        "mora_timing_proxy" in rhythm_interpretation or rhythm_value is not None
    )

    delivery_value = fluency.get("delivery_fluency_score", result.get("fluency_score"))
    delivery_available = score_available and delivery_value is not None

    fixed_reference = str(mode or "").strip() in {
        "reference",
        "reference_based",
        "reference_fixed_sentence",
        "fixed_reference",
    }
    f0_coverage = float(reliability.get("f0_coverage", 0.0) or 0.0)
    valid_mora = int(prosody.get("contour_valid_mora_count", prosody.get("valid_mora_count", 0)) or 0)
    mora_count = max(1, int(prosody.get("mora_count", len(result.get("moras") or [])) or 1))
    contour_corr = prosody.get("contour_corr")
    intonation_available = (
        score_available
        and fixed_reference
        and alignment_available
        and f0_coverage >= 0.50
        and valid_mora >= max(3, int((mora_count + 1) * 0.5))
        and contour_corr is not None
        and result.get("prosody_score") is not None
    )

    dimensions: List[Dict[str, Any]] = [
        _dimension(
            "mora_timing",
            "リズム",
            rhythm_value,
            available=rhythm_available,
            source_field="pronunciation_score",
            construct="mora_timing_and_special_mora_duration_proxy",
            note="legacy pronunciation_score is intentionally relabeled; it is not segmental pronunciation correctness",
        ),
        _dimension(
            "delivery_fluency",
            "流暢さ",
            delivery_value,
            available=delivery_available,
            source_field="details.fluency.delivery_fluency_score",
            construct="pause_and_delivery_continuity",
        ),
    ]

    if fixed_reference:
        dimensions.append(
            _dimension(
                "intonation",
                "抑揚",
                result.get("prosody_score"),
                available=intonation_available,
                source_field="prosody_score",
                construct="reference_relative_f0_contour_similarity",
                note="not lexical pitch-accent correctness",
            )
        )

    ssl = shadow.get("ssl_pronunciation") if isinstance(shadow.get("ssl_pronunciation"), Mapping) else {}
    mapped_pronunciation = ssl.get("mapped_score", ssl.get("value"))
    if bool(ssl.get("score_mapped")) and mapped_pronunciation is not None:
        dimensions.insert(
            0,
            _dimension(
                "pronunciation",
                "発音",
                mapped_pronunciation,
                available=score_available,
                source_field="details.shadow.ssl_pronunciation.mapped_score",
                construct="mapped_pronunciation_accuracy_evidence",
            ),
        )

    return dimensions
