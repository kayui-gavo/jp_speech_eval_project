"""Single source of truth for the C-end four-score contract.

This module intentionally contains product semantics, not model calibration.
Changing a weight, display transform, public dimension meaning, or history
comparability rule requires a new ``SCORE_CONTRACT_VERSION``. Old records stay
readable but must not be interpreted as directly comparable progress points.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


SCORE_CONTRACT_VERSION = "consumer_four_score_v2"
EVIDENCE_SCHEMA_VERSION = "consumer_evidence_v2"
SCORE_POLICY_ID = "semantic_four_component_product_heuristic_v2"

PRODUCT_COMPONENT_WEIGHTS: Dict[str, float] = {
    "clarity": 0.30,
    "mora_timing": 0.25,
    "delivery_fluency": 0.25,
    "intonation": 0.20,
}

PUBLIC_DIMENSIONS: Dict[str, Dict[str, str]] = {
    "clarity": {
        "component_key": "clarity",
        "label_ja": "明瞭さ",
        "construct": "broad_practice_clarity_not_strict_phone_accuracy",
    },
    "rhythm": {
        "component_key": "mora_timing",
        "label_ja": "リズム",
        "construct": "japanese_timing_structure_not_equal_mora_isochrony",
    },
    "fluency": {
        "component_key": "delivery_fluency",
        "label_ja": "流暢さ",
        "construct": "utterance_fluency_speed_and_breakdown_with_repair_pending",
    },
    "intonation": {
        "component_key": "intonation",
        "label_ja": "抑揚",
        "construct": "phrase_sentence_intonation_not_lexical_pitch_accent_correctness",
    },
}

DISPLAY_ANCHOR = 70.0
DISPLAY_STRETCH = 1.08

_FIXED_MODES = {
    "reference",
    "reference_based",
    "reference_fixed_sentence",
    "fixed_reference",
}
_WEAK_REFERENCE_MODES = {
    "asr_confirmed_weak_reference",
    "asr_pseudo_reference",
    "kanade_asr_voice_reference",
    "kanade_asr_confirmed_voice_reference",
}
_BROAD_MODES = {
    "reference_free_acoustic",
    "transcript_assisted_light",
    "reference_mismatch_general_japanese",
    "general_japanese",
}


def apply_display_transform(raw_weighted_score: float) -> float:
    """Apply the frozen UX transform for this score contract."""

    raw = float(raw_weighted_score)
    value = DISPLAY_ANCHOR + DISPLAY_STRETCH * (raw - DISPLAY_ANCHOR)
    return max(0.0, min(100.0, value))


def score_contract_payload() -> Dict[str, Any]:
    """Return serialisable contract metadata for API/history telemetry."""

    return {
        "version": SCORE_CONTRACT_VERSION,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "policy": SCORE_POLICY_ID,
        "weights": dict(PRODUCT_COMPONENT_WEIGHTS),
        "public_dimensions": {key: dict(value) for key, value in PUBLIC_DIMENSIONS.items()},
        "display_transform": {
            "type": "anchored_linear_ux_transform",
            "anchor": DISPLAY_ANCHOR,
            "stretch": DISPLAY_STRETCH,
            "formula": "70 + 1.08 * (raw - 70)",
        },
        "product_calibrated": False,
        "formal_educational_measurement": False,
    }


def mode_family(mode: Any) -> str:
    value = str(mode or "").strip()
    if value in _FIXED_MODES:
        return "fixed_reference"
    if value in _WEAK_REFERENCE_MODES:
        return "weak_reference"
    if value in _BROAD_MODES:
        return "general_japanese"
    if value.startswith("kanade"):
        return "demo"
    return value or "unknown"


def reference_identity(result: Mapping[str, Any]) -> Optional[str]:
    """Best-effort stable reference identity for progress comparability.

    Prefer explicit human/reference IDs. Generated references already expose a
    path-independent synthesis config hash, which is safer than a cache path.
    A filesystem prefix remains a legacy fallback only.
    """

    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    explicit = str(details.get("reference_id") or "").strip()
    if explicit:
        return f"reference:{explicit}"
    config_hash = str(details.get("reference_config_hash") or "").strip()
    if config_hash:
        return f"reference_config:{config_hash}"
    for value in (
        details.get("reference_cache_prefix"),
        result.get("cache_prefix"),
    ):
        text = str(value or "").strip()
        if text:
            return f"legacy_cache:{text}"
    return None


def comparison_context(result: Mapping[str, Any], *, mode: Any = None) -> Dict[str, Any]:
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    effective_mode = str(mode or details.get("mode") or "fixed_reference")
    return {
        "score_contract_version": SCORE_CONTRACT_VERSION,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "mode": effective_mode,
        "mode_family": mode_family(effective_mode),
        "target_text": str(result.get("target_text") or ""),
        "reference_id": reference_identity(result),
    }


def history_comparability(
    current: Mapping[str, Any],
    previous: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Decide whether two product score records may be shown as progress.

    ``None`` means there is genuinely no previous record. An empty mapping is
    different: it represents a legacy/unversioned record and is reported as
    such so migration/audit code can distinguish the two cases.
    """

    if previous is None:
        return {"comparable": False, "reason": "no_previous_record"}

    current_contract = str(current.get("score_contract_version") or "")
    previous_contract = str(previous.get("score_contract_version") or "")
    if not previous_contract:
        return {"comparable": False, "reason": "legacy_record_without_score_contract"}
    if current_contract != previous_contract:
        return {"comparable": False, "reason": "score_contract_version_mismatch"}

    current_family = str(current.get("mode_family") or "")
    previous_family = str(previous.get("mode_family") or "")
    if current_family != previous_family:
        return {"comparable": False, "reason": "evaluation_mode_family_mismatch"}

    current_target = str(current.get("target_text") or "")
    previous_target = str(previous.get("target_text") or "")
    if current_family == "fixed_reference" and current_target != previous_target:
        return {"comparable": False, "reason": "fixed_target_mismatch"}

    current_reference = str(current.get("reference_id") or "")
    previous_reference = str(previous.get("reference_id") or "")
    if current_family == "fixed_reference":
        if not current_reference or not previous_reference:
            return {"comparable": False, "reason": "fixed_reference_identity_missing"}
        if current_reference != previous_reference:
            return {"comparable": False, "reason": "fixed_reference_changed"}

    return {"comparable": True, "reason": "same_score_contract_and_context"}
