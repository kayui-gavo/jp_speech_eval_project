"""Construct-aware research view for Japanese phone evidence.

The criterion bundle already carries per-phone construct metadata. This module
must not silently re-derive a second, incompatible ontology from the phone token
alone: a repeated vowel token can be either an ordinary vowel or the extension
mora of a long vowel, while ``N`` / ``cl`` are special-mora timing events.

The view therefore treats the criterion row's audited ``construct_role`` as the
authoritative source and fails closed on contradictory applicability flags.
Ordinary segmental rows can expose the full research feature family; special
morae and long-vowel extension rows retain deletion/support evidence for future
construct-specific models but are excluded from ordinary clarity-primary
features.

Nothing in this module maps raw evidence to a learner-facing score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from .japanese_phone_roles import LONG_VOWEL_ROLE, ORDINARY_ROLE, SPECIAL_MORA_ROLE
from .phone_criterion_features import PhoneCriterionFeatureBundle


# Backward-compatible exported names now use the same values as the criterion
# bundle and Japanese target-role inference. The old ``*_support`` string value
# was a duplicate ontology and is intentionally retired.
ORDINARY_SEGMENTAL_CLARITY = ORDINARY_ROLE
SPECIAL_MORA_TIMING_SUPPORT = SPECIAL_MORA_ROLE
LONG_VOWEL_TIMING_SUPPORT = LONG_VOWEL_ROLE
SPECIAL_MORA_PHONES = frozenset({"N", "cl"})
ALLOWED_CONSTRUCT_ROLES = frozenset(
    {ORDINARY_SEGMENTAL_CLARITY, SPECIAL_MORA_TIMING_SUPPORT, LONG_VOWEL_TIMING_SUPPORT}
)


@dataclass(frozen=True)
class PhoneConstructRow:
    phone_index: int
    canonical_phone: str
    construct_role: str
    clarity_primary_eligible: bool
    timing_context_required: bool
    deletion_lpr: float
    substitution_lprs: Dict[str, float]
    enumerated_gop_sf_sd: float | None
    normalized_graph_gop_sf_sd: float | None
    occ_i: float | None
    best_noncanonical_type: str
    best_noncanonical_phone: str | None
    best_noncanonical_lpr: float
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def construct_role_for_phone(phone: str) -> str:
    """Legacy fallback when only a phone token is available.

    ``N`` and ``cl`` are unambiguously special morae. Ordinary vowel tokens are
    *not* enough to identify a long-vowel extension, so every other phone falls
    back to ordinary segmental clarity. Construct-aware callers should use the
    criterion row's explicit role instead.
    """
    return SPECIAL_MORA_TIMING_SUPPORT if str(phone) in SPECIAL_MORA_PHONES else ORDINARY_SEGMENTAL_CLARITY


def _unavailable(bundle: PhoneCriterionFeatureBundle, reason: str) -> Dict[str, Any]:
    return {
        "available": False,
        "schema": "phone_construct_role_view_v2",
        "model_id": bundle.model_id,
        "revision": bundle.revision,
        "reason": str(reason),
        "score_mapped": False,
        "product_calibrated": False,
        "product_score_changed": False,
    }


def build_construct_role_view(bundle: PhoneCriterionFeatureBundle) -> Dict[str, Any]:
    """Partition a criterion bundle without reinterpreting raw feature scales."""
    if not bundle.available:
        return _unavailable(bundle, "criterion_bundle_unavailable")

    rows: List[PhoneConstructRow] = []
    for source in bundle.rows:
        phone = str(source.canonical_phone)
        role = str(source.construct_role)
        if role not in ALLOWED_CONSTRUCT_ROLES:
            return _unavailable(bundle, f"unsupported_construct_role:{role}")

        is_special_phone = phone in SPECIAL_MORA_PHONES
        if is_special_phone and role != SPECIAL_MORA_TIMING_SUPPORT:
            return _unavailable(bundle, "special_mora_phone_role_mismatch")
        if not is_special_phone and role == SPECIAL_MORA_TIMING_SUPPORT:
            return _unavailable(bundle, "non_special_phone_marked_special_mora")

        expected_clarity = role == ORDINARY_SEGMENTAL_CLARITY
        if bool(source.ordinary_segmental_clarity_feature_applicable) != expected_clarity:
            return _unavailable(bundle, "clarity_applicability_role_mismatch")
        if bool(source.substitution_feature_applicable) != expected_clarity:
            return _unavailable(bundle, "substitution_applicability_role_mismatch")
        if not bool(source.deletion_feature_applicable):
            return _unavailable(bundle, "deletion_support_unexpectedly_disabled")
        if bool(source.normalized_occ_is_physical_duration):
            return _unavailable(bundle, "occ_i_must_not_be_reinterpreted_as_physical_duration")

        if expected_clarity:
            row = PhoneConstructRow(
                phone_index=int(source.phone_index),
                canonical_phone=phone,
                construct_role=role,
                clarity_primary_eligible=True,
                timing_context_required=False,
                deletion_lpr=float(source.deletion_lpr),
                substitution_lprs={str(k): float(v) for k, v in source.substitution_lprs.items()},
                enumerated_gop_sf_sd=float(source.enumerated_gop_sf_sd),
                normalized_graph_gop_sf_sd=float(source.normalized_graph_gop_sf_sd),
                occ_i=float(source.occ_i),
                best_noncanonical_type=str(source.best_noncanonical_type),
                best_noncanonical_phone=source.best_noncanonical_phone,
                best_noncanonical_lpr=float(source.best_noncanonical_lpr),
                notes=["requires_Japanese_L2_criterion_validation_before_clarity_mapping"],
            )
        else:
            timing_note = (
                "special_mora_requires_duration_and_context_evidence"
                if role == SPECIAL_MORA_TIMING_SUPPORT
                else "long_vowel_extension_requires_mora_timing_and_duration_evidence"
            )
            row = PhoneConstructRow(
                phone_index=int(source.phone_index),
                canonical_phone=phone,
                construct_role=role,
                clarity_primary_eligible=False,
                timing_context_required=True,
                deletion_lpr=float(source.deletion_lpr),
                substitution_lprs={},
                enumerated_gop_sf_sd=None,
                normalized_graph_gop_sf_sd=None,
                occ_i=None,
                best_noncanonical_type=str(source.best_noncanonical_type),
                best_noncanonical_phone=source.best_noncanonical_phone,
                best_noncanonical_lpr=float(source.best_noncanonical_lpr),
                notes=[
                    timing_note,
                    "ctc_deletion_signal_is_supporting_evidence_not_standalone_correctness",
                    "excluded_from_ordinary_clarity_primary_features",
                ],
            )
        rows.append(row)

    ordinary = [row for row in rows if row.construct_role == ORDINARY_SEGMENTAL_CLARITY]
    special = [row for row in rows if row.construct_role == SPECIAL_MORA_TIMING_SUPPORT]
    long_vowels = [row for row in rows if row.construct_role == LONG_VOWEL_TIMING_SUPPORT]
    return {
        "available": True,
        "schema": "phone_construct_role_view_v2",
        "model_id": bundle.model_id,
        "revision": bundle.revision,
        "rows": [row.to_dict() for row in rows],
        "summary": {
            "phone_count": len(rows),
            "ordinary_segmental_count": len(ordinary),
            "special_mora_count": len(special),
            "long_vowel_timing_count": len(long_vowels),
            "special_mora_phones": sorted(SPECIAL_MORA_PHONES),
            "construct_role_source": "criterion_bundle_explicit_metadata",
            "construct_role_rederived_from_phone_token": False,
            "special_mora_used_as_ordinary_clarity_primary": False,
            "long_vowel_used_as_ordinary_clarity_primary": False,
            "special_mora_requires_duration_context": True,
            "long_vowel_requires_separate_timing_evidence": True,
            "raw_feature_scale_changed": False,
            "score_mapped": False,
            "product_calibrated": False,
            "product_score_changed": False,
        },
    }


def clarity_primary_rows(view: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return only rows structurally eligible for future clarity modeling."""
    if not bool(view.get("available")):
        return []
    return [
        dict(row)
        for row in view.get("rows", [])
        if bool(row.get("clarity_primary_eligible"))
    ]


def special_mora_support_rows(view: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return dedicated special-mora supporting rows."""
    if not bool(view.get("available")):
        return []
    return [
        dict(row)
        for row in view.get("rows", [])
        if row.get("construct_role") == SPECIAL_MORA_TIMING_SUPPORT
    ]


def long_vowel_support_rows(view: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return long-vowel extension rows for future timing-specific modeling."""
    if not bool(view.get("available")):
        return []
    return [
        dict(row)
        for row in view.get("rows", [])
        if row.get("construct_role") == LONG_VOWEL_TIMING_SUPPORT
    ]
