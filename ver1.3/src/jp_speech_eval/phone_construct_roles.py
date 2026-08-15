"""Construct-role policy for Japanese phone-level research evidence.

Phone-CTC backends expose ``N`` and ``cl`` as labels, but that does not make
those morae interchangeable with ordinary consonant/vowel correctness. Japanese
special morae are strongly duration/context dependent and the product already
keeps special-mora/timing evidence conceptually separate from ordinary
segmental clarity.

This module provides a fail-safe research view over criterion bundles:

* ordinary segmental phones -> eligible for future clarity criterion modeling;
* ``N`` / ``cl`` -> special-mora timing-support role; CTC deletion/sequence
  evidence may be retained, but ordinary normalized-GOP / logit-competitor
  features are not automatically treated as clarity evidence;
* long-vowel correctness is not represented by a dedicated phone token here and
  remains a duration/mora-timing problem.

No feature is mapped to a learner-facing score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List

from .phone_criterion_features import PhoneCriterionFeatureBundle


ORDINARY_SEGMENTAL_CLARITY = "ordinary_segmental_clarity"
SPECIAL_MORA_TIMING_SUPPORT = "special_mora_timing_support"
SPECIAL_MORA_PHONES = frozenset({"N", "cl"})


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
    return SPECIAL_MORA_TIMING_SUPPORT if str(phone) in SPECIAL_MORA_PHONES else ORDINARY_SEGMENTAL_CLARITY


def build_construct_role_view(bundle: PhoneCriterionFeatureBundle) -> Dict[str, Any]:
    """Partition a criterion bundle without reinterpreting raw feature scales."""
    if not bundle.available:
        return {
            "available": False,
            "reason": "criterion_bundle_unavailable",
            "product_score_changed": False,
        }

    rows: List[PhoneConstructRow] = []
    for source in bundle.rows:
        role = construct_role_for_phone(source.canonical_phone)
        if role == SPECIAL_MORA_TIMING_SUPPORT:
            # Keep the target-vs-deletion signal because it directly asks
            # whether the special mora is acoustically supported. Do not expose
            # generic ordinary-phone substitution/normalized-GOP features as
            # primary clarity evidence for this row.
            row = PhoneConstructRow(
                phone_index=int(source.phone_index),
                canonical_phone=str(source.canonical_phone),
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
                    "special_mora_requires_duration_and_context_evidence",
                    "ctc_deletion_signal_is_supporting_evidence_not_standalone_correctness",
                    "excluded_from_ordinary_clarity_primary_features",
                ],
            )
        else:
            row = PhoneConstructRow(
                phone_index=int(source.phone_index),
                canonical_phone=str(source.canonical_phone),
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
        rows.append(row)

    ordinary = [row for row in rows if row.construct_role == ORDINARY_SEGMENTAL_CLARITY]
    special = [row for row in rows if row.construct_role == SPECIAL_MORA_TIMING_SUPPORT]
    return {
        "available": True,
        "schema": "phone_construct_role_view_v1",
        "model_id": bundle.model_id,
        "revision": bundle.revision,
        "rows": [row.to_dict() for row in rows],
        "summary": {
            "phone_count": len(rows),
            "ordinary_segmental_count": len(ordinary),
            "special_mora_count": len(special),
            "special_mora_phones": sorted(SPECIAL_MORA_PHONES),
            "special_mora_used_as_ordinary_clarity_primary": False,
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
