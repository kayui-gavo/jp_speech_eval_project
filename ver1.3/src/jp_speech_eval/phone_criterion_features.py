"""Criterion-ready assembly of Japanese phone research features.

This module joins the transparent enumerated segmentation-free feature family
(LPP/LPR/substitution/deletion) with the published SD normalized-forward
``Occ(i)`` diagnostics.  It creates a stable machine-readable bundle for future
labeled criterion experiments while deliberately refusing any correctness or
learner-facing score mapping.

A bundle is model-specific evidence. Raw values from different phone-CTC
backbones must not be averaged or assumed to share a calibrated scale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List

from .segmentation_free_gop import SegmentationFreeGopResult
from .segmentation_free_gop_norm import SegmentationFreeNormResult


SCHEMA = "phone_criterion_feature_bundle_v1"


@dataclass(frozen=True)
class PhoneCriterionFeatureRow:
    phone_index: int
    canonical_phone: str
    canonical_log_posterior: float
    canonical_log_posterior_per_frame: float
    deletion_lpr: float
    substitution_lprs: Dict[str, float]
    enumerated_gop_sf_sd: float
    normalized_graph_gop_sf_sd: float
    occ_i: float
    best_noncanonical_type: str
    best_noncanonical_phone: str | None
    best_noncanonical_lpr: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhoneCriterionFeatureBundle:
    available: bool
    schema: str
    model_id: str
    revision: str
    canonical_phones: List[str]
    substitution_phone_inventory: List[str]
    rows: List[PhoneCriterionFeatureRow]
    summary: Dict[str, Any]
    warnings: List[str]
    score_mapped: bool = False
    product_calibrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["rows"] = [row.to_dict() for row in self.rows]
        return payload


def _unavailable(reason: str, *, model_id: str = "", revision: str = "") -> PhoneCriterionFeatureBundle:
    return PhoneCriterionFeatureBundle(
        available=False,
        schema=SCHEMA,
        model_id=model_id,
        revision=revision,
        canonical_phones=[],
        substitution_phone_inventory=[],
        rows=[],
        summary={"reason": reason},
        warnings=[reason],
        score_mapped=False,
        product_calibrated=False,
    )


def build_phone_criterion_feature_bundle(
    enumerated: SegmentationFreeGopResult,
    normalized: SegmentationFreeNormResult,
) -> PhoneCriterionFeatureBundle:
    """Join alignment-free feature families by canonical phone position.

    The function is intentionally strict. A mismatch in model provenance,
    phone sequence, row count or phone identity is treated as unavailable rather
    than silently aligning unrelated evidence.
    """
    model_id = str(enumerated.model_id or normalized.model_id or "")
    revision = str(enumerated.revision or normalized.revision or "")
    if not enumerated.available:
        return _unavailable("enumerated_features_unavailable", model_id=model_id, revision=revision)
    if not normalized.available:
        return _unavailable("normalized_features_unavailable", model_id=model_id, revision=revision)
    if str(enumerated.model_id or "") != str(normalized.model_id or ""):
        return _unavailable("model_id_mismatch", model_id=model_id, revision=revision)
    if str(enumerated.revision or "") != str(normalized.revision or ""):
        return _unavailable("model_revision_mismatch", model_id=model_id, revision=revision)
    if list(enumerated.canonical_phones) != list(normalized.canonical_phones):
        return _unavailable("canonical_phone_sequence_mismatch", model_id=model_id, revision=revision)
    if len(enumerated.evidence) != len(normalized.evidence):
        return _unavailable("feature_row_count_mismatch", model_id=model_id, revision=revision)

    frame_count = int(normalized.summary.get("frame_count") or 0)
    if frame_count <= 0:
        return _unavailable("normalized_frame_count_missing", model_id=model_id, revision=revision)

    rows: List[PhoneCriterionFeatureRow] = []
    for enum_row, norm_row in zip(enumerated.evidence, normalized.evidence):
        if int(enum_row.phone_index) != int(norm_row.phone_index):
            return _unavailable("phone_index_mismatch", model_id=model_id, revision=revision)
        if str(enum_row.canonical_phone) != str(norm_row.canonical_phone):
            return _unavailable("canonical_phone_row_mismatch", model_id=model_id, revision=revision)
        values = [
            enum_row.canonical_log_posterior,
            enum_row.deletion_log_posterior_ratio,
            enum_row.gop_sf_sd,
            norm_row.gop_sf_sd_norm,
            norm_row.occ_i,
            enum_row.best_noncanonical_log_posterior_ratio,
            *enum_row.substitution_log_posterior_ratios.values(),
        ]
        if any(not math.isfinite(float(value)) for value in values):
            return _unavailable("nonfinite_phone_feature", model_id=model_id, revision=revision)

        rows.append(
            PhoneCriterionFeatureRow(
                phone_index=int(enum_row.phone_index),
                canonical_phone=str(enum_row.canonical_phone),
                canonical_log_posterior=float(enum_row.canonical_log_posterior),
                canonical_log_posterior_per_frame=float(enum_row.canonical_log_posterior) / frame_count,
                deletion_lpr=float(enum_row.deletion_log_posterior_ratio),
                substitution_lprs={
                    str(phone): float(value)
                    for phone, value in enum_row.substitution_log_posterior_ratios.items()
                },
                enumerated_gop_sf_sd=float(enum_row.gop_sf_sd),
                normalized_graph_gop_sf_sd=float(norm_row.gop_sf_sd_norm),
                occ_i=float(norm_row.occ_i),
                best_noncanonical_type=str(enum_row.best_noncanonical_alternative_type),
                best_noncanonical_phone=(
                    str(enum_row.best_noncanonical_alternative_phone)
                    if enum_row.best_noncanonical_alternative_phone is not None
                    else None
                ),
                best_noncanonical_lpr=float(enum_row.best_noncanonical_log_posterior_ratio),
            )
        )

    return PhoneCriterionFeatureBundle(
        available=True,
        schema=SCHEMA,
        model_id=model_id,
        revision=revision,
        canonical_phones=list(enumerated.canonical_phones),
        substitution_phone_inventory=list(enumerated.feature_phone_inventory),
        rows=rows,
        summary={
            "phone_count": len(rows),
            "frame_count": frame_count,
            "feature_family": "joint_LPP_LPR_enumerated_SD_graph_GOP_Occ",
            "model_specific_raw_scale": True,
            "cross_model_raw_averaging_allowed": False,
            "individual_feature_is_pronunciation_decision": False,
            "requires_labeled_phone_or_human_criterion": True,
            "intended_use": "future_labeled_MDD_or_pronunciation_regression_research",
            "product_score_changed": False,
        },
        warnings=sorted(set(list(enumerated.warnings) + list(normalized.warnings))),
        score_mapped=False,
        product_calibrated=False,
    )
