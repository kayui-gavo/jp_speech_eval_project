"""Hybrid phone-level research features for future Japanese criterion studies.

Alignment-free CTC-GOP avoids forced phone boundaries and can model
substitution/deletion evidence, while recent GOP work also finds value in
logit-based features and uncertainty. Rather than choosing one family before
Japanese learner validation, this module joins them with explicit provenance.

The frame-local branch still depends on a CTC Viterbi support path. Its support
frames are **not** physical phone boundaries/durations. In addition, Japanese
special morae ``N`` and ``cl`` are tagged as a separate construct: their raw
frame-local numbers may be retained for research, but they are not ordinary
segmental-clarity features and must not be silently mixed into a clarity model.

The resulting bundle is therefore a supervised-research design matrix, not a
pronunciation score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from .japanese_phoneme_gop import SPECIAL_MORA_TOKENS
from .phone_criterion_features import PhoneCriterionFeatureBundle
from .phoneme_gop import PhoneGopResult


SCHEMA = "hybrid_phone_criterion_feature_bundle_v2"


@dataclass(frozen=True)
class HybridPhoneCriterionRow:
    phone_index: int
    canonical_phone: str
    construct_role: str
    frame_local_ordinary_clarity_feature_applicable: bool
    alignment_free_substitution_feature_applicable: bool
    alignment_free_deletion_feature_applicable: bool
    frame_local_mean_logit_margin: float
    frame_local_max_logit_margin: float
    frame_local_posterior_gop_margin: float
    frame_local_mean_entropy: float
    frame_local_support_frame_count: int
    frame_local_support_mean_logprob: float
    alignment_free_deletion_lpr: float
    alignment_free_substitution_lprs: Dict[str, float]
    alignment_free_enumerated_gop_sf_sd: float
    alignment_free_normalized_graph_gop_sf_sd: float
    alignment_free_occ_i: float
    alignment_free_best_noncanonical_type: str
    alignment_free_best_noncanonical_phone: str | None
    alignment_free_best_noncanonical_lpr: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HybridPhoneCriterionBundle:
    available: bool
    schema: str
    model_id: str
    revision: str
    canonical_phones: List[str]
    rows: List[HybridPhoneCriterionRow]
    summary: Dict[str, Any]
    warnings: List[str]
    score_mapped: bool = False
    product_calibrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["rows"] = [row.to_dict() for row in self.rows]
        return payload


def _unavailable(reason: str, *, model_id: str = "", revision: str = "") -> HybridPhoneCriterionBundle:
    return HybridPhoneCriterionBundle(
        available=False,
        schema=SCHEMA,
        model_id=str(model_id),
        revision=str(revision),
        canonical_phones=[],
        rows=[],
        summary={"reason": str(reason), "product_score_changed": False},
        warnings=[str(reason)],
        score_mapped=False,
        product_calibrated=False,
    )


def _frame_model_matches(frame_local: PhoneGopResult, criterion: PhoneCriterionFeatureBundle) -> bool:
    frame_id = str(frame_local.model_id or "")
    base = str(criterion.model_id or "")
    revision = str(criterion.revision or "")
    if frame_id == base:
        return True
    if revision and frame_id == f"{base}@{revision}":
        return True
    return False


def build_hybrid_phone_criterion_bundle(
    frame_local: PhoneGopResult,
    criterion: PhoneCriterionFeatureBundle,
) -> HybridPhoneCriterionBundle:
    """Strictly join Viterbi/logit and alignment-free feature families.

    The join preserves every raw research feature, but adds explicit construct
    applicability flags so downstream criterion code can build separate
    ordinary-segmental and special-mora models instead of accidentally mixing
    them.
    """
    model_id = str(criterion.model_id or frame_local.model_id or "")
    revision = str(criterion.revision or "")
    if not frame_local.available:
        return _unavailable("frame_local_features_unavailable", model_id=model_id, revision=revision)
    if not criterion.available:
        return _unavailable("alignment_free_criterion_features_unavailable", model_id=model_id, revision=revision)
    if not _frame_model_matches(frame_local, criterion):
        return _unavailable("model_provenance_mismatch", model_id=model_id, revision=revision)
    if list(frame_local.canonical_phones) != list(criterion.canonical_phones):
        return _unavailable("canonical_phone_sequence_mismatch", model_id=model_id, revision=revision)
    if len(frame_local.evidence) != len(criterion.rows):
        return _unavailable("feature_row_count_mismatch", model_id=model_id, revision=revision)

    rows: List[HybridPhoneCriterionRow] = []
    for frame_row, af_row in zip(frame_local.evidence, criterion.rows):
        if int(frame_row.phone_index) != int(af_row.phone_index):
            return _unavailable("phone_index_mismatch", model_id=model_id, revision=revision)
        if str(frame_row.canonical_phone) != str(af_row.canonical_phone):
            return _unavailable("canonical_phone_row_mismatch", model_id=model_id, revision=revision)
        phone = str(frame_row.canonical_phone)
        special_mora = phone in SPECIAL_MORA_TOKENS
        rows.append(
            HybridPhoneCriterionRow(
                phone_index=int(frame_row.phone_index),
                canonical_phone=phone,
                construct_role=("special_mora_timing" if special_mora else "ordinary_segmental_clarity"),
                frame_local_ordinary_clarity_feature_applicable=not special_mora,
                alignment_free_substitution_feature_applicable=not special_mora,
                alignment_free_deletion_feature_applicable=True,
                frame_local_mean_logit_margin=float(frame_row.mean_logit_margin),
                frame_local_max_logit_margin=float(frame_row.max_logit_margin),
                frame_local_posterior_gop_margin=float(frame_row.posterior_gop_margin),
                frame_local_mean_entropy=float(frame_row.mean_entropy),
                frame_local_support_frame_count=int(frame_row.frame_count),
                frame_local_support_mean_logprob=float(frame_row.path_support_mean_logprob),
                alignment_free_deletion_lpr=float(af_row.deletion_lpr),
                alignment_free_substitution_lprs=dict(af_row.substitution_lprs),
                alignment_free_enumerated_gop_sf_sd=float(af_row.enumerated_gop_sf_sd),
                alignment_free_normalized_graph_gop_sf_sd=float(af_row.normalized_graph_gop_sf_sd),
                alignment_free_occ_i=float(af_row.occ_i),
                alignment_free_best_noncanonical_type=str(af_row.best_noncanonical_type),
                alignment_free_best_noncanonical_phone=af_row.best_noncanonical_phone,
                alignment_free_best_noncanonical_lpr=float(af_row.best_noncanonical_lpr),
            )
        )

    special_count = sum(row.construct_role == "special_mora_timing" for row in rows)
    return HybridPhoneCriterionBundle(
        available=True,
        schema=SCHEMA,
        model_id=model_id,
        revision=revision,
        canonical_phones=list(criterion.canonical_phones),
        rows=rows,
        summary={
            "phone_count": len(rows),
            "ordinary_segmental_row_count": len(rows) - special_count,
            "special_mora_row_count": special_count,
            "contains_frame_local_logit_features": True,
            "contains_frame_local_posterior_features": True,
            "contains_frame_local_uncertainty": True,
            "contains_alignment_free_substitution_deletion_features": True,
            "contains_normalized_graph_occ_features": True,
            "frame_local_support_requires_ctc_viterbi_path": True,
            "frame_local_support_is_physical_phone_boundary": False,
            "frame_local_support_frame_count_is_physical_duration": False,
            "special_mora_frame_local_raw_values_are_ordinary_clarity_features": False,
            "special_mora_requires_construct_specific_feature_selection": True,
            "special_mora_primary_local_alignment_free_feature": "deletion_lpr_with_timing_context_required",
            "feature_family_selection_requires_japanese_l2_criterion": True,
            "phone_specific_weighting_not_learned_yet": True,
            "cross_model_raw_averaging_allowed": False,
            "individual_feature_is_pronunciation_decision": False,
            "intended_use": "speaker_held_out_labeled_Japanese_MDD_or_pronunciation_regression_with_construct_specific_feature_selection",
            "product_score_changed": False,
        },
        warnings=sorted(set(list(frame_local.warnings) + list(criterion.warnings))),
        score_mapped=False,
        product_calibrated=False,
    )
