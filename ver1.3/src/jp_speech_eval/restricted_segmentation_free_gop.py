"""Restricted-substitution alignment-free GOP research wrapper.

Recent alignment-free GOP work reports benefits from restricting substitution
alternatives using phonological knowledge instead of enumerating an entire
phone inventory.  This module applies that idea conservatively to the Japanese
phone-CTC research stack without changing the validated unrestricted extractor.

Important boundaries:
- candidate neighborhoods are Japanese research hypotheses, not learner labels;
- ``N`` and ``cl`` use canonical-vs-deletion evidence only here;
- no value is mapped to pronunciation correctness or a learner-facing /100;
- unrestricted and restricted values are never averaged together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .japanese_phone_substitutions import substitution_token_ids_by_position
from .segmentation_free_gop import (
    SegmentationFreePhoneFeature,
    compute_enumerated_fgop_sf_sd_features,
)


METHOD = "restricted_enumerated_fgop_ctc_sf_sd_v1"


@dataclass(frozen=True)
class RestrictedPhoneFeature:
    phone_index: int
    canonical_phone: str
    candidate_phones: tuple[str, ...]
    candidate_count: int
    search_policy: str
    fallback_used: bool
    canonical_log_posterior: float
    gop_sf_sd: float
    deletion_lpr: float
    substitution_lprs: Dict[str, float]
    best_noncanonical_type: str
    best_noncanonical_phone: str | None
    best_noncanonical_lpr: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RestrictedSegmentationFreeGopResult:
    available: bool
    method: str
    model_id: str
    revision: str
    canonical_phones: list[str]
    rows: list[RestrictedPhoneFeature]
    candidate_provenance: list[Dict[str, Any]]
    summary: Dict[str, Any]
    warnings: list[str]
    score_mapped: bool = False
    product_calibrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["rows"] = [row.to_dict() for row in self.rows]
        return payload


def _unavailable(
    reason: str,
    *,
    model_id: str,
    revision: str,
    canonical_phones: Sequence[str],
    warnings: Sequence[str] = (),
) -> RestrictedSegmentationFreeGopResult:
    return RestrictedSegmentationFreeGopResult(
        available=False,
        method=METHOD,
        model_id=str(model_id),
        revision=str(revision),
        canonical_phones=[str(x) for x in canonical_phones],
        rows=[],
        candidate_provenance=[],
        summary={"reason": str(reason)},
        warnings=list(warnings) or [str(reason)],
    )


def compute_restricted_fgop_sf_sd_features(
    logits: np.ndarray,
    canonical_phones: Sequence[str],
    *,
    vocab: Mapping[str, int],
    blank_id: int,
    model_id: str = "",
    revision: str = "",
    fallback_to_unrestricted: bool = True,
) -> RestrictedSegmentationFreeGopResult:
    """Compute target-position-specific restricted substitution features.

    The established unrestricted implementation accepts one substitution set
    for the entire utterance.  To avoid changing that audited code path, this
    wrapper runs it once per target position with that position's Japanese
    candidate set and keeps only the corresponding row.  This is intentionally
    slower but transparent for Stage-0 research; an optimized graph version can
    replace it only after criterion evidence supports the restricted policy.
    """
    phones = [str(phone) for phone in canonical_phones if str(phone)]
    if not phones:
        return _unavailable(
            "empty_canonical_phone_sequence",
            model_id=model_id,
            revision=revision,
            canonical_phones=phones,
        )
    missing = sorted({phone for phone in phones if phone not in vocab})
    if missing:
        result = _unavailable(
            "canonical_phone_not_in_logical_vocabulary",
            model_id=model_id,
            revision=revision,
            canonical_phones=phones,
            warnings=["phone_inventory_mismatch"],
        )
        result.summary["missing_phones"] = missing
        return result

    by_position, provenance = substitution_token_ids_by_position(
        phones,
        vocab,
        fallback_to_unrestricted=fallback_to_unrestricted,
    )
    id_to_phone = {int(token_id): str(phone) for phone, token_id in vocab.items()}
    rows: list[RestrictedPhoneFeature] = []
    warnings: list[str] = []

    for index, canonical_phone in enumerate(phones):
        candidate_ids = list(by_position.get(index, ()))
        if not candidate_ids:
            return _unavailable(
                "empty_restricted_candidate_set",
                model_id=model_id,
                revision=revision,
                canonical_phones=phones,
                warnings=[f"empty_candidate_set:{index}:{canonical_phone}"],
            )
        result = compute_enumerated_fgop_sf_sd_features(
            logits,
            phones,
            vocab=vocab,
            blank_id=int(blank_id),
            substitution_token_ids=candidate_ids,
            model_id=model_id,
            revision=revision,
        )
        if not result.available or len(result.evidence) != len(phones):
            return _unavailable(
                "restricted_position_evaluation_failed",
                model_id=model_id,
                revision=revision,
                canonical_phones=phones,
                warnings=[*result.warnings, f"failed_position:{index}:{canonical_phone}"],
            )
        source: SegmentationFreePhoneFeature = result.evidence[index]
        meta = provenance[index]
        candidate_phones = tuple(
            sorted(id_to_phone[token_id] for token_id in candidate_ids if token_id in id_to_phone)
        )
        rows.append(
            RestrictedPhoneFeature(
                phone_index=index,
                canonical_phone=canonical_phone,
                candidate_phones=candidate_phones,
                candidate_count=len(candidate_phones),
                search_policy=str(meta.get("policy") or ""),
                fallback_used=bool(meta.get("fallback_used")),
                canonical_log_posterior=float(source.canonical_log_posterior),
                gop_sf_sd=float(source.gop_sf_sd),
                deletion_lpr=float(source.deletion_log_posterior_ratio),
                substitution_lprs={
                    str(phone): float(value)
                    for phone, value in source.substitution_log_posterior_ratios.items()
                },
                best_noncanonical_type=str(source.best_noncanonical_alternative_type),
                best_noncanonical_phone=(
                    str(source.best_noncanonical_alternative_phone)
                    if source.best_noncanonical_alternative_phone is not None
                    else None
                ),
                best_noncanonical_lpr=float(source.best_noncanonical_log_posterior_ratio),
            )
        )
        warnings.extend(result.warnings)

    values = np.asarray([row.gop_sf_sd for row in rows], dtype=np.float64)
    best = np.asarray([row.best_noncanonical_lpr for row in rows], dtype=np.float64)
    counts = [row.candidate_count for row in rows]
    if not np.isfinite(values).all() or not np.isfinite(best).all():
        return _unavailable(
            "nonfinite_restricted_features",
            model_id=model_id,
            revision=revision,
            canonical_phones=phones,
        )

    return RestrictedSegmentationFreeGopResult(
        available=True,
        method=METHOD,
        model_id=str(model_id),
        revision=str(revision),
        canonical_phones=phones,
        rows=rows,
        candidate_provenance=provenance,
        summary={
            "phone_count": len(rows),
            "candidate_count_min": min(counts),
            "candidate_count_max": max(counts),
            "candidate_count_mean": float(np.mean(counts)),
            "fallback_position_count": sum(row.fallback_used for row in rows),
            "phones_where_noncanonical_outscores_canonical": int(np.sum(best < 0.0)),
            "gop_sf_sd_mean": float(np.mean(values)),
            "gop_sf_sd_median": float(np.median(values)),
            "gop_sf_sd_min": float(np.min(values)),
            "search_space": "position_specific_restricted_japanese_phonology",
            "special_mora_policy": "canonical_plus_deletion_only",
            "criterion_validated_for_japanese_l2": False,
            "interpretation": "restricted_alignment_free_research_feature_not_pronunciation_score",
        },
        warnings=sorted(set(warnings)),
        score_mapped=False,
        product_calibrated=False,
    )


def compare_restricted_vs_unrestricted(
    restricted: RestrictedSegmentationFreeGopResult,
    unrestricted: Any,
) -> Dict[str, Any]:
    """Produce an audit-only RPS-vs-UPS comparison without score fusion."""
    if not restricted.available or not bool(getattr(unrestricted, "available", False)):
        return {"available": False, "reason": "one_or_both_feature_sets_unavailable"}
    if list(restricted.canonical_phones) != list(unrestricted.canonical_phones):
        return {"available": False, "reason": "canonical_phone_sequence_mismatch"}
    if len(restricted.rows) != len(unrestricted.evidence):
        return {"available": False, "reason": "row_count_mismatch"}

    rows = []
    for rps, ups in zip(restricted.rows, unrestricted.evidence):
        rows.append(
            {
                "phone_index": rps.phone_index,
                "canonical_phone": rps.canonical_phone,
                "rps_candidate_count": rps.candidate_count,
                "rps_search_policy": rps.search_policy,
                "rps_gop_sf_sd": rps.gop_sf_sd,
                "ups_gop_sf_sd": float(ups.gop_sf_sd),
                "rps_minus_ups": float(rps.gop_sf_sd - float(ups.gop_sf_sd)),
                "rps_best_noncanonical_phone": rps.best_noncanonical_phone,
                "ups_best_noncanonical_phone": ups.best_noncanonical_alternative_phone,
                "same_best_noncanonical_phone": rps.best_noncanonical_phone == ups.best_noncanonical_alternative_phone,
            }
        )
    return {
        "available": True,
        "rows": rows,
        "interpretation": "RPS_vs_UPS_search_space_diagnostic_not_pronunciation_quality",
        "raw_values_must_not_be_averaged": True,
        "product_score_changed": False,
    }
