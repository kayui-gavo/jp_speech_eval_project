"""Restricted-substitution alignment-free GOP research features.

Recent alignment-free GOP work reports benefits from restricting substitution
alternatives using phonological knowledge instead of enumerating an entire
phone inventory. This module applies that idea conservatively to the Japanese
phone-CTC research stack without changing the established unrestricted path.

Important boundaries:
- candidate neighborhoods are Japanese research hypotheses, not learner labels;
- ``N`` and ``cl`` use canonical-vs-deletion evidence only here;
- no value is mapped to pronunciation correctness or a learner-facing /100;
- the raw restricted SD-GOP denominator changes with the candidate set, so raw
  RPS GOP values are *not* directly comparable across phone types or against
  unrestricted SD-GOP values. RPS/UPS are compared by search-space behavior,
  target-specific LPRs and criterion performance, never by raw-score deltas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .ctc_sequence import ctc_forward_logprob_vectorized
from .japanese_phone_substitutions import substitution_token_ids_by_position


METHOD = "restricted_enumerated_fgop_ctc_sf_sd_v1"
CTC_FORWARD_IMPL = "rolling_vectorized_exact_ctc_v1"


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
    extra_summary: Mapping[str, Any] | None = None,
) -> RestrictedSegmentationFreeGopResult:
    summary: Dict[str, Any] = {"reason": str(reason)}
    if extra_summary:
        summary.update(dict(extra_summary))
    return RestrictedSegmentationFreeGopResult(
        available=False,
        method=METHOD,
        model_id=str(model_id),
        revision=str(revision),
        canonical_phones=[str(x) for x in canonical_phones],
        rows=[],
        candidate_provenance=[],
        summary=summary,
        warnings=list(warnings) or [str(reason)],
    )


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 1:
        raise ValueError("CTC logits must have shape (frames, vocabulary)")
    if not np.isfinite(values).all():
        raise ValueError("CTC logits contain non-finite values")
    maxima = np.max(values, axis=1, keepdims=True)
    shifted = values - maxima
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def _logsumexp(values: Sequence[float]) -> float:
    finite = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if finite.size == 0:
        return -np.inf
    maximum = float(np.max(finite))
    return maximum + math.log(float(np.sum(np.exp(finite - maximum))))


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
    """Compute exact CTC RPS substitution/deletion features per target phone.

    The canonical CTC posterior is computed once. Only the alternatives for the
    current target position are then evaluated, so complexity scales with the
    actual restricted candidate count. Fixed-sequence posteriors use the exact
    rolling/vectorized CTC recurrence, regression-tested against the earlier
    scalar implementation including repeated-label paths.

    ``gop_sf_sd`` is retained as a model feature because a future criterion
    model may learn from it. It must not be compared directly across rows with
    different candidate sets: changing the denominator search space changes its
    numerical scale. Target-specific LPRs have a clearer within-contrast
    interpretation and are preferred for controlled localization tests.
    """
    raw = np.asarray(logits, dtype=np.float64)
    try:
        log_probs = _log_softmax(raw)
    except ValueError as exc:
        return _unavailable(
            "invalid_ctc_logits",
            model_id=model_id,
            revision=revision,
            canonical_phones=canonical_phones,
            warnings=[str(exc)],
        )

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
        return _unavailable(
            "canonical_phone_not_in_logical_vocabulary",
            model_id=model_id,
            revision=revision,
            canonical_phones=phones,
            warnings=["phone_inventory_mismatch"],
            extra_summary={"missing_phones": missing},
        )

    token_ids = [int(vocab[phone]) for phone in phones]
    canonical_lp = ctc_forward_logprob_vectorized(
        log_probs, token_ids, blank_id=int(blank_id)
    )
    if not math.isfinite(canonical_lp):
        return _unavailable(
            "canonical_ctc_logposterior_nonfinite",
            model_id=model_id,
            revision=revision,
            canonical_phones=phones,
        )

    by_position, provenance = substitution_token_ids_by_position(
        phones,
        vocab,
        fallback_to_unrestricted=fallback_to_unrestricted,
    )
    id_to_phone = {int(token_id): str(phone) for phone, token_id in vocab.items()}
    rows: list[RestrictedPhoneFeature] = []
    warnings: list[str] = []

    for index, (canonical_phone, canonical_id) in enumerate(zip(phones, token_ids)):
        candidate_ids = list(by_position.get(index, ()))
        if not candidate_ids:
            return _unavailable(
                "empty_restricted_candidate_set",
                model_id=model_id,
                revision=revision,
                canonical_phones=phones,
                warnings=[f"empty_candidate_set:{index}:{canonical_phone}"],
            )

        substitution_logps: Dict[str, float] = {}
        substitution_lprs: Dict[str, float] = {}
        denominator_terms: list[float] = []
        for alternative_id in candidate_ids:
            sequence = list(token_ids)
            sequence[index] = int(alternative_id)
            alternative_lp = ctc_forward_logprob_vectorized(
                log_probs, sequence, blank_id=int(blank_id)
            )
            alternative_phone = id_to_phone[int(alternative_id)]
            substitution_logps[alternative_phone] = float(alternative_lp)
            substitution_lprs[alternative_phone] = float(canonical_lp - alternative_lp)
            denominator_terms.append(float(alternative_lp))

        deleted_sequence = token_ids[:index] + token_ids[index + 1 :]
        deletion_lp = ctc_forward_logprob_vectorized(
            log_probs, deleted_sequence, blank_id=int(blank_id)
        )
        deletion_lpr = float(canonical_lp - deletion_lp)
        denominator_terms.append(float(deletion_lp))
        denominator_lp = _logsumexp(denominator_terms)
        gop_sf_sd = float(canonical_lp - denominator_lp)

        noncanonical: list[tuple[float, str, str | None]] = [
            (lp, "substitution", phone)
            for phone, lp in substitution_logps.items()
            if int(vocab[phone]) != canonical_id
        ]
        noncanonical.append((float(deletion_lp), "deletion", None))
        best_lp, best_type, best_phone = max(noncanonical, key=lambda item: item[0])
        meta = provenance[index]
        candidate_phones = tuple(
            sorted(id_to_phone[token_id] for token_id in candidate_ids if token_id in id_to_phone)
        )
        best_lpr = float(canonical_lp - best_lp)
        if best_lpr < 0.0:
            warnings.append("one_or_more_restricted_alternatives_outscore_canonical_sequence")

        rows.append(
            RestrictedPhoneFeature(
                phone_index=index,
                canonical_phone=canonical_phone,
                candidate_phones=candidate_phones,
                candidate_count=len(candidate_phones),
                search_policy=str(meta.get("policy") or ""),
                fallback_used=bool(meta.get("fallback_used")),
                canonical_log_posterior=float(canonical_lp),
                gop_sf_sd=gop_sf_sd,
                deletion_lpr=deletion_lpr,
                substitution_lprs=substitution_lprs,
                best_noncanonical_type=best_type,
                best_noncanonical_phone=best_phone,
                best_noncanonical_lpr=best_lpr,
            )
        )

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
            "canonical_ctc_log_posterior": float(canonical_lp),
            "ctc_forward_implementation": CTC_FORWARD_IMPL,
            "ctc_forward_scalar_reference_regression_tested": True,
            "frame_count": int(raw.shape[0]),
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
            "alignment_free": True,
            "requires_external_phone_boundaries": False,
            "includes_substitution": True,
            "includes_deletion": True,
            "includes_insertion": False,
            "phone_dependent_denominator": True,
            "candidate_count_affects_raw_denominator": True,
            "cross_phone_raw_gop_comparison_allowed": False,
            "rps_vs_ups_raw_gop_direct_comparison_allowed": False,
            "preferred_controlled_diagnostic": "target_specific_substitution_or_deletion_LPR",
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
    """Compare RPS/UPS search behavior without subtracting raw GOP scales.

    Because the denominator changes with the candidate set, an RPS-minus-UPS
    raw GOP delta has no stable pronunciation interpretation. This diagnostic
    therefore reports candidate reduction and alternative identity only.
    """
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
                "ups_candidate_count": len(ups.substitution_log_posterior_ratios),
                "rps_best_noncanonical_type": rps.best_noncanonical_type,
                "rps_best_noncanonical_phone": rps.best_noncanonical_phone,
                "ups_best_noncanonical_type": ups.best_noncanonical_alternative_type,
                "ups_best_noncanonical_phone": ups.best_noncanonical_alternative_phone,
                "same_best_noncanonical_type": rps.best_noncanonical_type == ups.best_noncanonical_alternative_type,
                "same_best_noncanonical_phone": rps.best_noncanonical_phone == ups.best_noncanonical_alternative_phone,
                "raw_gop_values_intentionally_omitted": True,
            }
        )
    rps_counts = [row["rps_candidate_count"] for row in rows]
    ups_counts = [row["ups_candidate_count"] for row in rows]
    return {
        "available": True,
        "rows": rows,
        "candidate_count_reduction_mean": float(np.mean(ups_counts) - np.mean(rps_counts)),
        "candidate_count_ratio_mean": float(np.mean(rps_counts) / np.mean(ups_counts)) if np.mean(ups_counts) > 0 else None,
        "phone_dependent_denominator": True,
        "rps_vs_ups_raw_gop_direct_comparison_allowed": False,
        "raw_values_must_not_be_averaged": True,
        "interpretation": "RPS_vs_UPS_search_space_diagnostic_not_pronunciation_quality",
        "product_score_changed": False,
    }
