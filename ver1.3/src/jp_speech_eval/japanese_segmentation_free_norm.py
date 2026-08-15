"""Construct-aware Japanese normalized segmentation-free GOP features.

The paper-style normalized SD alternative graph is useful for ordinary phones,
but Japanese special morae cannot share exactly the same competitor semantics:
``N`` and ``cl`` are primarily timing/context constructs in this product, and
we intentionally exclude them from ordinary segmental clarity competitors.

If an ordinary-phone wildcard inventory is blindly reused for a canonical
``N``/``cl`` row, the denominator no longer even contains the canonical token,
which makes that row semantically inconsistent with ordinary rows. This module
fixes that by selecting the wildcard set per target role:

* ordinary phone -> ordinary segmental phone inventory;
* ``N`` / ``cl`` -> canonical token only, with the graph's skip path providing
  canonical-vs-deletion support.

The resulting raw values still are not learner scores and are not assumed to
share one cross-role numerical scale.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .japanese_phoneme_gop import SPECIAL_MORA_TOKENS, ctc_forward_logprob, segmental_competitor_ids
from .segmentation_free_gop_norm import (
    SdNormForwardResult,
    SegmentationFreeNormResult,
    sd_norm_alternative_graph_forward,
)


METHOD = "paper_sd_norm_forward_japanese_construct_aware_v3"


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 1:
        raise ValueError("CTC logits must have shape (frames, vocabulary)")
    if not np.isfinite(values).all():
        raise ValueError("CTC logits contain non-finite values")
    maxima = np.max(values, axis=1, keepdims=True)
    shifted = values - maxima
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def compute_japanese_construct_aware_norm_features(
    logits: np.ndarray,
    canonical_phones: Sequence[str],
    *,
    vocab: Mapping[str, int],
    blank_id: int,
    model_id: str = "",
    revision: str = "",
) -> SegmentationFreeNormResult:
    raw = np.asarray(logits, dtype=np.float64)
    log_probs = _log_softmax(raw)
    probs = np.exp(log_probs)
    phones = [str(phone) for phone in canonical_phones if str(phone)]
    if not phones:
        return SegmentationFreeNormResult(
            available=False,
            model_id=model_id,
            revision=revision,
            method=METHOD,
            canonical_phones=[],
            evidence=[],
            summary={"reason": "empty_canonical_phone_sequence"},
            warnings=["empty_phone_sequence"],
        )
    missing = sorted({phone for phone in phones if phone not in vocab})
    if missing:
        return SegmentationFreeNormResult(
            available=False,
            model_id=model_id,
            revision=revision,
            method=METHOD,
            canonical_phones=phones,
            evidence=[],
            summary={"reason": "phone_inventory_mismatch", "missing_phones": missing},
            warnings=["phone_inventory_mismatch"],
        )

    ordinary_ids = segmental_competitor_ids(vocab, blank_id=int(blank_id))
    if not ordinary_ids and any(phone not in SPECIAL_MORA_TOKENS for phone in phones):
        return SegmentationFreeNormResult(
            available=False,
            model_id=model_id,
            revision=revision,
            method=METHOD,
            canonical_phones=phones,
            evidence=[],
            summary={"reason": "no_ordinary_segmental_wildcard_inventory"},
            warnings=["empty_ordinary_phone_inventory"],
        )

    token_ids = [int(vocab[phone]) for phone in phones]
    canonical_lp = ctc_forward_logprob(log_probs, token_ids, blank_id=int(blank_id))
    if not math.isfinite(canonical_lp):
        return SegmentationFreeNormResult(
            available=False,
            model_id=model_id,
            revision=revision,
            method=METHOD,
            canonical_phones=phones,
            evidence=[],
            summary={"reason": "canonical_ctc_logposterior_nonfinite"},
            warnings=["canonical_ctc_logposterior_nonfinite"],
        )

    rows: list[SdNormForwardResult] = []
    special_indices: list[int] = []
    ordinary_indices: list[int] = []
    row_policies: list[str] = []
    for index, phone in enumerate(phones):
        if phone in SPECIAL_MORA_TOKENS:
            wildcard_ids = [int(vocab[phone])]
            special_indices.append(index)
            row_policies.append("special_mora_canonical_plus_deletion_only")
        else:
            wildcard_ids = ordinary_ids
            ordinary_indices.append(index)
            row_policies.append("ordinary_segmental_phone_only_wildcard")
        denominator_lp, occ_i = sd_norm_alternative_graph_forward(
            probs,
            token_ids,
            phone_index=index,
            blank_id=int(blank_id),
            wildcard_token_ids=wildcard_ids,
        )
        rows.append(
            SdNormForwardResult(
                phone_index=index,
                canonical_phone=phone,
                canonical_log_posterior=float(canonical_lp),
                denominator_graph_log_posterior=float(denominator_lp),
                gop_sf_sd_norm=float(canonical_lp - denominator_lp),
                occ_i=float(occ_i),
                frame_count=int(raw.shape[0]),
            )
        )

    values = np.asarray([row.gop_sf_sd_norm for row in rows], dtype=np.float64)
    occ = np.asarray([row.occ_i for row in rows], dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(occ).all():
        return SegmentationFreeNormResult(
            available=False,
            model_id=model_id,
            revision=revision,
            method=METHOD,
            canonical_phones=phones,
            evidence=[],
            summary={"reason": "nonfinite_construct_aware_norm_feature"},
            warnings=["nonfinite_construct_aware_norm_feature"],
        )

    id_to_phone = {int(index): str(phone) for phone, index in vocab.items()}
    ordinary_inventory = [id_to_phone[token_id] for token_id in ordinary_ids]
    return SegmentationFreeNormResult(
        available=True,
        model_id=model_id,
        revision=revision,
        method=METHOD,
        canonical_phones=phones,
        evidence=rows,
        summary={
            "phone_count": len(rows),
            "frame_count": int(raw.shape[0]),
            "ordinary_segmental_row_count": len(ordinary_indices),
            "special_mora_row_count": len(special_indices),
            "ordinary_segmental_indices": ordinary_indices,
            "special_mora_indices": special_indices,
            "row_policies": row_policies,
            "ordinary_wildcard_phone_inventory": ordinary_inventory,
            "ordinary_wildcard_excludes_special_mora": True,
            "special_mora_row_policy": "canonical_token_plus_graph_deletion_path_only",
            "special_mora_normalized_value_directly_comparable_to_ordinary_rows": False,
            "cross_role_raw_gop_comparison_allowed": False,
            "occ_i_is_physical_phone_duration": False,
            "canonical_log_posterior_is_utterance_sequence_level": True,
            "individual_feature_is_pronunciation_decision": False,
            "requires_labeled_downstream_validation": True,
            "score_mapped": False,
        },
        warnings=[],
        score_mapped=False,
        product_calibrated=False,
    )
