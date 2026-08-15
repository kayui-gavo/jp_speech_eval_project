"""Alignment-free Japanese phone features inspired by FGOP-CTC-SF-SD.

For each canonical phone position this module evaluates exact fixed-sequence CTC
posteriors for the canonical target, one-position substitutions, and deletion.
No external phone boundary is required.

Japanese adaptation:

* ordinary clarity positions compete only with ordinary segmental phones;
* ``N`` and ``cl`` remain canonical targets but use canonical-vs-deletion
  evidence only in this generic Japanese path;
* pause/control tokens are never pronunciation alternatives;
* raw SD-GOP/LPR values remain research features and are not mapped to /100.

The implementation enumerates alternatives transparently. The optimized
normalized SD graph + ``Occ(i)`` lives in ``segmentation_free_gop_norm``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .japanese_phoneme_gop import (
    SPECIAL_MORA_TOKENS,
    JapanesePhoneCtcBackend,
    ctc_forward_logprob,
    project_japanese_ctc_logits,
    sanitize_canonical_phones,
    segmental_competitor_ids,
)


METHOD = "enumerated_fgop_ctc_sf_sd_japanese_v2"


@dataclass(frozen=True)
class SegmentationFreePhoneFeature:
    phone_index: int
    canonical_phone: str
    canonical_log_posterior: float
    denominator_sd_log_posterior: float
    gop_sf_sd: float
    deletion_log_posterior: float
    deletion_log_posterior_ratio: float
    substitution_log_posterior_ratios: Dict[str, float]
    best_noncanonical_alternative_type: str
    best_noncanonical_alternative_phone: Optional[str]
    best_noncanonical_log_posterior: float
    best_noncanonical_log_posterior_ratio: float
    left_context_phone_count: int
    right_context_phone_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationFreeGopResult:
    available: bool
    model_id: str
    revision: str
    method: str
    canonical_phones: list[str]
    feature_phone_inventory: list[str]
    evidence: list[SegmentationFreePhoneFeature]
    summary: Dict[str, Any]
    warnings: list[str]
    score_mapped: bool = False
    product_calibrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = [row.to_dict() for row in self.evidence]
        return payload


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


def _unavailable(
    *,
    model_id: str,
    revision: str,
    canonical_phones: Sequence[str],
    reason: str,
    warning: str,
) -> SegmentationFreeGopResult:
    return SegmentationFreeGopResult(
        available=False,
        model_id=str(model_id),
        revision=str(revision),
        method=METHOD,
        canonical_phones=list(canonical_phones),
        feature_phone_inventory=[],
        evidence=[],
        summary={"reason": str(reason)},
        warnings=[str(warning)],
        score_mapped=False,
        product_calibrated=False,
    )


def _ordinary_substitution_ids(
    vocab: Mapping[str, int],
    *,
    blank_id: int,
    supplied: Optional[Sequence[int]],
    vocab_size: int,
) -> list[int]:
    legal_default = set(segmental_competitor_ids(vocab, blank_id=int(blank_id)))
    if supplied is None:
        return sorted(legal_default)
    return sorted({
        int(token_id)
        for token_id in supplied
        if 0 <= int(token_id) < int(vocab_size)
        and int(token_id) != int(blank_id)
        and int(token_id) in legal_default
    })


def compute_enumerated_fgop_sf_sd_features(
    logits: np.ndarray,
    canonical_phones: Sequence[str],
    *,
    vocab: Mapping[str, int],
    blank_id: int,
    substitution_token_ids: Optional[Sequence[int]] = None,
    model_id: str = "",
    revision: str = "",
) -> SegmentationFreeGopResult:
    """Compute exact alignment-free substitution/deletion features.

    Ordinary phone positions use the ordinary segmental candidate inventory and
    always include the canonical phone itself in the SD denominator. Special
    mora targets ``N``/``cl`` use only canonical + deletion here; their main
    educational interpretation still requires dedicated timing/context
    evidence.
    """
    raw = np.asarray(logits, dtype=np.float64)
    log_probs = _log_softmax(raw)
    phones = [str(phone) for phone in canonical_phones if str(phone)]
    if not phones:
        return _unavailable(
            model_id=model_id,
            revision=revision,
            canonical_phones=[],
            reason="empty_canonical_phone_sequence",
            warning="empty_phone_sequence",
        )
    missing = sorted({phone for phone in phones if phone not in vocab})
    if missing:
        result = _unavailable(
            model_id=model_id,
            revision=revision,
            canonical_phones=phones,
            reason="canonical_phone_not_in_logical_vocabulary",
            warning="phone_inventory_mismatch",
        )
        return SegmentationFreeGopResult(
            **{**result.__dict__, "summary": {**result.summary, "missing_phones": missing}}
        )

    token_ids = [int(vocab[phone]) for phone in phones]
    ordinary_ids = _ordinary_substitution_ids(
        vocab,
        blank_id=int(blank_id),
        supplied=substitution_token_ids,
        vocab_size=raw.shape[1],
    )
    if not ordinary_ids:
        raise ValueError("no ordinary segmental substitution tokens are available")
    id_to_phone = {int(token_id): str(phone) for phone, token_id in vocab.items()}
    ordinary_phones = [id_to_phone[token_id] for token_id in ordinary_ids]

    canonical_lp = ctc_forward_logprob(log_probs, token_ids, blank_id=int(blank_id))
    if not math.isfinite(canonical_lp):
        return _unavailable(
            model_id=model_id,
            revision=revision,
            canonical_phones=phones,
            reason="canonical_ctc_logposterior_nonfinite",
            warning="canonical_ctc_logposterior_nonfinite",
        )

    rows: list[SegmentationFreePhoneFeature] = []
    candidate_counts: list[int] = []
    for index, (canonical_phone, canonical_id) in enumerate(zip(phones, token_ids)):
        if canonical_phone in SPECIAL_MORA_TOKENS:
            position_ids = [canonical_id]
        else:
            position_ids = sorted(set(ordinary_ids) | {canonical_id})
        candidate_counts.append(len(position_ids))

        substitution_logps: Dict[str, float] = {}
        substitution_lprs: Dict[str, float] = {}
        alternative_logps: list[float] = []
        for alternative_id in position_ids:
            sequence = list(token_ids)
            sequence[index] = int(alternative_id)
            alternative_lp = ctc_forward_logprob(log_probs, sequence, blank_id=int(blank_id))
            alternative_phone = id_to_phone[int(alternative_id)]
            substitution_logps[alternative_phone] = float(alternative_lp)
            substitution_lprs[alternative_phone] = float(canonical_lp - alternative_lp)
            alternative_logps.append(float(alternative_lp))

        deleted_sequence = token_ids[:index] + token_ids[index + 1 :]
        deletion_lp = ctc_forward_logprob(log_probs, deleted_sequence, blank_id=int(blank_id))
        deletion_lpr = float(canonical_lp - deletion_lp)
        alternative_logps.append(float(deletion_lp))

        denominator_lp = _logsumexp(alternative_logps)
        gop_sf_sd = float(canonical_lp - denominator_lp)
        noncanonical_candidates: list[tuple[float, str, Optional[str]]] = [
            (float(lp), "substitution", phone)
            for phone, lp in substitution_logps.items()
            if int(vocab[phone]) != canonical_id
        ]
        noncanonical_candidates.append((float(deletion_lp), "deletion", None))
        best_lp, best_type, best_phone = max(noncanonical_candidates, key=lambda item: item[0])

        rows.append(
            SegmentationFreePhoneFeature(
                phone_index=index,
                canonical_phone=canonical_phone,
                canonical_log_posterior=float(canonical_lp),
                denominator_sd_log_posterior=float(denominator_lp),
                gop_sf_sd=gop_sf_sd,
                deletion_log_posterior=float(deletion_lp),
                deletion_log_posterior_ratio=deletion_lpr,
                substitution_log_posterior_ratios=substitution_lprs,
                best_noncanonical_alternative_type=best_type,
                best_noncanonical_alternative_phone=best_phone,
                best_noncanonical_log_posterior=float(best_lp),
                best_noncanonical_log_posterior_ratio=float(canonical_lp - best_lp),
                left_context_phone_count=index,
                right_context_phone_count=len(phones) - index - 1,
            )
        )

    values = np.asarray([row.gop_sf_sd for row in rows], dtype=np.float64)
    best_ratios = np.asarray([row.best_noncanonical_log_posterior_ratio for row in rows], dtype=np.float64)
    warnings: list[str] = []
    if np.any(best_ratios < 0):
        warnings.append("one_or_more_noncanonical_sd_alternatives_outscore_canonical_sequence")

    special_count = sum(phone in SPECIAL_MORA_TOKENS for phone in phones)
    return SegmentationFreeGopResult(
        available=True,
        model_id=str(model_id),
        revision=str(revision),
        method=METHOD,
        canonical_phones=phones,
        feature_phone_inventory=ordinary_phones,
        evidence=rows,
        summary={
            "canonical_ctc_log_posterior": float(canonical_lp),
            "phone_count": len(phones),
            "ordinary_feature_phone_inventory_size": len(ordinary_phones),
            "special_mora_position_count": special_count,
            "candidate_count_min": min(candidate_counts),
            "candidate_count_max": max(candidate_counts),
            "candidate_count_varies_by_phone": len(set(candidate_counts)) > 1,
            "ordinary_position_includes_canonical_substitution": True,
            "special_mora_policy": "canonical_plus_deletion_only",
            "gop_sf_sd_mean": float(np.mean(values)),
            "gop_sf_sd_median": float(np.median(values)),
            "gop_sf_sd_min": float(np.min(values)),
            "best_noncanonical_lpr_min": float(np.min(best_ratios)),
            "phones_where_noncanonical_outscores_canonical": int(np.sum(best_ratios < 0)),
            "alignment_free": True,
            "requires_external_phone_boundaries": False,
            "includes_substitution": True,
            "includes_deletion": True,
            "includes_insertion": False,
            "occ_activation_normalization_implemented": False,
            "efficient_alternative_graph_implemented": False,
            "cross_phone_raw_gop_comparison_requires_caution": True,
            "individual_lpr_sign_is_pronunciation_error_rule": False,
            "interpretation": "raw_alignment_free_Japanese_phone_features_not_pronunciation_score",
        },
        warnings=warnings,
        score_mapped=False,
        product_calibrated=False,
    )


def evaluate_backend_fgop_sf_sd_shadow(
    backend: JapanesePhoneCtcBackend,
    audio: np.ndarray,
    canonical_phones: Sequence[str],
    *,
    sr: int = 16000,
) -> SegmentationFreeGopResult:
    """Infer logical Japanese CTC logits and compute alignment-free features."""
    backend._load()  # package-private research backend; validates pinned model contract
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    clean_phones, _dropped = sanitize_canonical_phones(canonical_phones)
    model_id = str(backend.model_id)
    revision = str(backend.revision)

    if waveform.size == 0:
        return _unavailable(model_id=model_id, revision=revision, canonical_phones=clean_phones, reason="empty_audio", warning="empty_audio")
    if not np.isfinite(waveform).all():
        return _unavailable(model_id=model_id, revision=revision, canonical_phones=clean_phones, reason="nonfinite_audio", warning="nonfinite_audio")
    if int(sr) != backend._expected_sample_rate():
        return _unavailable(model_id=model_id, revision=revision, canonical_phones=clean_phones, reason="sampling_rate_mismatch", warning="sampling_rate_mismatch")

    inputs = backend.processor(waveform, sampling_rate=sr, return_tensors="pt")
    model_inputs = {key: value.to(backend.device) for key, value in inputs.items()}
    with backend._torch.no_grad():
        output = backend.model(**model_inputs)
    raw_logits = output.logits.squeeze(0).detach().cpu().numpy()
    raw_vocab = backend.vocabulary()
    if raw_logits.ndim != 2:
        return _unavailable(model_id=model_id, revision=revision, canonical_phones=clean_phones, reason="unexpected_model_output_shape", warning="unexpected_model_output_shape")
    config_vocab = int(getattr(backend.model.config, "vocab_size", 0) or 0)
    if raw_logits.shape[1] != config_vocab:
        return _unavailable(model_id=model_id, revision=revision, canonical_phones=clean_phones, reason="model_output_vocab_size_mismatch", warning="model_output_vocab_size_mismatch")

    logical_logits, logical_vocab, _projection = project_japanese_ctc_logits(raw_logits, raw_vocab)
    if "PAD" not in logical_vocab:
        return _unavailable(model_id=model_id, revision=revision, canonical_phones=clean_phones, reason="logical_blank_token_missing", warning="logical_blank_token_missing")
    return compute_enumerated_fgop_sf_sd_features(
        logical_logits,
        clean_phones,
        vocab=logical_vocab,
        blank_id=int(logical_vocab["PAD"]),
        model_id=model_id,
        revision=revision,
    )
