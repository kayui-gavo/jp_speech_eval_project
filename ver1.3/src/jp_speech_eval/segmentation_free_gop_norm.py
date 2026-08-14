"""Paper-aligned normalized segmentation-free CTC GOP diagnostics.

This module independently re-implements the SD alternative-graph forward
recursion published with Cao et al., *Segmentation-Free Goodness of
Pronunciation* (IEEE TASLP 2026 / arXiv:2507.16838).  The authors' public
``taslpro26/gop_sf_sd_norm.py`` code uses normalized forward variables over an
arbitrary-token state and reports both the alternative-graph likelihood and
``Occ(i)``.  We reproduce that algorithm in NumPy so it can be regression-
tested and combined with this project's existing LPP/LPR feature extractor.

Important semantics:

* ``Occ(i)`` is occupancy/activation of the alternative graph's central state;
  it is **not** a physical phone duration and must never be exposed as one.
* an individual LPR sign is not a pronunciation-error decision rule;
* these values are research features, not a learner-facing score;
* no product /100 mapping is implemented here.

The implementation is intentionally named ``paper_sd_norm_forward_v1`` rather
than claiming that the whole Japanese assessment pipeline reproduces the paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .japanese_phoneme_gop import ctc_forward_logprob


METHOD = "paper_sd_norm_forward_v1"
REFERENCE_IMPLEMENTATION = "frank613/CTC-based-GOP:taslpro26/gop_sf_sd_norm.py"


@dataclass(frozen=True)
class SdNormForwardResult:
    phone_index: int
    canonical_phone: str
    canonical_log_posterior: float
    denominator_graph_log_posterior: float
    gop_sf_sd_norm: float
    occ_i: float
    frame_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationFreeNormResult:
    available: bool
    model_id: str
    revision: str
    method: str
    canonical_phones: list[str]
    evidence: list[SdNormForwardResult]
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


def _validate_sequence(token_ids: Sequence[int], *, vocab_size: int, blank_id: int) -> list[int]:
    seq = [int(value) for value in token_ids]
    if not seq:
        raise ValueError("canonical token sequence is empty")
    if not (0 <= int(blank_id) < int(vocab_size)):
        raise ValueError("blank id is outside the CTC vocabulary")
    if any(value < 0 or value >= vocab_size for value in seq):
        raise ValueError("canonical token id is outside the CTC vocabulary")
    if any(value == int(blank_id) for value in seq):
        raise ValueError("canonical phone sequence must not contain the CTC blank token")
    return seq


def sd_norm_alternative_graph_forward(
    probabilities: np.ndarray,
    token_ids: Sequence[int],
    *,
    phone_index: int,
    blank_id: int,
) -> tuple[float, float]:
    """Return ``(log p(L_SD), Occ(i))`` using the paper's normalized forward DP.

    ``probabilities`` has shape ``(frames, vocabulary)`` and each frame must sum
    to one.  The recursion uses an extra vocabulary axis only at the wildcard
    phone state.  Forward variables are normalized at each frame; the sum of
    log normalization constants recovers the graph log posterior while the
    normalized wildcard-state mass summed over time gives ``Occ(i)``.

    The disambiguation/skip-path rules follow the authors' public TASLP-2026
    implementation.  This routine intentionally keeps the paper-reference
    all-token alternative semantics; Japanese-specific competitor filtering is
    a separate modeling choice and is not silently mixed into this function.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] <= 0 or probs.shape[1] <= 1:
        raise ValueError("probabilities must have shape (frames, vocabulary)")
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("CTC probabilities must be finite and non-negative")
    frame_sums = np.sum(probs, axis=1)
    if not np.allclose(frame_sums, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("each CTC probability frame must sum to one")

    frames, vocab_size = probs.shape
    seq = _validate_sequence(token_ids, vocab_size=vocab_size, blank_id=int(blank_id))
    pos = int(phone_index)
    if pos < 0 or pos >= len(seq):
        raise ValueError("phone_index is outside the canonical sequence")

    state_count = 2 * len(seq) + 1
    # Deterministic CTC states use vocabulary slot 0 as storage.  The wildcard
    # label state uses the whole last axis to retain which alternative token is
    # active.  This storage axis is independent of the actual CTC blank id.
    alpha = np.zeros((state_count, frames, vocab_size), dtype=np.float64)
    alpha_bar = np.zeros(frames, dtype=np.float64)
    next_label_id = None if pos == len(seq) - 1 else int(seq[pos + 1])

    if pos == 0:
        alpha[0, 0, 0] = probs[0, blank_id]
        # State 2 is deliberately not initialized: skipping the wildcard phone
        # is handled explicitly to avoid duplicate paths, matching the paper code.
        if len(seq) > 1:
            alpha[3, 0, 0] = probs[0, seq[1]]
        alpha[1, 0, :] = probs[0, :]
        alpha[1, 0, blank_id] = 0.0
        if next_label_id is not None:
            alpha[1, 0, next_label_id] = 0.0
        alpha_bar[0] = float(np.sum(alpha[:, 0, :]))
    else:
        alpha[0, 0, 0] = probs[0, blank_id]
        alpha[1, 0, 0] = probs[0, seq[0]]
        alpha_bar[0] = float(alpha[0, 0, 0] + alpha[1, 0, 0])

    if not math.isfinite(alpha_bar[0]) or alpha_bar[0] <= 0.0:
        raise ValueError("alternative graph has zero probability at frame 0")
    alpha[:, 0, :] /= alpha_bar[0]

    wildcard_state = 2 * pos + 1

    for t in range(1, frames):
        lowest_state = state_count - 1 - 2 * (frames - t)
        if (lowest_state - 1) / 2 == pos:
            lowest_state -= 2
        start = max(0, lowest_state)

        for state in range(start, state_count):
            label_pos = int((state - 1) / 2)

            if state % 2 == 0:  # blank state
                if state == 0:
                    alpha[state, t, 0] = alpha[state, t - 1, 0] * probs[t, blank_id]
                elif state - 1 == wildcard_state:
                    incoming = alpha[state - 1, t - 1, :].copy()
                    incoming[blank_id] = 0.0
                    alpha[state, t, 0] = (
                        alpha[state, t - 1, 0] + float(np.sum(incoming))
                    ) * probs[t, blank_id]
                else:
                    alpha[state, t, 0] = (
                        alpha[state, t - 1, 0] + alpha[state - 1, t - 1, 0]
                    ) * probs[t, blank_id]
                continue

            if pos != label_pos and pos != label_pos - 1:
                label_id = seq[label_pos]
                if state == 1 or seq[label_pos] == seq[label_pos - 1]:
                    incoming = alpha[state, t - 1, 0] + alpha[state - 1, t - 1, 0]
                else:
                    incoming = (
                        alpha[state, t - 1, 0]
                        + alpha[state - 1, t - 1, 0]
                        + alpha[state - 2, t - 1, 0]
                    )
                alpha[state, t, 0] = incoming * probs[t, label_id]
                continue

            if pos == label_pos - 1:
                # Exit from the wildcard state into the following canonical
                # label, including the paper's explicit deletion/skip handling.
                label_id = seq[label_pos]
                incoming_wildcard = alpha[state - 2, t - 1, :].copy()
                incoming_wildcard[blank_id] = 0.0
                incoming_wildcard[label_id] = 0.0
                wildcard_sum = float(np.sum(incoming_wildcard))

                if label_pos - 2 < 0 or seq[label_pos - 2] == label_id:
                    skip_token = 0.0
                else:
                    skip_token = alpha[state - 4, t - 1, 0] * probs[t, label_id]
                skip_empty = alpha[state - 3, t - 1, 0] * probs[t, label_id]
                alpha[state, t, 0] = (
                    alpha[state, t - 1, 0]
                    + alpha[state - 1, t - 1, 0]
                    + wildcard_sum
                ) * probs[t, label_id] + skip_empty + skip_token
                continue

            # Current label is the wildcard/arbitrary token.  Staying in the
            # state preserves token identity (identity transition); entering it
            # can occur from the preceding blank and, where legal, by skip.
            if state == 1:
                empty_prob = alpha[state - 1, t - 1, 0] * probs[t, :].copy()
                empty_prob[blank_id] = 0.0
                if next_label_id is not None:
                    empty_prob[next_label_id] = 0.0
                stay = alpha[state, t - 1, :] * probs[t, :]
                if next_label_id is not None:
                    stay[next_label_id] = 0.0
                alpha[state, t, :] = stay + empty_prob
            else:
                skip_prob = alpha[state - 2, t - 1, 0] * probs[t, :].copy()
                skip_prob[seq[label_pos - 1]] = 0.0
                skip_prob[blank_id] = 0.0
                empty_prob = alpha[state - 1, t - 1, 0] * probs[t, :].copy()
                empty_prob[blank_id] = 0.0
                if next_label_id is not None:
                    skip_prob[next_label_id] = 0.0
                    empty_prob[next_label_id] = 0.0
                stay = alpha[state, t - 1, :] * probs[t, :]
                if next_label_id is not None:
                    stay[next_label_id] = 0.0
                alpha[state, t, :] = stay + skip_prob + empty_prob

        alpha_bar[t] = float(np.sum(alpha[:, t, :]))
        if not math.isfinite(alpha_bar[t]) or alpha_bar[t] <= 0.0:
            raise ValueError(f"alternative graph has zero/non-finite probability at frame {t}")
        alpha[:, t, :] /= alpha_bar[t]

    occ_i = float(np.sum(alpha[wildcard_state, :, :]))
    graph_log_posterior = float(np.sum(np.log(alpha_bar)))
    return graph_log_posterior, occ_i


def compute_segmentation_free_norm_features(
    logits: np.ndarray,
    canonical_phones: Sequence[str],
    *,
    vocab: Mapping[str, int],
    blank_id: int,
    model_id: str = "",
    revision: str = "",
) -> SegmentationFreeNormResult:
    """Compute paper-aligned SD-graph GOP normalization + ``Occ(i)`` per phone."""
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
    warnings: list[str] = []
    for index, phone in enumerate(phones):
        denominator_lp, occ_i = sd_norm_alternative_graph_forward(
            probs,
            token_ids,
            phone_index=index,
            blank_id=int(blank_id),
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

    occ_values = np.asarray([row.occ_i for row in rows], dtype=np.float64)
    gop_values = np.asarray([row.gop_sf_sd_norm for row in rows], dtype=np.float64)
    if np.any(~np.isfinite(occ_values)) or np.any(occ_values < 0):
        warnings.append("invalid_occ_i_detected")
    return SegmentationFreeNormResult(
        available=True,
        model_id=model_id,
        revision=revision,
        method=METHOD,
        canonical_phones=phones,
        evidence=rows,
        summary={
            "reference_implementation": REFERENCE_IMPLEMENTATION,
            "phone_count": len(phones),
            "frame_count": int(raw.shape[0]),
            "gop_sf_sd_norm_mean": float(np.mean(gop_values)),
            "gop_sf_sd_norm_median": float(np.median(gop_values)),
            "occ_i_mean": float(np.mean(occ_values)),
            "occ_i_min": float(np.min(occ_values)),
            "occ_i_max": float(np.max(occ_values)),
            "occ_i_is_physical_phone_duration": False,
            "individual_feature_is_pronunciation_decision": False,
            "requires_labeled_downstream_validation": True,
            "score_mapped": False,
        },
        warnings=warnings,
        score_mapped=False,
        product_calibrated=False,
    )
