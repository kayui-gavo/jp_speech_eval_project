"""Exact CTC fixed-sequence probability utilities.

The pronunciation research stack evaluates many target substitutions/deletions.
A Python loop over every CTC state for every alternative made Stage-0 native
preflights unnecessarily expensive.  This module implements the same standard
CTC forward recurrence with a rolling, vectorized state axis.

It computes a sequence likelihood only.  It performs no pronunciation scoring,
normalization, or learner-error decision.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _extended_labels(token_ids: Sequence[int], blank_id: int) -> np.ndarray:
    tokens = np.asarray([int(value) for value in token_ids], dtype=np.int64)
    labels = np.empty(2 * len(tokens) + 1, dtype=np.int64)
    labels[0::2] = int(blank_id)
    if tokens.size:
        labels[1::2] = tokens
    return labels


def _logsumexp_pair(left: float, right: float) -> float:
    return float(np.logaddexp(float(left), float(right)))


def ctc_forward_logprob_vectorized(
    log_probs: np.ndarray,
    token_ids: Sequence[int],
    *,
    blank_id: int,
) -> float:
    """Return exact standard-CTC log p(``token_ids`` | ``log_probs``).

    ``log_probs`` must already be frame-wise log probabilities with shape
    ``(frames, vocabulary)``.  Repeated target labels obey the normal CTC rule:
    a two-state skip is disabled when the current label equals the label two
    states earlier, forcing an intervening blank.
    """
    probs = np.asarray(log_probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] <= 0 or probs.shape[1] <= 1:
        raise ValueError("log_probs must have shape (frames, vocabulary)")
    if not np.isfinite(probs).all():
        raise ValueError("log_probs contain non-finite values")
    blank = int(blank_id)
    if blank < 0 or blank >= probs.shape[1]:
        raise ValueError("blank id is outside CTC vocabulary")
    tokens = [int(value) for value in token_ids]
    if any(value < 0 or value >= probs.shape[1] for value in tokens):
        raise ValueError("token id is outside CTC vocabulary")
    if any(value == blank for value in tokens):
        raise ValueError("target sequence must not contain CTC blank")

    labels = _extended_labels(tokens, blank)
    states = labels.size
    previous = np.full(states, -np.inf, dtype=np.float64)
    previous[0] = probs[0, blank]
    if states > 1:
        previous[1] = probs[0, labels[1]]

    skip_allowed = np.zeros(states, dtype=bool)
    if states > 2:
        indices = np.arange(2, states)
        skip_allowed[2:] = (
            (labels[2:] != blank)
            & (labels[2:] != labels[:-2])
        )

    for frame in range(1, probs.shape[0]):
        advance_one = np.full(states, -np.inf, dtype=np.float64)
        advance_one[1:] = previous[:-1]
        incoming = np.logaddexp(previous, advance_one)

        if states > 2:
            advance_two = np.full(states, -np.inf, dtype=np.float64)
            advance_two[2:] = previous[:-2]
            incoming = np.where(
                skip_allowed,
                np.logaddexp(incoming, advance_two),
                incoming,
            )
        previous = incoming + probs[frame, labels]

    if states == 1:
        return float(previous[0])
    return _logsumexp_pair(previous[-1], previous[-2])
