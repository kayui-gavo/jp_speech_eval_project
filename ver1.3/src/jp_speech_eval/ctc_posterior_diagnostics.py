"""Model-level CTC posterior/logit diagnostics for pronunciation research.

Standard CTC can be very peaky: most acoustic evidence collapses onto sparse
frames, which can make posterior-derived GOP unstable. These diagnostics make
that model property explicit before any Japanese learner criterion mapping.

The v2 schema also records shift-invariant logit margins. Recent pronunciation
assessment work suggests that logit-domain competition can complement posterior
GOP, but these values remain descriptive shadow evidence until they are tested
against Japanese-L2 human criterion labels.

They are **not** pronunciation scores and deliberately define no universal or
engineering alert threshold. The intended use is descriptive comparison of
pinned phone-CTC backbones followed by criterion validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Sequence

import numpy as np


SCHEMA = "ctc_posterior_diagnostics_v2"


@dataclass(frozen=True)
class CtcPosteriorDiagnostics:
    available: bool
    schema: str
    frame_count: int
    vocabulary_size: int
    phone_token_count: int
    blank_id: int
    top1_posterior_mean: float
    top1_posterior_p95: float
    top1_posterior_max: float
    blank_top1_fraction: float
    blank_posterior_mean: float
    blank_posterior_p95: float
    phone_mass_mean: float
    phone_mass_p05: float
    phone_top1_posterior_mean: float
    phone_top1_posterior_p95: float
    phone_top1_margin_mean: float
    full_normalized_entropy_mean: float
    full_normalized_entropy_p05: float
    phone_conditional_normalized_entropy_mean: float
    phone_conditional_normalized_entropy_p05: float
    full_top1_logit_margin_mean: float
    full_top1_logit_margin_p05: float
    phone_top1_logit_margin_mean: float
    phone_top1_logit_margin_p05: float
    blank_minus_phone_top_logit_margin_mean: float
    summary: Dict[str, Any]
    warnings: list[str]
    product_calibrated: bool = False
    score_mapped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 1:
        raise ValueError("CTC logits must have shape (frames, vocabulary)")
    if not np.isfinite(values).all():
        raise ValueError("CTC logits contain non-finite values")
    maxima = np.max(values, axis=1, keepdims=True)
    exp = np.exp(values - maxima)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _normalized_entropy(probabilities: np.ndarray, *, axis_size: int) -> np.ndarray:
    if axis_size <= 1:
        return np.zeros(probabilities.shape[0], dtype=np.float64)
    clipped = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)
    entropy = -np.sum(probabilities * np.log(clipped), axis=1)
    return entropy / math.log(float(axis_size))


def _top1_margin(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] <= 0:
        raise ValueError("margin input must have shape (frames, classes)")
    if arr.shape[1] == 1:
        return np.zeros(arr.shape[0], dtype=np.float64)
    top2 = np.partition(arr, -2, axis=1)[:, -2:]
    return np.max(top2, axis=1) - np.min(top2, axis=1)


def compute_ctc_posterior_diagnostics(
    logits: np.ndarray,
    *,
    blank_id: int,
    phone_token_ids: Sequence[int],
) -> CtcPosteriorDiagnostics:
    """Summarize CTC peakiness/uncertainty without interpreting pronunciation."""
    values = np.asarray(logits, dtype=np.float64)
    probs = _softmax(values)
    frames, vocab_size = probs.shape
    blank = int(blank_id)
    if blank < 0 or blank >= vocab_size:
        raise ValueError("blank id is outside vocabulary")
    phone_ids = sorted({int(value) for value in phone_token_ids if int(value) != blank})
    if not phone_ids:
        raise ValueError("phone token inventory is empty")
    if any(value < 0 or value >= vocab_size for value in phone_ids):
        raise ValueError("phone token id is outside vocabulary")

    top1 = np.max(probs, axis=1)
    top1_ids = np.argmax(probs, axis=1)
    blank_probs = probs[:, blank]
    phone_probs = probs[:, phone_ids]
    phone_mass = np.sum(phone_probs, axis=1)
    phone_top1 = np.max(phone_probs, axis=1)
    phone_margin = _top1_margin(phone_probs)

    full_entropy = _normalized_entropy(probs, axis_size=vocab_size)
    conditional = phone_probs / np.maximum(phone_mass[:, None], np.finfo(np.float64).tiny)
    phone_entropy = _normalized_entropy(conditional, axis_size=len(phone_ids))

    full_logit_margin = _top1_margin(values)
    phone_logits = values[:, phone_ids]
    phone_logit_margin = _top1_margin(phone_logits)
    blank_minus_phone = values[:, blank] - np.max(phone_logits, axis=1)

    return CtcPosteriorDiagnostics(
        available=True,
        schema=SCHEMA,
        frame_count=int(frames),
        vocabulary_size=int(vocab_size),
        phone_token_count=len(phone_ids),
        blank_id=blank,
        top1_posterior_mean=float(np.mean(top1)),
        top1_posterior_p95=float(np.quantile(top1, 0.95)),
        top1_posterior_max=float(np.max(top1)),
        blank_top1_fraction=float(np.mean(top1_ids == blank)),
        blank_posterior_mean=float(np.mean(blank_probs)),
        blank_posterior_p95=float(np.quantile(blank_probs, 0.95)),
        phone_mass_mean=float(np.mean(phone_mass)),
        phone_mass_p05=float(np.quantile(phone_mass, 0.05)),
        phone_top1_posterior_mean=float(np.mean(phone_top1)),
        phone_top1_posterior_p95=float(np.quantile(phone_top1, 0.95)),
        phone_top1_margin_mean=float(np.mean(phone_margin)),
        full_normalized_entropy_mean=float(np.mean(full_entropy)),
        full_normalized_entropy_p05=float(np.quantile(full_entropy, 0.05)),
        phone_conditional_normalized_entropy_mean=float(np.mean(phone_entropy)),
        phone_conditional_normalized_entropy_p05=float(np.quantile(phone_entropy, 0.05)),
        full_top1_logit_margin_mean=float(np.mean(full_logit_margin)),
        full_top1_logit_margin_p05=float(np.quantile(full_logit_margin, 0.05)),
        phone_top1_logit_margin_mean=float(np.mean(phone_logit_margin)),
        phone_top1_logit_margin_p05=float(np.quantile(phone_logit_margin, 0.05)),
        blank_minus_phone_top_logit_margin_mean=float(np.mean(blank_minus_phone)),
        summary={
            "interpretation": "model_posterior_and_logit_competition_diagnostics_not_pronunciation_quality",
            "standard_ctc_peakiness_is_known_pronunciation_assessment_risk": True,
            "logit_margin_diagnostics_available": True,
            "logit_margin_is_pronunciation_error": False,
            "universal_threshold_defined": False,
            "heuristic_alert_thresholds_defined": False,
            "cross_model_diagnostic_comparison_allowed": True,
            "raw_gop_cross_model_averaging_allowed": False,
            "individual_frame_entropy_is_pronunciation_error": False,
            "product_score_changed": False,
        },
        warnings=[],
        product_calibrated=False,
        score_mapped=False,
    )