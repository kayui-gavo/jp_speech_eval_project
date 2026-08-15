"""Model-level CTC posterior diagnostics for pronunciation research.

Standard CTC can be very peaky: most acoustic evidence collapses onto sparse
frames, which can make posterior-derived GOP unstable.  These diagnostics make
that model property explicit before any Japanese learner criterion mapping.

They are **not** pronunciation scores and they do not define a universal
"good" entropy or peakiness threshold.  The intended use is to compare pinned
phone-CTC backbones and to explain why a downstream GOP feature may be brittle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Sequence

import numpy as np


SCHEMA = "ctc_posterior_diagnostics_v1"


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


def compute_ctc_posterior_diagnostics(
    logits: np.ndarray,
    *,
    blank_id: int,
    phone_token_ids: Sequence[int],
) -> CtcPosteriorDiagnostics:
    """Summarize CTC peakiness/uncertainty without interpreting pronunciation."""
    probs = _softmax(logits)
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
    if phone_probs.shape[1] >= 2:
        top2 = np.partition(phone_probs, -2, axis=1)[:, -2:]
        phone_margin = np.max(top2, axis=1) - np.min(top2, axis=1)
    else:
        phone_margin = phone_top1.copy()

    full_entropy = _normalized_entropy(probs, axis_size=vocab_size)
    conditional = phone_probs / np.maximum(phone_mass[:, None], np.finfo(np.float64).tiny)
    phone_entropy = _normalized_entropy(conditional, axis_size=len(phone_ids))

    warnings: list[str] = []
    if float(np.mean(phone_mass)) < 0.25:
        warnings.append("low_mean_phone_probability_mass")
    if float(np.mean(top1)) > 0.95:
        warnings.append("very_peaky_mean_top1_posterior")

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
        summary={
            "interpretation": "model_posterior_peakiness_and_uncertainty_not_pronunciation_quality",
            "standard_ctc_peakiness_is_known_pronunciation_assessment_risk": True,
            "universal_threshold_defined": False,
            "cross_model_diagnostic_comparison_allowed": True,
            "raw_gop_cross_model_averaging_allowed": False,
            "individual_frame_entropy_is_pronunciation_error": False,
            "product_score_changed": False,
        },
        warnings=warnings,
        product_calibrated=False,
        score_mapped=False,
    )
