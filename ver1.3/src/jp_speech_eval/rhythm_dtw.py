"""Research-only DTW warp-path rhythm evidence.

The implementation follows the metric definitions in McIntosh et al. (2026),
*Self-supervised Speech Comparison for L2 Phone, Rhythm, and Intonation
Scoring* (arXiv:2607.13721): tempo irregularity is derived from local warp-path
angles after mapping each template frame to its mean learner-frame position and
smoothing that trace over five frames. Interval distortion is provided as a
pure metric over externally supplied vowel/consonant/silence labels; this module
deliberately does not pretend that the repository already has a validated
Japanese interval classifier.

These are distance-like shadow metrics. Lower values mean more locally uniform
relative timing. They have no /100 mapping and no user-facing threshold.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import numpy as np


RHYTHM_DTW_SCHEMA = "wavlm_dtw_rhythm_shadow_v1"


def _path_array(path: Any) -> np.ndarray:
    value = np.asarray(path, dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != 2 or len(value) < 2:
        raise ValueError("DTW path must have shape (steps, 2) with at least two steps")
    if np.any(value < 0):
        raise ValueError("DTW path indices must be non-negative")
    # Chronological order is convenient but not required for grouping. Keep a
    # deterministic order in case a caller passes a backtracked path.
    order = np.lexsort((value[:, 1], value[:, 0]))
    return value[order]


def template_to_learner_trace(path: Any, reference_frame_count: int) -> np.ndarray:
    """Map every template frame to the mean learner frame aligned to it."""

    path_arr = _path_array(path)
    count = int(reference_frame_count)
    if count <= 0:
        raise ValueError("reference_frame_count must be positive")
    trace = np.full(count, np.nan, dtype=float)
    for ref_index in range(count):
        matched = path_arr[path_arr[:, 0] == ref_index, 1]
        if matched.size:
            trace[ref_index] = float(np.mean(matched))
    finite = np.isfinite(trace)
    if not np.any(finite):
        raise ValueError("DTW path does not map any template frame")
    if not np.all(finite):
        known = np.flatnonzero(finite)
        missing = np.flatnonzero(~finite)
        trace[missing] = np.interp(missing, known, trace[known])
    return trace


def _moving_average(values: np.ndarray, window_frames: int) -> np.ndarray:
    window = int(window_frames)
    if window <= 1:
        return values.astype(float, copy=True)
    if window % 2 == 0:
        raise ValueError("moving-average window must be odd")
    if len(values) < window:
        return values.astype(float, copy=True)
    radius = window // 2
    padded = np.pad(values.astype(float), (radius, radius), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=float)
    return np.convolve(padded, kernel, mode="valid")


def tempo_irregularity_from_dtw_path(
    path: Any,
    *,
    reference_frame_count: int,
    smoothing_frames: int = 5,
) -> Dict[str, Any]:
    """Compute McIntosh-style local tempo irregularity from a DTW path.

    A perfectly constant local pacing ratio has irregularity near zero even if
    the learner is globally faster/slower. This is intentional: global duration
    mismatch is a separate rhythm/tempo feature.
    """

    trace = template_to_learner_trace(path, reference_frame_count)
    smoothed = _moving_average(trace, smoothing_frames)
    if len(smoothed) < 3:
        raise ValueError("at least three mapped template frames are required")

    # Central difference over a three-frame window. Template-frame x spacing is
    # two frames, so divide by 2. DTW monotonicity should keep slopes >= 0; tiny
    # numerical negatives are clipped before converting to a warp angle.
    slopes = (smoothed[2:] - smoothed[:-2]) / 2.0
    slopes = np.maximum(slopes, 0.0)
    angles = np.arctan(slopes)
    angle_mean = float(np.mean(angles))
    deviations = np.abs(angles - angle_mean)
    irregularity = float(np.mean(deviations))

    return {
        "schema": RHYTHM_DTW_SCHEMA,
        "metric": "tempo_irregularity",
        "tempo_irregularity_rad": irregularity,
        "tempo_irregularity_deg": float(np.degrees(irregularity)),
        "mean_warp_angle_rad": angle_mean,
        "warp_angle_std_rad": float(np.std(angles)),
        "angle_count": int(len(angles)),
        "smoothing_frames": int(smoothing_frames),
        "interpretation": "lower_is_more_locally_uniform_relative_tempo_shadow_only",
        "score_mapped": False,
        "product_calibrated": False,
        "user_facing": False,
    }


def _contiguous_intervals(labels: Sequence[str]) -> list[tuple[int, int, str]]:
    normalized = [str(item).strip().lower() for item in labels]
    allowed = {"vowel", "consonant", "silence"}
    if any(item not in allowed for item in normalized):
        raise ValueError("interval labels must be vowel, consonant, or silence")
    intervals: list[tuple[int, int, str]] = []
    start = 0
    for index in range(1, len(normalized) + 1):
        if index == len(normalized) or normalized[index] != normalized[start]:
            label = normalized[start]
            if label != "silence":
                intervals.append((start, index, label))
            start = index
    return intervals


def interval_distortion_from_dtw_path(
    path: Any,
    *,
    reference_interval_labels: Sequence[str],
) -> Dict[str, Any]:
    """Compute interval-duration distortion for pre-classified template frames.

    The caller must provide validated frame labels. A future Japanese interval
    classifier can feed this function, but its errors must be validated
    separately before this metric is promoted.
    """

    if not reference_interval_labels:
        raise ValueError("reference_interval_labels must not be empty")
    path_arr = _path_array(path)
    intervals = _contiguous_intervals(reference_interval_labels)
    log_ratios: list[float] = []
    kinds: list[str] = []
    for start, end, label in intervals:
        matched = path_arr[(path_arr[:, 0] >= start) & (path_arr[:, 0] < end), 1]
        if not matched.size:
            continue
        native_frames = max(end - start, 1)
        learner_frames = max(int(np.max(matched) - np.min(matched) + 1), 1)
        log_ratios.append(float(np.log(learner_frames / native_frames)))
        kinds.append(label)
    if not log_ratios:
        raise ValueError("no labeled reference intervals could be mapped through the DTW path")
    values = np.asarray(log_ratios, dtype=float)
    mean_ratio = float(np.mean(values))
    distortion = float(np.mean(np.abs(values - mean_ratio)))
    return {
        "schema": RHYTHM_DTW_SCHEMA,
        "metric": "interval_distortion",
        "interval_distortion_log_ratio_mad": distortion,
        "mean_log_duration_ratio": mean_ratio,
        "interval_count": int(len(values)),
        "vocalic_interval_count": int(sum(kind == "vowel" for kind in kinds)),
        "consonantal_interval_count": int(sum(kind == "consonant" for kind in kinds)),
        "interpretation": "lower_is_more_locally_uniform_interval_mapping_shadow_only",
        "requires_validated_interval_classifier": True,
        "score_mapped": False,
        "product_calibrated": False,
        "user_facing": False,
    }
