from __future__ import annotations

import numpy as np

from jp_speech_eval.rhythm_dtw import (
    interval_distortion_from_dtw_path,
    tempo_irregularity_from_dtw_path,
)
from jp_speech_eval.ssl_features import cosine_dtw_alignment, cosine_dtw_distance


def _path_from_trace(trace: list[int]) -> np.ndarray:
    return np.asarray([[index, value] for index, value in enumerate(trace)], dtype=np.int64)


def test_tempo_irregularity_is_near_zero_for_constant_local_pacing():
    diagonal = _path_from_trace(list(range(24)))
    stretched = _path_from_trace([2 * index for index in range(24)])
    first = tempo_irregularity_from_dtw_path(diagonal, reference_frame_count=24)
    second = tempo_irregularity_from_dtw_path(stretched, reference_frame_count=24)
    assert first["tempo_irregularity_rad"] < 1e-8
    assert second["tempo_irregularity_rad"] < 1e-8
    assert first["score_mapped"] is False


def test_tempo_irregularity_detects_local_rate_change_not_just_global_duration():
    trace = []
    for index in range(30):
        if index < 10:
            trace.append(index)
        elif index < 20:
            trace.append(10 + 2 * (index - 10))
        else:
            trace.append(30 + (index - 20))
    metric = tempo_irregularity_from_dtw_path(
        _path_from_trace(trace),
        reference_frame_count=30,
    )
    assert metric["tempo_irregularity_rad"] > 0.05
    assert metric["angle_count"] == 28


def test_interval_distortion_metric_accepts_external_interval_labels_only():
    # First interval maps close to 1:1; second is stretched. The metric measures
    # dispersion around the utterance mean rather than treating global tempo as
    # the rhythm error itself.
    path = _path_from_trace([0, 1, 2, 3, 4, 5, 7, 9, 11, 13])
    labels = ["vowel"] * 5 + ["consonant"] * 5
    metric = interval_distortion_from_dtw_path(path, reference_interval_labels=labels)
    assert metric["interval_count"] == 2
    assert metric["vocalic_interval_count"] == 1
    assert metric["consonantal_interval_count"] == 1
    assert metric["interval_distortion_log_ratio_mad"] > 0
    assert metric["requires_validated_interval_classifier"] is True


def test_cosine_dtw_distance_api_does_not_leak_path_but_alignment_can_reuse_it():
    reference = np.eye(6, dtype=np.float32)
    user = np.eye(6, dtype=np.float32)
    distance = cosine_dtw_distance(reference, user)
    alignment = cosine_dtw_alignment(reference, user)
    assert "path" not in distance
    assert alignment["path"].shape[1] == 2
    assert distance["normalized_cumulative_distance"] == alignment["normalized_cumulative_distance"]
