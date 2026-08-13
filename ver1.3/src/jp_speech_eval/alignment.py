from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import librosa
import numpy as np

from .audio_features import trim_silence
from .sentence_cache import SentenceCache, tts_reference


@dataclass(frozen=True)
class AlignmentResult:
    """Alignment boundaries plus provenance required for evidence consumers."""

    boundaries: List[Tuple[float, float]]
    method: str
    available: bool
    confidence: float
    used_equal_fallback: bool
    failure_reason: str = ""
    normalized_dtw_cost: Optional[float] = None
    path_length: Optional[int] = None
    path_coverage: Optional[float] = None
    path_slope_cv: Optional[float] = None
    feature_kind: str = "mfcc"
    band_rad: Optional[float] = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["boundaries"] = [[round(float(s), 6), round(float(e), 6)] for s, e in self.boundaries]
        for key in ("confidence", "normalized_dtw_cost", "path_coverage", "path_slope_cv", "band_rad"):
            if payload[key] is not None:
                payload[key] = round(float(payload[key]), 6)
        return payload


def estimate_mora_boundaries_equal(duration: float, mora_count: int) -> List[Tuple[float, float]]:
    n = max(int(mora_count), 1)
    step = max(0.0, float(duration)) / n
    return [(i * step, (i + 1) * step) for i in range(n)]


def _equal_result(
    duration: float,
    mora_count: int,
    reason: str,
    *,
    method: str,
    feature_kind: str = "mfcc",
    band_rad: Optional[float] = None,
) -> AlignmentResult:
    return AlignmentResult(
        boundaries=estimate_mora_boundaries_equal(duration, mora_count),
        method=method,
        available=False,
        confidence=0.0,
        used_equal_fallback=True,
        failure_reason=reason,
        feature_kind=feature_kind,
        band_rad=band_rad,
    )


def _normalize_feature(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    return ((feature - feature.mean(axis=1, keepdims=True)) / (feature.std(axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def _user_mfcc(user_y: np.ndarray, sr: int, hop: int = 160, n_fft: int = 512) -> np.ndarray:
    return _normalize_feature(librosa.feature.mfcc(y=user_y.astype(float), sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop))


def _feature_pair(cache: SentenceCache, user_y: np.ndarray, sr: int, hop: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    kind = str(kind).lower()
    if kind == "mfcc":
        return np.asarray(cache.ref_mfcc, dtype=np.float32), _user_mfcc(user_y, sr, hop=hop)
    if kind == "mfcc_delta":
        ref = np.asarray(cache.ref_mfcc, dtype=np.float32)
        usr = _user_mfcc(user_y, sr, hop=hop)
        if min(ref.shape[1], usr.shape[1]) < 5:
            raise ValueError("insufficient_frames_for_mfcc_delta")
        return _normalize_feature(np.vstack((ref, librosa.feature.delta(ref), librosa.feature.delta(ref, order=2)))), _normalize_feature(np.vstack((usr, librosa.feature.delta(usr), librosa.feature.delta(usr, order=2))))
    if kind == "logmel":
        def mel(y: np.ndarray) -> np.ndarray:
            power = librosa.feature.melspectrogram(y=y.astype(float), sr=sr, n_mels=40, n_fft=512, hop_length=hop, power=2.0)
            return _normalize_feature(librosa.power_to_db(power, ref=np.max))
        return mel(cache.ref_y), mel(user_y)
    raise ValueError(f"unknown alignment feature kind: {kind}")


def _map_ref_boundaries_to_user(
    ref_boundaries: List[Tuple[float, float]], wp: np.ndarray, ref_frame_count: int,
    user_duration: float, sr: int, hop: int,
) -> List[Tuple[float, float]]:
    wp = np.asarray(wp)
    if wp.size == 0:
        return []
    wp = wp[np.argsort(wp[:, 0])]
    ref_to_usr = {}
    for r in np.unique(wp[:, 0]):
        values = wp[wp[:, 0] == r, 1]
        ref_to_usr[int(r)] = int(np.median(values))
    keys = np.array(sorted(ref_to_usr))
    if not keys.size:
        return []
    mapped: List[Tuple[float, float]] = []
    for rs, re in ref_boundaries:
        rf_s = int(np.clip(round(rs * sr / hop), 0, ref_frame_count - 1))
        rf_e = int(np.clip(round(re * sr / hop), 0, ref_frame_count - 1))
        us_idx = ref_to_usr[int(keys[np.argmin(np.abs(keys - rf_s))])]
        ue_idx = ref_to_usr[int(keys[np.argmin(np.abs(keys - rf_e))])]
        start = float(np.clip(us_idx * hop / sr, 0.0, user_duration))
        end = float(np.clip(ue_idx * hop / sr, 0.0, user_duration))
        if end < start:
            start, end = end, start
        if end - start < .02:
            end = min(user_duration, start + .02)
        mapped.append((start, end))
    fixed: List[Tuple[float, float]] = []
    previous_end = 0.0
    for start, end in mapped:
        start, end = max(start, previous_end), min(user_duration, max(end, start + .02))
        fixed.append((start, end)); previous_end = end
    return fixed


def _path_metrics(wp: np.ndarray, ref_frames: int, user_frames: int) -> tuple[float, float]:
    path = np.asarray(wp)
    if not len(path):
        return 0.0, float("inf")
    coverage = min(len(np.unique(path[:, 0])) / max(ref_frames, 1), len(np.unique(path[:, 1])) / max(user_frames, 1))
    ordered = path[np.argsort(path[:, 0])]
    per_ref = np.array([np.median(ordered[ordered[:, 0] == r, 1]) for r in np.unique(ordered[:, 0])], dtype=float)
    steps = np.diff(per_ref)
    positive = steps[steps > 0]
    slope_cv = float(np.std(positive) / max(np.mean(positive), 1e-8)) if len(positive) > 1 else 0.0
    return float(coverage), slope_cv


def _boundaries_plausible(boundaries: List[Tuple[float, float]]) -> bool:
    """Reject mapped boundaries that are usable arrays but unusable evidence.

    DTW can return a path for heavily channel-altered or wrong-target speech.
    Such a path must not silently become local-mora evidence merely because it
    can be mapped into monotonically ordered pairs.  This is deliberately the
    same conservative health rule formerly applied only in the evaluator, so
    a failed first pass can now trigger the documented wider-band retry.
    """
    if not boundaries:
        return False
    durations = np.asarray([max(0.0, end - start) for start, end in boundaries], dtype=float)
    average = float(np.mean(durations))
    return not (
        float(np.std(durations) / max(average, 1e-8)) > .75
        or float(np.min(durations)) < .07
        or float(durations[0]) < .10
        or float(np.max(durations)) > max(.55, 3.2 * average)
    )


def _cached_dtw_attempt(cache: SentenceCache, user_y: np.ndarray, sr: int, *, band_rad: float, feature_kind: str) -> AlignmentResult:
    mora_count = cache.mora_count
    user_y, _ = trim_silence(user_y, top_db=30.0)
    duration = len(user_y) / max(sr, 1)
    method = f"cached_dtw_{feature_kind}_band_{band_rad:g}"
    if mora_count <= 0:
        return AlignmentResult([], method, False, 0.0, False, "invalid_mora_count", feature_kind=feature_kind, band_rad=band_rad)
    if len(user_y) < sr * .15:
        return _equal_result(duration, mora_count, "user_audio_too_short", method=method, feature_kind=feature_kind, band_rad=band_rad)
    try:
        ref, usr = _feature_pair(cache, user_y, sr, 160, feature_kind)
    except Exception as exc:
        return _equal_result(duration, mora_count, f"feature_extraction_failed:{type(exc).__name__}", method=method, feature_kind=feature_kind, band_rad=band_rad)
    if ref.ndim != 2 or usr.ndim != 2 or min(ref.shape[1], usr.shape[1]) < 2:
        return _equal_result(duration, mora_count, "insufficient_feature_frames", method=method, feature_kind=feature_kind, band_rad=band_rad)
    try:
        costs, path = librosa.sequence.dtw(X=ref, Y=usr, metric="euclidean", global_constraints=True, band_rad=band_rad)
    except TypeError:
        try:
            costs, path = librosa.sequence.dtw(X=ref, Y=usr, metric="euclidean")
        except Exception as exc:
            return _equal_result(duration, mora_count, f"dtw_exception:{type(exc).__name__}", method=method, feature_kind=feature_kind, band_rad=band_rad)
    except Exception as exc:
        return _equal_result(duration, mora_count, f"dtw_exception:{type(exc).__name__}", method=method, feature_kind=feature_kind, band_rad=band_rad)
    mapped = _map_ref_boundaries_to_user(cache.meta.ref_mora_boundaries, path, ref.shape[1], duration, sr, 160)
    cost = float(costs[-1, -1] / max(len(path), 1))
    coverage, slope_cv = _path_metrics(path, ref.shape[1], usr.shape[1])
    if len(mapped) != mora_count or not mapped or mapped[-1][1] <= 0:
        return _equal_result(duration, mora_count, "boundary_mapping_failure", method=method, feature_kind=feature_kind, band_rad=band_rad)
    if not _boundaries_plausible(mapped):
        return _equal_result(duration, mora_count, "boundary_health_unstable", method=method, feature_kind=feature_kind, band_rad=band_rad)
    # Confidence is deliberately a weak provenance/reliability scalar; score
    # policy continues to use separate evidence availability and health checks.
    confidence = float(np.clip(.95 * coverage * (1.0 / (1.0 + max(cost, 0.0) / 6.0)), .05, .95))
    return AlignmentResult(mapped, method, True, confidence, False, "", cost, len(path), coverage, slope_cv, feature_kind, band_rad)


def estimate_mora_boundaries_cached_dtw_result(cache: SentenceCache, user_y: np.ndarray, sr: int, band_rad: float = .25, feature_kind: str = "mfcc", second_pass_wider_band: bool = False) -> AlignmentResult:
    result = _cached_dtw_attempt(cache, user_y, sr, band_rad=band_rad, feature_kind=feature_kind)
    if result.available or not second_pass_wider_band:
        return result
    wide = _cached_dtw_attempt(cache, user_y, sr, band_rad=max(.45, band_rad), feature_kind=feature_kind)
    if wide.available:
        return AlignmentResult(**{**asdict(wide), "method": f"{wide.method}_second_pass"})
    return AlignmentResult(**{**asdict(result), "failure_reason": f"{result.failure_reason};second_pass:{wide.failure_reason}"})


def estimate_mora_boundaries_cached_dtw(cache: SentenceCache, user_y: np.ndarray, sr: int, band_rad: float = .25) -> List[Tuple[float, float]]:
    """Compatibility wrapper; new callers should consume AlignmentResult."""
    return estimate_mora_boundaries_cached_dtw_result(cache, user_y, sr, band_rad=band_rad).boundaries


def estimate_mora_boundaries_dtw_result(text: str, user_y: np.ndarray, sr: int, mora_count: int) -> AlignmentResult:
    duration = len(user_y) / max(sr, 1)
    if mora_count <= 0:
        return AlignmentResult([], "legacy_dtw", False, 0.0, False, "invalid_mora_count")
    try:
        ref_y = tts_reference(text, sr=sr)
        ref_y, _ = trim_silence(ref_y, top_db=30.0); user_y, _ = trim_silence(user_y, top_db=30.0)
        if len(ref_y) < sr * .2 or len(user_y) < sr * .2:
            return _equal_result(len(user_y) / sr, mora_count, "legacy_audio_too_short", method="legacy_dtw_mfcc")
        ref, usr = _user_mfcc(ref_y, sr), _user_mfcc(user_y, sr)
        costs, path = librosa.sequence.dtw(X=ref, Y=usr, metric="euclidean", global_constraints=True, band_rad=.25)
        mapped = _map_ref_boundaries_to_user(estimate_mora_boundaries_equal(len(ref_y) / sr, mora_count), path, ref.shape[1], len(user_y) / sr, sr, 160)
        if len(mapped) != mora_count or not mapped:
            return _equal_result(len(user_y) / sr, mora_count, "legacy_boundary_mapping_failure", method="legacy_dtw_mfcc")
        coverage, slope_cv = _path_metrics(path, ref.shape[1], usr.shape[1])
        return AlignmentResult(mapped, "legacy_dtw_mfcc", True, .7 * coverage, False, "", float(costs[-1, -1] / max(len(path), 1)), len(path), coverage, slope_cv, "mfcc", .25)
    except Exception as exc:
        return _equal_result(duration, mora_count, f"legacy_dtw_exception:{type(exc).__name__}", method="legacy_dtw_mfcc")


def estimate_mora_boundaries_dtw(text: str, user_y: np.ndarray, sr: int, mora_count: int) -> List[Tuple[float, float]]:
    return estimate_mora_boundaries_dtw_result(text, user_y, sr, mora_count).boundaries


def estimate_mora_boundaries(text: str, y_trim: np.ndarray, sr: int, mora_count: int, mode: str = "dtw", cache: Optional[SentenceCache] = None, *, return_result: bool = False, feature_kind: str = "mfcc", band_rad: float = .25, second_pass_wider_band: bool = False):
    """Compatibility wrapper that can return explicit alignment provenance."""
    duration = len(y_trim) / max(sr, 1)
    if mode == "equal":
        result = _equal_result(duration, mora_count, "explicit_equal_mode", method="equal_requested")
    elif mode == "cached_dtw":
        result = estimate_mora_boundaries_cached_dtw_result(cache, y_trim, sr, band_rad, feature_kind, second_pass_wider_band) if cache is not None else estimate_mora_boundaries_dtw_result(text, y_trim, sr, mora_count)
    elif mode == "dtw":
        result = estimate_mora_boundaries_dtw_result(text, y_trim, sr, mora_count)
    else:
        raise ValueError(f"Unknown alignment mode: {mode}")
    return result if return_result else result.boundaries
