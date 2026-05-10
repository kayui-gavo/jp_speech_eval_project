from __future__ import annotations

from typing import List, Optional, Tuple

import librosa
import numpy as np

from .audio_features import trim_silence
from .sentence_cache import SentenceCache, tts_reference


def estimate_mora_boundaries_equal(duration: float, mora_count: int) -> List[Tuple[float, float]]:
    n = max(int(mora_count), 1)
    step = duration / n
    return [(i * step, (i + 1) * step) for i in range(n)]


def _normalize_mfcc(mfcc: np.ndarray) -> np.ndarray:
    return ((mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def _user_mfcc(user_y: np.ndarray, sr: int, hop: int = 160, n_fft: int = 512) -> np.ndarray:
    m = librosa.feature.mfcc(y=user_y.astype(float), sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
    return _normalize_mfcc(m)


def _map_ref_boundaries_to_user(
    ref_boundaries: List[Tuple[float, float]],
    wp: np.ndarray,
    ref_frame_count: int,
    user_duration: float,
    sr: int,
    hop: int,
) -> List[Tuple[float, float]]:
    """Map reference mora boundaries to user time using a DTW warping path."""
    wp = np.asarray(wp)
    if wp.size == 0:
        return []
    wp = wp[np.argsort(wp[:, 0])]

    ref_to_usr = {}
    for r in np.unique(wp[:, 0]):
        u_vals = wp[wp[:, 0] == r, 1]
        ref_to_usr[int(r)] = int(np.median(u_vals))

    keys = np.array(sorted(ref_to_usr.keys()))
    if keys.size == 0:
        return []

    user_boundaries: List[Tuple[float, float]] = []
    for rs, re in ref_boundaries:
        rf_s = int(round(rs * sr / hop))
        rf_e = int(round(re * sr / hop))
        rf_s = int(np.clip(rf_s, 0, ref_frame_count - 1))
        rf_e = int(np.clip(rf_e, 0, ref_frame_count - 1))

        us_idx = ref_to_usr[int(keys[np.argmin(np.abs(keys - rf_s))])]
        ue_idx = ref_to_usr[int(keys[np.argmin(np.abs(keys - rf_e))])]
        us = float(np.clip(us_idx * hop / sr, 0.0, user_duration))
        ue = float(np.clip(ue_idx * hop / sr, 0.0, user_duration))
        if ue < us:
            us, ue = ue, us
        if ue - us < 0.02:
            ue = min(user_duration, us + 0.02)
        user_boundaries.append((us, ue))

    fixed: List[Tuple[float, float]] = []
    prev_end = 0.0
    for s, e in user_boundaries:
        s = max(float(s), prev_end)
        e = max(float(e), s + 0.02)
        e = min(e, user_duration)
        fixed.append((s, e))
        prev_end = e
    return fixed


def estimate_mora_boundaries_cached_dtw(
    cache: SentenceCache,
    user_y: np.ndarray,
    sr: int,
    band_rad: float = 0.25,
) -> List[Tuple[float, float]]:
    """
    Fast-ish DTW alignment using precomputed reference MFCC from sentence cache.

    This removes the slowest step from V0.1/V0.2: generating TTS + reference MFCC
    every time the user speaks.
    """
    mora_count = cache.mora_count
    user_y, _ = trim_silence(user_y, top_db=30.0)
    user_duration = len(user_y) / sr
    if mora_count <= 0:
        return []
    if len(user_y) < sr * 0.15:
        return estimate_mora_boundaries_equal(user_duration, mora_count)

    hop = 160
    usr_mfcc = _user_mfcc(user_y, sr=sr, hop=hop)
    ref_mfcc = cache.ref_mfcc
    if ref_mfcc.ndim != 2 or usr_mfcc.ndim != 2 or ref_mfcc.shape[1] < 2 or usr_mfcc.shape[1] < 2:
        return estimate_mora_boundaries_equal(user_duration, mora_count)

    try:
        # global_constraints=True applies a Sakoe-Chiba band and can be much faster.
        _D, wp = librosa.sequence.dtw(
            X=ref_mfcc,
            Y=usr_mfcc,
            metric="euclidean",
            global_constraints=True,
            band_rad=band_rad,
        )
    except TypeError:
        # Older librosa: no global constraint args.
        try:
            _D, wp = librosa.sequence.dtw(X=ref_mfcc, Y=usr_mfcc, metric="euclidean")
        except Exception:
            return estimate_mora_boundaries_equal(user_duration, mora_count)
    except Exception:
        return estimate_mora_boundaries_equal(user_duration, mora_count)

    fixed = _map_ref_boundaries_to_user(
        ref_boundaries=cache.meta.ref_mora_boundaries,
        wp=wp,
        ref_frame_count=ref_mfcc.shape[1],
        user_duration=user_duration,
        sr=sr,
        hop=hop,
    )
    if len(fixed) != mora_count or not fixed or fixed[-1][1] <= 0:
        return estimate_mora_boundaries_equal(user_duration, mora_count)
    return fixed


def estimate_mora_boundaries_dtw(
    text: str,
    user_y: np.ndarray,
    sr: int,
    mora_count: int,
) -> List[Tuple[float, float]]:
    """
    Legacy DTW alignment. Kept for debugging when no cache is prepared.
    Prefer `estimate_mora_boundaries_cached_dtw` for product-like evaluation.
    """
    if mora_count <= 0:
        return []

    ref_y = tts_reference(text, sr=sr)
    user_y, _ = trim_silence(user_y, top_db=30.0)
    if len(ref_y) < sr * 0.2 or len(user_y) < sr * 0.2:
        return estimate_mora_boundaries_equal(len(user_y) / sr, mora_count)

    hop = 160
    ref_mfcc = _user_mfcc(ref_y, sr=sr, hop=hop)
    usr_mfcc = _user_mfcc(user_y, sr=sr, hop=hop)

    try:
        _D, wp = librosa.sequence.dtw(
            X=ref_mfcc,
            Y=usr_mfcc,
            metric="euclidean",
            global_constraints=True,
            band_rad=0.25,
        )
    except TypeError:
        try:
            _D, wp = librosa.sequence.dtw(X=ref_mfcc, Y=usr_mfcc, metric="euclidean")
        except Exception:
            return estimate_mora_boundaries_equal(len(user_y) / sr, mora_count)
    except Exception:
        return estimate_mora_boundaries_equal(len(user_y) / sr, mora_count)

    ref_duration = len(ref_y) / sr
    ref_boundaries = estimate_mora_boundaries_equal(ref_duration, mora_count)
    fixed = _map_ref_boundaries_to_user(
        ref_boundaries=ref_boundaries,
        wp=wp,
        ref_frame_count=ref_mfcc.shape[1],
        user_duration=len(user_y) / sr,
        sr=sr,
        hop=hop,
    )
    if len(fixed) != mora_count or not fixed:
        return estimate_mora_boundaries_equal(len(user_y) / sr, mora_count)
    return fixed


def estimate_mora_boundaries(
    text: str,
    y_trim: np.ndarray,
    sr: int,
    mora_count: int,
    mode: str = "dtw",
    cache: Optional[SentenceCache] = None,
) -> List[Tuple[float, float]]:
    duration = len(y_trim) / sr
    if mode == "equal":
        return estimate_mora_boundaries_equal(duration, mora_count)
    if mode == "cached_dtw":
        if cache is None:
            # No cache passed. Fall back to legacy DTW rather than crashing.
            return estimate_mora_boundaries_dtw(text, y_trim, sr, mora_count)
        return estimate_mora_boundaries_cached_dtw(cache, y_trim, sr)
    if mode == "dtw":
        return estimate_mora_boundaries_dtw(text, y_trim, sr, mora_count)
    raise ValueError(f"Unknown alignment mode: {mode}")
