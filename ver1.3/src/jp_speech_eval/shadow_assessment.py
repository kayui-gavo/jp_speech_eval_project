"""Exception-isolated orchestration for audit-only assessment candidates."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


_SSL_EXTRACTOR_CACHE: Dict[str, Any] = {}


def _failure(backend: str, exc: Exception, elapsed: float) -> Dict[str, Any]:
    return {
        "available": False,
        "backend": backend,
        "error": f"{type(exc).__name__}: {exc}",
        "latency_ms": round(elapsed * 1000.0, 3),
        "user_facing": False,
    }


def _audio(path: str, sample_rate: int):
    from .audio_features import load_audio

    loaded = load_audio(path, sr=sample_rate)
    return loaded.y, loaded.sr


def run_assessment_shadows(
    result: Dict[str, Any],
    *,
    user_audio_path: str,
    sample_rate: int = 16000,
    enable_ssl_shadow: bool = False,
    ssl_model_id: str = "microsoft/wavlm-large",
    ssl_layer: int = 12,
    ssl_timeout_sec: float = 30.0,
    enable_special_mora_v2_shadow: bool = False,
    enable_phrase_intonation_shadow: bool = False,
    enable_accent_nucleus_shadow: bool = False,
    ssl_extractor: Optional[Any] = None,
) -> Dict[str, Any]:
    details = result.setdefault("details", {})
    shadow = details.setdefault("shadow", {})
    waveform = None
    sr = sample_rate

    if enable_ssl_shadow:
        started = time.perf_counter()
        try:
            from .ssl_features import SSLFeatureExtractor, cosine_dtw_distance

            waveform, sr = _audio(user_audio_path, sample_rate)
            cache_prefix = result.get("cache_prefix")
            if not cache_prefix:
                fixed_debug = details.get("fixed_reference_debug")
                if isinstance(fixed_debug, Mapping):
                    cache_prefix = fixed_debug.get("cache_prefix")
            reference_path = Path(f"{cache_prefix}.ref.wav") if cache_prefix else None
            if reference_path is None or not reference_path.exists():
                raise FileNotFoundError("fixed-reference audio is unavailable")
            reference, reference_sr = _audio(str(reference_path), sample_rate)
            extractor = ssl_extractor
            if extractor is None:
                extractor = _SSL_EXTRACTOR_CACHE.get(ssl_model_id)
                if extractor is None:
                    extractor = SSLFeatureExtractor(model_id=ssl_model_id)
                    _SSL_EXTRACTOR_CACHE[ssl_model_id] = extractor
            user_features = extractor.extract_layer(waveform, ssl_layer, sr)
            reference_features = extractor.extract_layer(reference, ssl_layer, reference_sr)
            distance = cosine_dtw_distance(reference_features, user_features)
            reference_distance = {
                "reference_id": details.get("reference_id"),
                "distance": distance["normalized_cumulative_distance"],
            }
            elapsed = time.perf_counter() - started
            if elapsed > ssl_timeout_sec:
                raise TimeoutError(f"SSL shadow exceeded {ssl_timeout_sec:.1f}s budget")
            shadow["ssl_pronunciation"] = {
                "available": True,
                "backend": "huggingface_ssl_cosine_dtw",
                "model_id": ssl_model_id,
                "layer": ssl_layer,
                "reference_id": details.get("reference_id"),
                **distance,
                "dtw_distance": distance["normalized_cumulative_distance"],
                "frame_count_reference": distance["reference_frame_count"],
                "frame_count_user": distance["user_frame_count"],
                "reference_distances": [reference_distance],
                "aggregate_strategy": "median",
                "interpretation": "shadow_not_user_score",
                "timeout_budget_sec": ssl_timeout_sec,
                "latency_ms": round(elapsed * 1000.0, 3),
                "user_facing": False,
            }
        except Exception as exc:
            shadow["ssl_pronunciation"] = _failure(
                "huggingface_ssl_cosine_dtw", exc, time.perf_counter() - started
            )

    if enable_special_mora_v2_shadow:
        started = time.perf_counter()
        try:
            from .special_mora_shadow_v2 import compute_special_mora_v2_shadow

            if waveform is None:
                waveform, sr = _audio(user_audio_path, sample_rate)
            payload = compute_special_mora_v2_shadow(result, waveform, sr)
            payload["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            shadow["special_mora_v2"] = payload
        except Exception as exc:
            shadow["special_mora_v2"] = _failure("numpy_local_roi_v2", exc, time.perf_counter() - started)

    if enable_phrase_intonation_shadow:
        started = time.perf_counter()
        try:
            from .prosody_shadows import compute_phrase_intonation_shadow

            payload = compute_phrase_intonation_shadow(result)
            payload["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            shadow["phrase_intonation_v1"] = payload
        except Exception as exc:
            shadow["phrase_intonation_v1"] = _failure("mora_log_f0_semitone_v1", exc, time.perf_counter() - started)

    if enable_accent_nucleus_shadow:
        started = time.perf_counter()
        try:
            from .prosody_shadows import compute_accent_nucleus_shadow

            payload = compute_accent_nucleus_shadow(result)
            payload["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            shadow["accent_nucleus_v0"] = payload
        except Exception as exc:
            shadow["accent_nucleus_v0"] = _failure("mora_log_f0_drop_v0", exc, time.perf_counter() - started)

    return shadow
