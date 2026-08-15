"""Exception-isolated orchestration for audit-only assessment candidates."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np


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


def _single_cached_reference(result: Mapping[str, Any], details: Mapping[str, Any]) -> list[Dict[str, Any]]:
    cache_prefix = result.get("cache_prefix")
    if not cache_prefix:
        fixed_debug = details.get("fixed_reference_debug")
        if isinstance(fixed_debug, Mapping):
            cache_prefix = fixed_debug.get("cache_prefix")
    reference_path = Path(f"{cache_prefix}.ref.wav") if cache_prefix else None
    if reference_path is None or not reference_path.exists():
        raise FileNotFoundError("fixed-reference audio is unavailable")
    return [
        {
            "reference_id": str(details.get("reference_id") or reference_path.stem),
            "audio_path": str(reference_path),
            "speaker_id": "",
            "reference_kind": str(details.get("reference_source") or "legacy_single_reference"),
            "provenance": "legacy_cache_reference",
        }
    ]


def _ssl_reference_rows(
    result: Mapping[str, Any],
    details: Mapping[str, Any],
    *,
    panel_path: Optional[str],
) -> tuple[list[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Resolve explicit multi-native references or the legacy single reference."""

    if panel_path:
        from .ssl_reference_panel import load_ssl_reference_panel

        panel = load_ssl_reference_panel(panel_path, require_audio_exists=True)
        target_text = str(result.get("target_text") or "").strip()
        selected = panel.references_for_target(target_text, allow_tts_fallback=False)
        if selected:
            return [item.to_dict() for item in selected], {
                **panel.to_dict(),
                "selected_target_text": target_text,
                "selected_reference_count": len(selected),
                "human_only": True,
            }
        # An explicitly supplied panel should not silently pretend to cover a
        # target it does not contain. Falling back to the old cache would mix
        # reference generations and make research results hard to reproduce.
        raise ValueError(f"SSL reference panel has no human reference for target: {target_text!r}")
    return _single_cached_reference(result, details), None


def run_assessment_shadows(
    result: Dict[str, Any],
    *,
    user_audio_path: str,
    sample_rate: int = 16000,
    enable_ssl_shadow: bool = False,
    ssl_model_id: str = "microsoft/wavlm-large",
    ssl_layer: int = 12,
    ssl_timeout_sec: float = 30.0,
    ssl_reference_panel_path: Optional[str] = None,
    ssl_reference_aggregation: str = "median",
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
            from .rhythm_dtw import tempo_irregularity_from_dtw_path
            from .ssl_features import (
                SSLFeatureExtractor,
                aggregate_reference_distances,
                cosine_dtw_alignment,
            )

            waveform, sr = _audio(user_audio_path, sample_rate)
            references, panel_meta = _ssl_reference_rows(
                result,
                details,
                panel_path=ssl_reference_panel_path,
            )
            extractor = ssl_extractor
            if extractor is None:
                extractor = _SSL_EXTRACTOR_CACHE.get(ssl_model_id)
                if extractor is None:
                    extractor = SSLFeatureExtractor(model_id=ssl_model_id)
                    _SSL_EXTRACTOR_CACHE[ssl_model_id] = extractor
            user_features = extractor.extract_layer(waveform, ssl_layer, sr)

            reference_distances: list[Dict[str, Any]] = []
            reference_rhythm: list[Dict[str, Any]] = []
            reference_frame_counts: list[int] = []
            path_lengths: list[int] = []
            cumulative_distances: list[float] = []
            for reference_row in references:
                if time.perf_counter() - started > ssl_timeout_sec:
                    raise TimeoutError(f"SSL shadow exceeded {ssl_timeout_sec:.1f}s budget")
                reference, reference_sr = _audio(str(reference_row["audio_path"]), sample_rate)
                reference_features = extractor.extract_layer(reference, ssl_layer, reference_sr)
                alignment = cosine_dtw_alignment(reference_features, user_features)
                path = alignment.pop("path")
                distance = float(alignment["normalized_cumulative_distance"])
                reference_distances.append(
                    {
                        "reference_id": reference_row["reference_id"],
                        "speaker_id": reference_row.get("speaker_id", ""),
                        "reference_kind": reference_row.get("reference_kind", ""),
                        "provenance": reference_row.get("provenance", ""),
                        "distance": distance,
                        "reference_frame_count": int(alignment["reference_frame_count"]),
                        "path_length": int(alignment["path_length"]),
                    }
                )
                reference_frame_counts.append(int(alignment["reference_frame_count"]))
                path_lengths.append(int(alignment["path_length"]))
                cumulative_distances.append(float(alignment["cumulative_distance"]))
                rhythm = tempo_irregularity_from_dtw_path(
                    path,
                    reference_frame_count=int(alignment["reference_frame_count"]),
                    smoothing_frames=5,
                )
                reference_rhythm.append(
                    {
                        "reference_id": reference_row["reference_id"],
                        "tempo_irregularity_rad": float(rhythm["tempo_irregularity_rad"]),
                        "global_frame_duration_ratio": (
                            float(alignment["user_frame_count"])
                            / max(float(alignment["reference_frame_count"]), 1.0)
                        ),
                        "angle_count": int(rhythm["angle_count"]),
                    }
                )

            aggregate_distance = aggregate_reference_distances(
                [item["distance"] for item in reference_distances],
                strategy=ssl_reference_aggregation,
            )
            aggregate_irregularity = aggregate_reference_distances(
                [item["tempo_irregularity_rad"] for item in reference_rhythm],
                strategy=ssl_reference_aggregation,
            )
            aggregate_duration_ratio = aggregate_reference_distances(
                [item["global_frame_duration_ratio"] for item in reference_rhythm],
                strategy=ssl_reference_aggregation,
            )
            elapsed = time.perf_counter() - started
            if elapsed > ssl_timeout_sec:
                raise TimeoutError(f"SSL shadow exceeded {ssl_timeout_sec:.1f}s budget")

            panel_id = panel_meta.get("panel_id") if panel_meta else None
            reference_identity = panel_id or reference_distances[0]["reference_id"]
            shadow["ssl_pronunciation"] = {
                "available": True,
                "backend": "huggingface_ssl_cosine_dtw",
                "model_id": ssl_model_id,
                "layer": ssl_layer,
                "reference_id": reference_identity,
                "reference_panel": panel_meta,
                "reference_count": len(reference_distances),
                "normalized_cumulative_distance": float(aggregate_distance),
                "dtw_distance": float(aggregate_distance),
                "path_length": int(round(float(np.median(path_lengths)))),
                "reference_frame_count": int(round(float(np.median(reference_frame_counts)))),
                "user_frame_count": int(len(user_features)),
                "cumulative_distance": float(np.median(cumulative_distances)),
                "frame_count_reference": int(round(float(np.median(reference_frame_counts)))),
                "frame_count_user": int(len(user_features)),
                "reference_distances": reference_distances,
                "aggregate_strategy": ssl_reference_aggregation,
                "interpretation": "multi_reference_shadow_not_user_score",
                "timeout_budget_sec": ssl_timeout_sec,
                "latency_ms": round(elapsed * 1000.0, 3),
                "score_mapped": False,
                "product_calibrated": False,
                "user_facing": False,
            }
            shadow["rhythm_dtw_v1"] = {
                "available": True,
                "backend": "wavlm_cosine_dtw_warp_path",
                "model_id": ssl_model_id,
                "layer": ssl_layer,
                "reference_id": reference_identity,
                "reference_panel": panel_meta,
                "reference_count": len(reference_rhythm),
                "aggregate_strategy": ssl_reference_aggregation,
                "tempo_irregularity_rad": float(aggregate_irregularity),
                "tempo_irregularity_deg": float(np.degrees(aggregate_irregularity)),
                "global_frame_duration_ratio": float(aggregate_duration_ratio),
                "reference_rhythm": reference_rhythm,
                "interpretation": "lower_is_more_locally_uniform_relative_tempo_shadow_only",
                "score_mapped": False,
                "product_calibrated": False,
                "user_facing": False,
                "latency_ms": round(elapsed * 1000.0, 3),
            }
        except Exception as exc:
            failure = _failure("huggingface_ssl_cosine_dtw", exc, time.perf_counter() - started)
            shadow["ssl_pronunciation"] = failure
            shadow.setdefault("rhythm_dtw_v1", {
                **failure,
                "backend": "wavlm_cosine_dtw_warp_path",
            })

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
