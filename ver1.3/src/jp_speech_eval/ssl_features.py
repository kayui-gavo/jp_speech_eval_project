"""Lazy self-supervised speech features for audit-only pronunciation shadows.

Nothing in this module is loaded by the product score path unless the explicit
SSL shadow flag is enabled. Distances are research measurements, not /100
pronunciation scores.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np


DEFAULT_SSL_MODEL = "microsoft/wavlm-large"


def normalize_ssl_frames(features: np.ndarray) -> np.ndarray:
    frames = np.asarray(features, dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError("SSL features must have shape (frames, dimensions)")
    norms = np.linalg.norm(frames, axis=1, keepdims=True)
    return frames / np.maximum(norms, 1e-8)


def cosine_dtw_alignment(reference: np.ndarray, user: np.ndarray) -> Dict[str, Any]:
    """Return cosine-DTW distance plus the optimal reference/user frame path.

    The path is research evidence. Callers should normally avoid serialising it
    into product telemetry; it is exposed so rhythm shadows can derive warp-path
    statistics without recomputing DTW.
    """

    ref = normalize_ssl_frames(reference)
    hyp = normalize_ssl_frames(user)
    if not len(ref) or not len(hyp):
        raise ValueError("SSL DTW requires non-empty frame sequences")
    import librosa

    costs = 1.0 - np.clip(ref @ hyp.T, -1.0, 1.0)
    accumulated, path = librosa.sequence.dtw(C=costs, backtrack=True)
    path = np.asarray(path, dtype=np.int64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("librosa DTW returned an invalid warping path")
    # librosa returns the backtracked path from the final cell toward the
    # origin. Chronological ordering is easier and less error-prone for rhythm
    # analysis, while cumulative cost is order-invariant.
    if len(path) >= 2 and tuple(path[0]) > tuple(path[-1]):
        path = path[::-1].copy()
    path_length = int(len(path))
    cumulative = float(accumulated[-1, -1])
    return {
        "normalized_cumulative_distance": cumulative / max(path_length, 1),
        "cumulative_distance": cumulative,
        "path_length": path_length,
        "reference_frame_count": int(len(ref)),
        "user_frame_count": int(len(hyp)),
        "path": path,
    }


def cosine_dtw_distance(reference: np.ndarray, user: np.ndarray) -> Dict[str, float | int]:
    """Return normalized cumulative cosine DTW distance and path length."""

    aligned = cosine_dtw_alignment(reference, user)
    return {
        key: value
        for key, value in aligned.items()
        if key != "path"
    }


def aggregate_reference_distances(distances: Iterable[float], strategy: str = "median", top_k: int = 2) -> float:
    """Aggregate already-computed reference distances for audit-only studies."""
    values = np.asarray([float(value) for value in distances if np.isfinite(value)], dtype=float)
    if not values.size:
        raise ValueError("at least one finite SSL distance is required")
    strategy = str(strategy).lower()
    if strategy == "mean":
        return float(np.mean(values))
    if strategy == "median":
        return float(np.median(values))
    if strategy == "trimmed_mean":
        ordered = np.sort(values)
        trim = int(len(ordered) * 0.2)
        return float(np.mean(ordered[trim:len(ordered) - trim] if len(ordered) - 2 * trim else ordered))
    if strategy == "nearest":
        return float(np.min(values))
    if strategy == "top_k_mean":
        return float(np.mean(np.sort(values)[:max(1, min(int(top_k), len(values)))]))
    raise ValueError(f"unknown reference aggregation strategy: {strategy}")


def robust_distance_normalize(distance: float, native_distances: Iterable[float], *, mad_floor: float = 1e-4) -> Dict[str, float]:
    """Normalise one SSL distance using native-reference median/MAD.

    The values remain distances; this only puts layer 12 and layer 24 on a
    comparable native-relative scale before a fusion experiment. It is not a
    learner-score mapping and must be fit on a development/native panel only.
    """
    values = np.asarray([float(item) for item in native_distances if np.isfinite(item)], dtype=float)
    if not values.size:
        raise ValueError("native distances are required for robust normalization")
    median = float(np.median(values))
    mad = max(float(np.median(np.abs(values - median))), float(mad_floor))
    return {
        "distance": float(distance),
        "native_median": median,
        "native_mad": mad,
        "normalized_distance": float((float(distance) - median) / mad),
    }


def fuse_layer_distances(
    distance_12: float,
    distance_24: float,
    *,
    native_12: Iterable[float],
    native_24: Iterable[float],
    alpha: float = 0.5,
) -> Dict[str, float]:
    """Fuse robust-normalised WavLM layer-12 and layer-24 distances."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    first = robust_distance_normalize(distance_12, native_12)
    second = robust_distance_normalize(distance_24, native_24)
    return {
        "alpha_layer12": alpha,
        "distance_layer12": float(distance_12),
        "distance_layer24": float(distance_24),
        "normalized_distance_layer12": first["normalized_distance"],
        "normalized_distance_layer24": second["normalized_distance"],
        "fused_normalized_distance": float(alpha * first["normalized_distance"] + (1.0 - alpha) * second["normalized_distance"]),
        "layer12_native_median": first["native_median"],
        "layer12_native_mad": first["native_mad"],
        "layer24_native_median": second["native_median"],
        "layer24_native_mad": second["native_mad"],
    }


class SSLFeatureExtractor:
    """Lazy Hugging Face WavLM/HubERT extractor.

    Instantiation is cheap. The optional torch/transformers dependencies and
    model checkpoint are loaded only on the first extraction call.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_SSL_MODEL,
        device: Optional[str] = None,
        local_files_only: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        # Shadows must not make an ordinary product/test invocation download a
        # multi-GB checkpoint. A benchmark may explicitly provide a local
        # snapshot; an unavailable snapshot is reported as a normal shadow
        # failure by its caller.
        self.local_files_only = bool(local_files_only)
        self.model = None
        self.processor = None
        self.num_layers = 0
        self._torch = None

    def _load_model(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise RuntimeError("SSL shadow requires optional torch and transformers") from exc
        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = AutoFeatureExtractor.from_pretrained(
                self.model_id, local_files_only=self.local_files_only,
            )
            self.model = AutoModel.from_pretrained(
                self.model_id, output_hidden_states=True, local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load SSL checkpoint {self.model_id}: {exc}") from exc
        self.model.to(self.device)
        self.model.eval()
        self.num_layers = int(getattr(self.model.config, "num_hidden_layers", 0))

    def extract_all_layers(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        normalize: bool = True,
    ) -> Dict[int, np.ndarray]:
        self._load_model()
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 1.0:
            waveform = waveform / peak
        inputs = self.processor(waveform, sampling_rate=sr, return_tensors="pt")
        model_inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self.model(
                **model_inputs,
                output_hidden_states=True,
            )
        layers: Dict[int, np.ndarray] = {}
        for index, hidden in enumerate(outputs.hidden_states):
            value = hidden.squeeze(0).detach().cpu().numpy()
            layers[index] = normalize_ssl_frames(value) if normalize else value
        return layers

    def extract_layer(self, audio: np.ndarray, layer_idx: int, sr: int = 16000) -> np.ndarray:
        layers = self.extract_all_layers(audio, sr=sr, normalize=True)
        if layer_idx not in layers:
            raise ValueError(f"SSL layer {layer_idx} is unavailable; got {sorted(layers)}")
        return layers[layer_idx]

    def get_frame_times(self, audio_length: int, sr: int = 16000) -> np.ndarray:
        hop_length = 320
        return np.arange((audio_length + sr // 2) // hop_length) * (hop_length / sr)


# Backward-compatible research name; it now supports WavLM as the valid default.
HuBERTFeatureExtractor = SSLFeatureExtractor


def extract_ssl_features(
    audio: np.ndarray,
    sr: int = 16000,
    model_id: str = DEFAULT_SSL_MODEL,
    layers: Optional[List[int]] = None,
) -> Dict[int, np.ndarray]:
    extracted = SSLFeatureExtractor(model_id=model_id).extract_all_layers(audio, sr)
    return extracted if layers is None else {key: extracted[key] for key in layers if key in extracted}
