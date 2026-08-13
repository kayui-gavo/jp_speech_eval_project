"""Lazy self-supervised speech features for audit-only pronunciation shadows.

Nothing in this module is loaded by the product score path unless the explicit
SSL shadow flag is enabled.  Distances are research measurements, not /100
pronunciation scores.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


DEFAULT_SSL_MODEL = "microsoft/wavlm-large"


def normalize_ssl_frames(features: np.ndarray) -> np.ndarray:
    frames = np.asarray(features, dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError("SSL features must have shape (frames, dimensions)")
    norms = np.linalg.norm(frames, axis=1, keepdims=True)
    return frames / np.maximum(norms, 1e-8)


def cosine_dtw_distance(reference: np.ndarray, user: np.ndarray) -> Dict[str, float | int]:
    """Return normalized cumulative cosine DTW distance and path length."""
    ref = normalize_ssl_frames(reference)
    hyp = normalize_ssl_frames(user)
    if not len(ref) or not len(hyp):
        raise ValueError("SSL DTW requires non-empty frame sequences")
    previous = np.full(len(hyp) + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    path_lengths = np.zeros(len(hyp) + 1, dtype=np.int32)
    for ref_frame in ref:
        current = np.full(len(hyp) + 1, np.inf, dtype=np.float64)
        current_lengths = np.zeros(len(hyp) + 1, dtype=np.int32)
        costs = 1.0 - np.clip(hyp @ ref_frame, -1.0, 1.0)
        for j, cost in enumerate(costs, start=1):
            options = (previous[j], current[j - 1], previous[j - 1])
            choice = int(np.argmin(options))
            parent_length = (
                path_lengths[j]
                if choice == 0
                else current_lengths[j - 1]
                if choice == 1
                else path_lengths[j - 1]
            )
            current[j] = options[choice] + float(cost)
            current_lengths[j] = parent_length + 1
        previous, path_lengths = current, current_lengths
    path_length = int(path_lengths[-1])
    cumulative = float(previous[-1])
    return {
        "normalized_cumulative_distance": cumulative / max(path_length, 1),
        "cumulative_distance": cumulative,
        "path_length": path_length,
        "reference_frame_count": int(len(ref)),
        "user_frame_count": int(len(hyp)),
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
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.num_layers = 0
        self._torch = None

    def _load_model(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("SSL shadow requires optional torch and transformers") from exc
        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id, output_hidden_states=True)
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
        with self._torch.no_grad():
            outputs = self.model(
                inputs["input_values"].to(self.device),
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
