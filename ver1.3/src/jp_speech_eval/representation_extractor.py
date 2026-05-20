from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np

from .audio_features import load_audio
from .vad import trim_to_speech


MODEL_ALIASES: Dict[str, str] = {
    "wavlm_base_plus": "microsoft/wavlm-base-plus",
    "wav2vec2_xlsr": "facebook/wav2vec2-large-xlsr-53",
    "hubert_base": "facebook/hubert-base-ls960",
    "hubert_large_zh": "TencentGameMate/chinese-hubert-large",
    "mhubert_147": "utter-project/mHuBERT-147",
}


@dataclass(frozen=True)
class Representation:
    """Frame-level representation extracted from one audio file."""

    values: np.ndarray
    model_name: str
    resolved_model_name: str
    layer_id: int
    sample_rate: int
    frame_hop_sec: Optional[float]
    backend: str


def resolve_model_name(model_name: str) -> str:
    """Resolve a short alias to a Hugging Face model id when available."""

    return MODEL_ALIASES.get(model_name, model_name)


def _normalize_mfcc(mfcc: np.ndarray) -> np.ndarray:
    return ((mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def _extract_mfcc(audio_path: str | Path, layer_id: int = 0) -> Representation:
    if int(layer_id) != 0:
        raise ValueError("mfcc only exposes layer 0")
    audio = load_audio(str(audio_path), sr=16000)
    y_speech, _ = trim_to_speech(audio.y, audio.sr)
    hop = 160
    mfcc = librosa.feature.mfcc(y=y_speech.astype(float), sr=audio.sr, n_mfcc=13, n_fft=512, hop_length=hop)
    return Representation(
        values=_normalize_mfcc(mfcc).T,
        model_name="mfcc",
        resolved_model_name="mfcc",
        layer_id=0,
        sample_rate=audio.sr,
        frame_hop_sec=hop / audio.sr,
        backend="librosa",
    )


class RepresentationExtractor:
    """Unified wrapper for light acoustic baselines and HF speech encoders."""

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self._hf_models: Dict[str, object] = {}
        self._hf_processors: Dict[str, object] = {}

    def available_layers(self, model_name: str) -> List[int]:
        """Return extractable layer ids for a model."""

        if model_name == "mfcc":
            return [0]
        model = self._load_hf_model(model_name)
        n = int(getattr(model.config, "num_hidden_layers", 0))
        if n <= 0:
            raise ValueError(f"model does not expose transformer layers: {model_name}")
        # Hugging Face hidden_states include the frontend projection at index 0.
        return list(range(n + 1))

    def extract(self, audio_path: str | Path, model_name: str, layer_id: int) -> Representation:
        """Extract one frame-level representation matrix `[T, D]`."""

        if model_name == "mfcc":
            return _extract_mfcc(audio_path, layer_id=layer_id)

        resolved = resolve_model_name(model_name)
        model = self._load_hf_model(model_name)
        processor = self._hf_processors[resolved]
        audio = load_audio(str(audio_path), sr=16000)
        y_speech, _ = trim_to_speech(audio.y, audio.sr)

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for Hugging Face speech representations") from exc

        inputs = processor(
            np.asarray(y_speech, dtype=np.float32),
            sampling_rate=audio.sr,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self._device())
        with torch.no_grad():
            outputs = model(input_values, output_hidden_states=True)

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError(f"model did not return hidden_states: {resolved}")
        if layer_id < 0 or layer_id >= len(hidden_states):
            raise ValueError(f"layer {layer_id} out of range for {resolved}; valid 0..{len(hidden_states) - 1}")

        values = hidden_states[layer_id].squeeze(0).detach().cpu().numpy().astype(np.float32)
        hop = None
        if values.shape[0] > 0:
            hop = float(len(y_speech) / audio.sr / values.shape[0])
        return Representation(
            values=values,
            model_name=model_name,
            resolved_model_name=resolved,
            layer_id=int(layer_id),
            sample_rate=audio.sr,
            frame_hop_sec=hop,
            backend="huggingface_transformers",
        )

    def _device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _load_hf_model(self, model_name: str):
        resolved = resolve_model_name(model_name)
        if resolved in self._hf_models:
            return self._hf_models[resolved]
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("transformers is required for Hugging Face speech representations") from exc

        processor = AutoProcessor.from_pretrained(resolved)
        model = AutoModel.from_pretrained(resolved)
        model.to(self._device())
        model.eval()
        self._hf_models[resolved] = model
        self._hf_processors[resolved] = processor
        return model


def extract_representations(audio_path: str | Path, model_name: str, layer_id: int) -> np.ndarray:
    """Convenience wrapper matching the public MVP interface."""

    return RepresentationExtractor().extract(audio_path, model_name, layer_id).values


def parse_layer_spec(spec: str, extractor: RepresentationExtractor, model_name: str) -> List[int]:
    """Parse `all`, a single integer, or a comma-separated layer list."""

    spec = spec.strip().lower()
    if spec == "all":
        return extractor.available_layers(model_name)
    layers = [int(item.strip()) for item in spec.split(",") if item.strip()]
    if not layers:
        raise ValueError("layer spec is empty")
    return layers
