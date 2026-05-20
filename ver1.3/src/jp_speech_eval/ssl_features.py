"""
HuBERT self-supervised learning features for Japanese speech evaluation.

This module extracts multilayer representations from HuBERT models,
following the Prosodic ABX framework from Sun & McIntosh (INTERSPEECH 2026).

Recommendation: HuBERT-Large(ZH) or HuBERT-Large-Japanese provides
best performance on cross-linguistic prosodic tasks.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from transformers import AutoModel, AutoProcessor, HubertModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "torch and transformers not available. Install with: "
        "pip install torch transformers"
    )


class HuBERTFeatureExtractor:
    """
    Extract HuBERT features from audio at multiple layers.
    
    Usage:
        extractor = HuBERTFeatureExtractor(model_id="facebook/hubert-large-ls60-japanese")
        features_by_layer = extractor.extract_all_layers(audio_array, sr=16000)
        # features_by_layer[layer_idx] -> shape (n_frames, hidden_dim)
    """
    
    def __init__(
        self,
        model_id: str = "facebook/hubert-large-ls60-japanese",
        device: Optional[str] = None,
    ):
        """
        Initialize HuBERT extractor.
        
        Args:
            model_id: Hugging Face model ID. Recommended:
                - "facebook/hubert-large-ls60-japanese" (Japanese-pretrained)
                - "facebook/hubert-large-ls60" (English)
                - "facebook/hubert-large-xlsr-53" (multilingual)
            device: "cuda", "cpu", or None (auto-detect)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch and transformers required. Install them first.")
        
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load HuBERT model and processor."""
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id, output_hidden_states=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Infer number of layers
            if hasattr(self.model, 'config'):
                self.num_layers = self.model.config.num_hidden_layers
            else:
                self.num_layers = 24  # Default for large models
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_id}: {e}")
    
    def extract_all_layers(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        normalize: bool = True,
    ) -> Dict[int, np.ndarray]:
        """
        Extract features from all HuBERT layers.
        
        Args:
            audio: Audio waveform, shape (n_samples,). Should be normalized to [-1, 1].
            sr: Sample rate.
            normalize: If True, normalize to log-scale with mean/std.
        
        Returns:
            {layer_idx: features} where features shape is (n_frames, hidden_dim)
        """
        if self.model is None:
            self._load_model()
        
        # Ensure audio is on correct device and dtype
        audio = np.asarray(audio, dtype=np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Process with HuBERT processor
        try:
            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
            input_values = inputs["input_values"].to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to process audio: {e}")
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)
        
        # Extract features from each layer
        all_layers = {}
        for layer_idx, hidden_state in enumerate(outputs.hidden_states):
            # hidden_state: (batch=1, frames, hidden_dim)
            feat = hidden_state.squeeze(0).cpu().numpy()  # (frames, hidden_dim)
            
            if normalize:
                # Normalize per frame (optional)
                feat = (feat - np.mean(feat, axis=1, keepdims=True)) / (
                    np.std(feat, axis=1, keepdims=True) + 1e-8
                )
            
            all_layers[layer_idx] = feat
        
        return all_layers
    
    def extract_layer(
        self,
        audio: np.ndarray,
        layer_idx: int,
        sr: int = 16000,
    ) -> np.ndarray:
        """
        Extract features from a single layer.
        
        Args:
            audio: Audio waveform.
            layer_idx: Layer index (0 to num_layers-1).
            sr: Sample rate.
        
        Returns:
            Features shape (n_frames, hidden_dim).
        """
        all_layers = self.extract_all_layers(audio, sr, normalize=True)
        return all_layers[layer_idx]
    
    def get_frame_times(
        self,
        audio_length: int,
        sr: int = 16000,
    ) -> np.ndarray:
        """
        Get time stamps for each frame.
        
        HuBERT typically has a hop_length of 320 samples (20ms at 16kHz).
        
        Args:
            audio_length: Length of audio array.
            sr: Sample rate.
        
        Returns:
            Frame times in seconds, shape (n_frames,).
        """
        hop_length = 320
        n_frames = (audio_length + sr // 2) // hop_length
        return np.arange(n_frames) * (hop_length / sr)


def extract_ssl_features(
    audio: np.ndarray,
    sr: int = 16000,
    model_id: str = "facebook/hubert-large-ls60-japanese",
    layers: Optional[List[int]] = None,
) -> Dict[int, np.ndarray]:
    """
    Convenience function to extract SSL features.
    
    Args:
        audio: Audio array.
        sr: Sample rate.
        model_id: HuBERT model ID.
        layers: Specific layers to extract. If None, extract all.
    
    Returns:
        {layer_idx: features}.
    """
    extractor = HuBERTFeatureExtractor(model_id=model_id)
    all_layers = extractor.extract_all_layers(audio, sr)
    
    if layers is None:
        return all_layers
    else:
        return {k: all_layers[k] for k in layers if k in all_layers}


if __name__ == "__main__":
    # Example usage
    import librosa
    
    print("Loading example audio...")
    audio, sr = librosa.load(librosa.ex("brahms"), sr=16000, mono=True, duration=3)
    
    print("Extracting HuBERT features...")
    extractor = HuBERTFeatureExtractor(
        model_id="facebook/hubert-large-ls60-japanese"
    )
    
    features = extractor.extract_all_layers(audio, sr)
    print(f"Extracted {len(features)} layers")
    for layer_idx, feat in features.items():
        print(f"  Layer {layer_idx}: {feat.shape}")
