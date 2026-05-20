"""
DTW-based feature comparison for MFCC vs HuBERT representations.

Follows the Prosodic ABX methodology: compute distances between
representations using DTW to measure phonetic/prosodic contrast.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import librosa
import numpy as np


def extract_mfcc_features(
    audio: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract MFCC features.
    
    Args:
        audio: Audio waveform.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.
    
    Returns:
        (features, times) where features shape is (n_mfcc, n_frames),
        times shape is (n_frames,).
    """
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Get frame times (hop_length=512 by default in librosa.feature.mfcc)
    hop_length = 512
    times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=hop_length)
    
    # Transpose to (n_frames, n_mfcc)
    return mfcc.T, times


def resample_features(
    features: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """
    Resample features to align with target time grid.
    
    Args:
        features: Input features, shape (n_frames_source, feature_dim).
        source_times: Source frame times.
        target_times: Target frame times.
    
    Returns:
        Resampled features, shape (len(target_times), feature_dim).
    """
    if features.shape[0] != len(source_times):
        raise ValueError("features and source_times length mismatch")
    
    # Linear interpolation for each feature dimension
    resampled = np.interp(
        target_times,
        source_times,
        features,
        axis=0,
        left=features[0],
        right=features[-1],
    )
    return resampled


def dtw_distance(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DTW distance between two sequences.
    
    Args:
        X: Feature matrix, shape (n_X, feature_dim).
        Y: Feature matrix, shape (n_Y, feature_dim).
        metric: Distance metric ("euclidean", "cosine", etc.).
    
    Returns:
        (D, wp) where D is the cost matrix and wp is the warping path.
    """
    D, wp = librosa.sequence.dtw(X=X, Y=Y, metric=metric)
    return D, wp


def normalized_dtw_cost(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """
    Compute normalized DTW cost.
    
    Args:
        X: Feature matrix.
        Y: Feature matrix.
        metric: Distance metric.
    
    Returns:
        Normalized cost (divided by path length).
    """
    D, wp = dtw_distance(X, Y, metric=metric)
    # Normalize by path length
    cost = D[-1, -1] / len(wp)
    return float(cost)


class FeatureComparator:
    """
    Compare MFCC and HuBERT features using DTW.
    
    Usage:
        from ssl_features import HuBERTFeatureExtractor
        from feature_comparison import FeatureComparator
        
        comparator = FeatureComparator()
        
        # Extract both representations
        mfcc_feat, mfcc_times = comparator.get_mfcc(audio, sr=16000)
        hubert_feat = comparator.get_hubert_layer(audio, sr=16000, layer=9)
        
        # Compare with DTW
        mfcc_cost = comparator.dtw_cost(mfcc_feat, ref_mfcc)
        hubert_cost = comparator.dtw_cost(hubert_feat, ref_hubert)
    """
    
    def __init__(self):
        self.hubert_extractor = None
        self._hubert_cache = {}  # Cache HuBERT models
    
    def get_mfcc(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract MFCC features."""
        return extract_mfcc_features(audio, sr=sr, n_mfcc=13)
    
    def get_hubert_layer(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        layer: int = 9,
        model_id: str = "facebook/hubert-large-ls60-japanese",
    ) -> np.ndarray:
        """
        Extract HuBERT features from a specific layer.
        
        Args:
            audio: Audio waveform.
            sr: Sample rate.
            layer: Layer index.
            model_id: HuBERT model.
        
        Returns:
            Features, shape (n_frames, hidden_dim).
        """
        try:
            from .ssl_features import HuBERTFeatureExtractor
        except ImportError:
            raise RuntimeError("ssl_features module not found")
        
        # Load model once
        if model_id not in self._hubert_cache:
            self._hubert_cache[model_id] = HuBERTFeatureExtractor(model_id=model_id)
        
        extractor = self._hubert_cache[model_id]
        all_layers = extractor.extract_all_layers(audio, sr)
        
        if layer not in all_layers:
            raise ValueError(f"Layer {layer} not found. Model has {len(all_layers)} layers.")
        
        return all_layers[layer]
    
    def dtw_cost(
        self,
        user_features: np.ndarray,
        ref_features: np.ndarray,
        metric: str = "euclidean",
    ) -> float:
        """
        Compute normalized DTW cost between user and reference features.
        
        Args:
            user_features: User's features, shape (n_frames, feature_dim).
            ref_features: Reference features, shape (n_frames, feature_dim).
            metric: Distance metric.
        
        Returns:
            Normalized DTW cost.
        """
        # Ensure same feature dimension
        if user_features.shape[1] != ref_features.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: {user_features.shape[1]} vs {ref_features.shape[1]}"
            )
        
        return normalized_dtw_cost(user_features, ref_features, metric=metric)
    
    def compare_representations(
        self,
        user_audio: np.ndarray,
        ref_audio: np.ndarray,
        sr: int = 16000,
        hubert_layers: Optional[list] = None,
    ) -> Dict[str, float]:
        """
        Compare user and reference audio using multiple representations.
        
        Args:
            user_audio: User's audio.
            ref_audio: Reference audio.
            sr: Sample rate.
            hubert_layers: HuBERT layers to test. Default: [9, 15, 23].
        
        Returns:
            {representation: dtw_cost}.
        """
        if hubert_layers is None:
            hubert_layers = [9, 15, 23]
        
        results = {}
        
        # MFCC baseline
        try:
            user_mfcc, _ = self.get_mfcc(user_audio, sr)
            ref_mfcc, _ = self.get_mfcc(ref_audio, sr)
            results["mfcc"] = self.dtw_cost(user_mfcc, ref_mfcc)
        except Exception as e:
            results["mfcc_error"] = str(e)
        
        # HuBERT at multiple layers
        for layer in hubert_layers:
            try:
                user_hubert = self.get_hubert_layer(user_audio, sr, layer=layer)
                ref_hubert = self.get_hubert_layer(ref_audio, sr, layer=layer)
                results[f"hubert_layer_{layer}"] = self.dtw_cost(user_hubert, ref_hubert)
            except Exception as e:
                results[f"hubert_layer_{layer}_error"] = str(e)
        
        return results


if __name__ == "__main__":
    import librosa
    
    # Example
    print("Loading audio...")
    audio1, sr = librosa.load(librosa.ex("brahms"), sr=16000, mono=True, duration=2)
    audio2, _ = librosa.load(librosa.ex("brahms"), sr=sr, mono=True, offset=2, duration=2)
    
    print("Comparing representations...")
    comparator = FeatureComparator()
    
    results = comparator.compare_representations(audio1, audio2, sr=sr)
    
    print("\nDTW Costs:")
    for name, cost in results.items():
        if "error" not in name:
            print(f"  {name}: {cost:.4f}")
