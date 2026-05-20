"""
Pronunciation confusion detection in L2 speech.

Based on Sun & McIntosh's current work with UCL team:
detect confused phoneme pairs using Bhattacharyya coefficient (BC)
on phoneme posteriorgrams.

Detects patterns like /l/–/r/ confusion in spontaneous speech.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import bhattacharyya
from scipy.stats import entropy


def bhattacharyya_coefficient(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """
    Compute Bhattacharyya coefficient between two probability distributions.
    
    BC = 0: completely different
    BC = 1: identical
    
    Args:
        p: Probability distribution 1, shape (n_classes,).
        q: Probability distribution 2, shape (n_classes,).
    
    Returns:
        BC value in [0, 1].
    """
    # Normalize to ensure valid probabilities
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    p = p / (np.sum(p) + 1e-10)
    q = q / (np.sum(q) + 1e-10)
    
    # BC distance
    bc_dist = bhattacharyya(p, q)
    
    # Convert distance to similarity (0-1)
    bc = np.exp(-bc_dist)
    
    return float(bc)


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """
    Compute KL divergence from p to q.
    
    Args:
        p: Reference distribution.
        q: Comparison distribution.
    
    Returns:
        KL divergence.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    p = p / (np.sum(p) + 1e-10)
    q = q / (np.sum(q) + 1e-10)
    
    kl = entropy(p, q)
    return float(kl)


class PhonemeConfusionDetector:
    """
    Detect confused phoneme pairs in L2 speech.
    
    Usage:
        detector = PhonemeConfusionDetector()
        
        # Get posteriorgrams from ASR model
        # posteriorgrams[t] = log probabilities for each phoneme at frame t
        
        confusion_pairs = detector.detect_confusions(
            posteriorgrams,
            phoneme_list,
            threshold=0.15,  # BC < 0.15 means confused
        )
        
        for pair, bc_score in confusion_pairs:
            print(f"Confused: {pair[0]} <-> {pair[1]}, BC={bc_score:.3f}")
    """
    
    def __init__(self):
        """Initialize detector."""
        self.confusion_history = {}
    
    @staticmethod
    def compute_phoneme_distribution(
        posteriorgrams: np.ndarray,
        phoneme_idx: int,
    ) -> np.ndarray:
        """
        Aggregate posteriorgram for a specific phoneme across time.
        
        Args:
            posteriorgrams: Log probabilities, shape (n_frames, n_phonemes).
            phoneme_idx: Phoneme index.
        
        Returns:
            Aggregated distribution (mean posteriorgram).
        """
        return np.mean(posteriorgrams[:, phoneme_idx:phoneme_idx+1], axis=0)
    
    def detect_confusions(
        self,
        posteriorgrams: np.ndarray,
        phoneme_list: List[str],
        similarity_metric: str = "bc",
        threshold: float = 0.15,
        top_k: Optional[int] = 10,
    ) -> List[Tuple[Tuple[str, str], float]]:
        """
        Detect confused phoneme pairs.
        
        Args:
            posteriorgrams: Log probabilities, shape (n_frames, n_phonemes).
            phoneme_list: List of phoneme names.
            similarity_metric: "bc" (Bhattacharyya) or "kl" (KL divergence).
            threshold: Threshold for confusion (lower = more confused).
                For BC: < 0.15 indicates confusion
                For KL: > 0.5 indicates confusion
            top_k: Return top K most confused pairs. If None, return all.
        
        Returns:
            List of [(phoneme1, phoneme2), score] sorted by confusion strength.
        """
        n_phonemes = len(phoneme_list)
        
        # Compute distance matrix
        confusion_matrix = np.zeros((n_phonemes, n_phonemes))
        
        for i in range(n_phonemes):
            p_i = posteriorgrams[:, i]
            
            for j in range(i + 1, n_phonemes):
                p_j = posteriorgrams[:, j]
                
                if similarity_metric == "bc":
                    score = bhattacharyya_coefficient(p_i, p_j)
                elif similarity_metric == "kl":
                    score = kl_divergence(p_i, p_j)
                else:
                    raise ValueError(f"Unknown metric: {similarity_metric}")
                
                confusion_matrix[i, j] = score
                confusion_matrix[j, i] = score
        
        # Extract confused pairs
        confused_pairs = []
        
        for i in range(n_phonemes):
            for j in range(i + 1, n_phonemes):
                score = confusion_matrix[i, j]
                
                # Check if confused (threshold depends on metric)
                is_confused = False
                if similarity_metric == "bc" and score < threshold:
                    is_confused = True
                elif similarity_metric == "kl" and score > threshold:
                    is_confused = True
                
                if is_confused:
                    confused_pairs.append((
                        (phoneme_list[i], phoneme_list[j]),
                        float(score),
                    ))
        
        # Sort by confusion strength (ascending for BC, descending for KL)
        if similarity_metric == "bc":
            confused_pairs.sort(key=lambda x: x[1])
        else:
            confused_pairs.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            confused_pairs = confused_pairs[:top_k]
        
        return confused_pairs
    
    def analyze_speaker(
        self,
        speech_samples: Dict[str, np.ndarray],
        phoneme_list: List[str],
        similarity_metric: str = "bc",
        threshold: float = 0.15,
    ) -> Dict:
        """
        Analyze a single L2 speaker's pronunciation confusions.
        
        Args:
            speech_samples: {phoneme: posteriorgrams_array}.
            phoneme_list: List of phoneme names.
            similarity_metric: "bc" or "kl".
            threshold: Confusion threshold.
        
        Returns:
            Analysis report.
        """
        # Stack posteriorgrams
        posteriorgrams_list = []
        phoneme_indices = {}
        
        for i, phoneme in enumerate(phoneme_list):
            if phoneme in speech_samples:
                posteriors = speech_samples[phoneme]
                if posteriors.size > 0:
                    posteriorgrams_list.append(posteriors)
                    phoneme_indices[i] = phoneme
        
        if not posteriorgrams_list:
            return {
                "status": "no_data",
                "message": "No posteriorgrams found",
            }
        
        # Concatenate all posteriorgrams
        all_posteriorgrams = np.vstack(posteriorgrams_list)
        
        # Detect confusions
        confused_pairs = self.detect_confusions(
            all_posteriorgrams,
            phoneme_list,
            similarity_metric=similarity_metric,
            threshold=threshold,
        )
        
        return {
            "status": "success",
            "confused_pairs": confused_pairs,
            "total_phonemes": len(phoneme_list),
            "metric": similarity_metric,
            "threshold": threshold,
        }
    
    @staticmethod
    def recommend_confusion_pairs_japanese() -> List[Tuple[str, str]]:
        """
        Return common confusion pairs for Japanese L2 speakers.
        
        Based on linguistic literature.
        """
        return [
            ("ら", "だ"),  # /r/ vs /d/
            ("ら", "ぱ"),  # /r/ vs /p/
            ("ぱ", "ば"),  # /p/ vs /b/
            ("ぱ", "ふぁ"), # /p/ vs /f/
            ("さ", "しゃ"), # /s/ vs /ʃ/
            ("り", "り"),  # Pitch accent variation
        ]


def extract_posteriorgrams_from_whisper(
    audio: np.ndarray,
    sr: int = 16000,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract phoneme posteriorgrams from faster-whisper CTC layer.
    
    Note: This is a workaround. For production, train a dedicated
    phoneme recognizer or use wav2vec2 + CTC head.
    
    Args:
        audio: Audio waveform.
        sr: Sample rate.
    
    Returns:
        (posteriorgrams, phoneme_list) where posteriorgrams has shape
        (n_frames, n_phonemes).
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("faster-whisper not installed")
    
    model = WhisperModel("tiny")
    segments, _ = model.transcribe(audio, language="ja")
    
    # TODO: Extract CTC posteriors if available
    # This is a placeholder - faster-whisper doesn't directly expose CTC outputs
    
    warnings.warn(
        "Direct CTC posterior extraction from faster-whisper not yet implemented. "
        "Consider using wav2vec2-large-xlsr-japanese with CTC head for phoneme posteriors."
    )
    
    return None, None


if __name__ == "__main__":
    # Example
    np.random.seed(42)
    
    # Simulate posteriorgrams for 5 phonemes, 100 frames
    n_frames = 100
    n_phonemes = 5
    phoneme_list = ["a", "i", "u", "e", "o"]
    
    # Create confusion: phoneme 0 and 1 are similar
    posteriorgrams = np.random.dirichlet(np.ones(n_phonemes), size=n_frames)
    posteriorgrams[:, 0] = 0.4 + 0.1 * np.random.randn(n_frames)
    posteriorgrams[:, 1] = 0.35 + 0.1 * np.random.randn(n_frames)
    posteriorgrams = np.abs(posteriorgrams) / np.sum(posteriorgrams, axis=1, keepdims=True)
    
    detector = PhonemeConfusionDetector()
    confused = detector.detect_confusions(
        posteriorgrams,
        phoneme_list,
        similarity_metric="bc",
        threshold=0.3,
    )
    
    print("Detected confusions:")
    for (p1, p2), score in confused:
        print(f"  {p1} <-> {p2}: BC={score:.4f}")
