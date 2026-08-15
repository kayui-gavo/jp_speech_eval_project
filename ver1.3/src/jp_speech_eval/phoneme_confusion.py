"""Legacy phoneme-confusion helpers, quarantined from pronunciation scoring.

The previous version of this module was not scientifically safe:

* it cited an unverifiable project attribution;
* it treated a *low* Bhattacharyya coefficient (low distributional overlap) as
  evidence that two phones were "confused", reversing the metric's semantics;
* it compared posterior columns across time as if they were calibrated phone
  distributions; and
* it exposed kana strings and even an identity pair as "phoneme" confusions.

None of that is permitted to feed the Japanese pronunciation product.  The
active research route is target-conditioned phone CTC / GOP in
``japanese_phoneme_gop.py`` and the restricted substitution policy in
``japanese_phone_substitutions.py``.

The generic Bhattacharyya/KL utilities are retained because they are
mathematically useful when applied to *proper probability distributions*.
``PhonemeConfusionDetector`` remains only as a compatibility quarantine: its
legacy heuristic is disabled unless a caller explicitly opts in, and even then
its output is labeled exploratory rather than a pronunciation decision.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import entropy

from .japanese_phone_substitutions import RESTRICTED_NEIGHBORS


LEGACY_HEURISTIC_STATUS = "deprecated_not_validated_for_pronunciation_scoring"


def _probability_vector(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("distribution must be finite and non-empty")
    if np.any(array < 0.0):
        raise ValueError("distribution must be non-negative")
    total = float(np.sum(array))
    if total <= 0.0:
        raise ValueError("distribution must have positive mass")
    return array / total


def bhattacharyya_coefficient(p: np.ndarray, q: np.ndarray) -> float:
    """Return probability-distribution overlap in ``[0, 1]``.

    ``1`` means identical distributions and ``0`` means disjoint support.
    A high coefficient is *similarity*, not evidence of a pronunciation error.
    """
    p_norm = _probability_vector(p)
    q_norm = _probability_vector(q)
    if p_norm.shape != q_norm.shape:
        raise ValueError("distributions must have the same shape")
    return float(np.clip(np.sum(np.sqrt(p_norm * q_norm)), 0.0, 1.0))


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Return ``-log(BC)``; zero means identical distributions."""
    coefficient = bhattacharyya_coefficient(p, q)
    return float(-np.log(max(coefficient, np.finfo(np.float64).tiny)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Return ``D_KL(p || q)`` for proper probability distributions."""
    p_norm = _probability_vector(p)
    q_norm = _probability_vector(q)
    if p_norm.shape != q_norm.shape:
        raise ValueError("distributions must have the same shape")
    return float(entropy(p_norm, q_norm))


class PhonemeConfusionDetector:
    """Compatibility wrapper around the quarantined legacy heuristic.

    This class must not be used by the product scorer.  New code should use
    target-conditioned Japanese phone CTC/GOP evidence.  To make accidental
    reuse fail closed, legacy execution requires ``allow_legacy_heuristic``.
    """

    def __init__(self, *, allow_legacy_heuristic: bool = False) -> None:
        self.allow_legacy_heuristic = bool(allow_legacy_heuristic)
        self.confusion_history: Dict = {}

    def _guard(self) -> None:
        if not self.allow_legacy_heuristic:
            raise RuntimeError(
                "PhonemeConfusionDetector is deprecated and disabled: use "
                "Japanese phone CTC/GOP evidence instead"
            )
        warnings.warn(
            "Running deprecated posterior-overlap heuristic; output is exploratory "
            "and must not be interpreted as pronunciation correctness.",
            RuntimeWarning,
            stacklevel=2,
        )

    @staticmethod
    def compute_phoneme_distribution(
        posteriorgrams: np.ndarray,
        phoneme_idx: int,
    ) -> np.ndarray:
        """Return the selected posterior column as a normalized time profile.

        This is retained solely for legacy reproducibility.  It is *not* a
        calibrated categorical distribution over phones and should not be used
        for current MDD/GOP decisions.
        """
        values = np.asarray(posteriorgrams, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("posteriorgrams must have shape (frames, classes)")
        if not 0 <= int(phoneme_idx) < values.shape[1]:
            raise IndexError("phoneme_idx out of range")
        return _probability_vector(np.clip(values[:, int(phoneme_idx)], 0.0, None))

    def detect_confusions(
        self,
        posteriorgrams: np.ndarray,
        phoneme_list: Sequence[str],
        similarity_metric: str = "bc",
        threshold: float = 0.85,
        top_k: Optional[int] = 10,
    ) -> List[Tuple[Tuple[str, str], float]]:
        """Run the disabled legacy time-profile similarity heuristic.

        When explicitly enabled, high BC means similar temporal profiles; low
        Bhattacharyya distance means similar profiles.  This fixes the old
        reversed inequality but still does **not** establish perceptual phone
        confusion or learner error.
        """
        self._guard()
        values = np.asarray(posteriorgrams, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(phoneme_list):
            raise ValueError("posteriorgrams/classes do not match phoneme_list")

        pairs: List[Tuple[Tuple[str, str], float]] = []
        profiles = [self.compute_phoneme_distribution(values, i) for i in range(values.shape[1])]
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                if similarity_metric == "bc":
                    score = bhattacharyya_coefficient(profiles[i], profiles[j])
                    selected = score >= float(threshold)
                elif similarity_metric == "bhattacharyya_distance":
                    score = bhattacharyya_distance(profiles[i], profiles[j])
                    selected = score <= float(threshold)
                elif similarity_metric == "kl":
                    score = kl_divergence(profiles[i], profiles[j])
                    selected = score <= float(threshold)
                else:
                    raise ValueError(f"unknown metric: {similarity_metric}")
                if selected:
                    pairs.append(((str(phoneme_list[i]), str(phoneme_list[j])), float(score)))

        if similarity_metric == "bc":
            pairs.sort(key=lambda item: item[1], reverse=True)
        else:
            pairs.sort(key=lambda item: item[1])
        return pairs if top_k is None else pairs[: int(top_k)]

    def analyze_speaker(
        self,
        speech_samples: Dict[str, np.ndarray],
        phoneme_list: Sequence[str],
        similarity_metric: str = "bc",
        threshold: float = 0.85,
    ) -> Dict:
        self._guard()
        arrays = [
            np.asarray(speech_samples[phone])
            for phone in phoneme_list
            if phone in speech_samples and np.asarray(speech_samples[phone]).size > 0
        ]
        if not arrays:
            return {
                "status": "no_data",
                "heuristic_status": LEGACY_HEURISTIC_STATUS,
                "confused_pairs": [],
            }
        all_posteriorgrams = np.vstack(arrays)
        return {
            "status": "legacy_exploratory_only",
            "heuristic_status": LEGACY_HEURISTIC_STATUS,
            "confused_pairs": self.detect_confusions(
                all_posteriorgrams,
                phoneme_list,
                similarity_metric=similarity_metric,
                threshold=threshold,
            ),
            "metric": similarity_metric,
            "threshold": float(threshold),
        }

    @staticmethod
    def recommend_confusion_pairs_japanese() -> List[Tuple[str, str]]:
        """Return canonical-phone research neighbors from the active policy.

        These are candidate search neighbors, not established learner-error
        frequencies.  Each unordered pair is emitted once.
        """
        pairs: set[Tuple[str, str]] = set()
        for phone, neighbors in RESTRICTED_NEIGHBORS.items():
            for neighbor in neighbors:
                pair = tuple(sorted((str(phone), str(neighbor))))
                if pair[0] != pair[1]:
                    pairs.add(pair)
        return sorted(pairs)


def extract_posteriorgrams_from_whisper(
    audio: np.ndarray,
    sr: int = 16000,
) -> Tuple[np.ndarray, List[str]]:
    """Fail closed: Whisper does not expose a Japanese phoneme CTC head here."""
    del audio, sr
    raise NotImplementedError(
        "faster-whisper is not a phoneme-CTC backend in this project; use the "
        "pinned Japanese phone-CTC research backends instead"
    )
