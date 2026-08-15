from __future__ import annotations

import unittest

import numpy as np

from jp_speech_eval.ctc_sequence import ctc_forward_logprob_vectorized
from jp_speech_eval.japanese_phoneme_gop import ctc_forward_logprob


class VectorizedCtcForwardTest(unittest.TestCase):
    @staticmethod
    def _log_probs(seed: int, frames: int = 12, vocab: int = 5) -> np.ndarray:
        rng = np.random.default_rng(seed)
        logits = rng.normal(size=(frames, vocab))
        maxima = np.max(logits, axis=1, keepdims=True)
        shifted = logits - maxima
        return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

    def test_matches_scalar_reference_for_distinct_targets(self) -> None:
        log_probs = self._log_probs(1)
        for target in ([1], [1, 2], [1, 2, 3], [3, 2, 1]):
            scalar = ctc_forward_logprob(log_probs, target, blank_id=0)
            vectorized = ctc_forward_logprob_vectorized(log_probs, target, blank_id=0)
            self.assertAlmostEqual(vectorized, scalar, places=10)

    def test_matches_scalar_reference_for_repeated_labels(self) -> None:
        log_probs = self._log_probs(2, frames=15)
        for target in ([1, 1], [1, 2, 2], [3, 3, 3]):
            scalar = ctc_forward_logprob(log_probs, target, blank_id=0)
            vectorized = ctc_forward_logprob_vectorized(log_probs, target, blank_id=0)
            self.assertAlmostEqual(vectorized, scalar, places=10)

    def test_matches_scalar_reference_for_empty_sequence(self) -> None:
        log_probs = self._log_probs(3)
        scalar = ctc_forward_logprob(log_probs, [], blank_id=0)
        vectorized = ctc_forward_logprob_vectorized(log_probs, [], blank_id=0)
        self.assertAlmostEqual(vectorized, scalar, places=10)

    def test_impossible_long_repeated_target_remains_negative_infinity(self) -> None:
        # Three repeated phones require at least five CTC frames (a blank must
        # separate each repetition). With only four frames the path is impossible.
        log_probs = self._log_probs(4, frames=4)
        scalar = ctc_forward_logprob(log_probs, [1, 1, 1], blank_id=0)
        vectorized = ctc_forward_logprob_vectorized(log_probs, [1, 1, 1], blank_id=0)
        self.assertTrue(np.isneginf(scalar))
        self.assertTrue(np.isneginf(vectorized))

    def test_rejects_blank_inside_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain CTC blank"):
            ctc_forward_logprob_vectorized(self._log_probs(5), [1, 0, 2], blank_id=0)


if __name__ == "__main__":
    unittest.main()
