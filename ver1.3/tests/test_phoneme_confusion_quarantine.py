from __future__ import annotations

import unittest

import numpy as np

from jp_speech_eval.phoneme_confusion import (
    PhonemeConfusionDetector,
    bhattacharyya_coefficient,
    bhattacharyya_distance,
    extract_posteriorgrams_from_whisper,
)


class PhonemeConfusionQuarantineTest(unittest.TestCase):
    def test_bhattacharyya_semantics_are_not_reversed(self) -> None:
        identical = bhattacharyya_coefficient(np.array([0.8, 0.2]), np.array([0.8, 0.2]))
        disjoint = bhattacharyya_coefficient(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        self.assertAlmostEqual(identical, 1.0, places=7)
        self.assertAlmostEqual(disjoint, 0.0, places=7)
        self.assertAlmostEqual(bhattacharyya_distance(np.array([0.8, 0.2]), np.array([0.8, 0.2])), 0.0, places=7)

    def test_detector_fails_closed_by_default(self) -> None:
        detector = PhonemeConfusionDetector()
        posterior = np.asarray([[0.8, 0.2], [0.7, 0.3]], dtype=np.float64)
        with self.assertRaisesRegex(RuntimeError, "deprecated and disabled"):
            detector.detect_confusions(posterior, ["a", "i"])

    def test_japanese_recommendations_are_canonical_phone_pairs(self) -> None:
        pairs = PhonemeConfusionDetector.recommend_confusion_pairs_japanese()
        self.assertIn(("b", "p"), pairs)
        self.assertIn(("s", "ts"), pairs)
        self.assertNotIn(("り", "り"), pairs)
        self.assertTrue(all(left != right for left, right in pairs))

    def test_whisper_placeholder_now_fails_closed(self) -> None:
        with self.assertRaises(NotImplementedError):
            extract_posteriorgrams_from_whisper(np.zeros(160, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
