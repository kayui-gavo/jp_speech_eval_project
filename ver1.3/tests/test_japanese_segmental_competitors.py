from __future__ import annotations

import unittest

from jp_speech_eval.japanese_phoneme_gop import segmental_competitor_ids


class JapaneseSegmentalCompetitorTest(unittest.TestCase):
    def test_ordinary_clarity_competitors_exclude_special_mora_and_controls(self) -> None:
        vocab = {
            "PAD": 0,
            "a": 1,
            "k": 2,
            "N": 3,
            "cl": 4,
            "pau": 5,
            "sil": 6,
            "UNK": 7,
        }
        ids = set(segmental_competitor_ids(vocab, blank_id=0))
        self.assertEqual(ids, {1, 2})

    def test_special_mora_can_still_exist_as_target_token(self) -> None:
        vocab = {"PAD": 0, "N": 1, "cl": 2, "a": 3}
        # This helper defines the *competitor* inventory only. Presence of N/cl
        # in vocab remains intact for canonical target/deletion evidence.
        self.assertIn("N", vocab)
        self.assertIn("cl", vocab)
        self.assertEqual(set(segmental_competitor_ids(vocab, blank_id=0)), {3})


if __name__ == "__main__":
    unittest.main()
