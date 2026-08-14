import unittest

import numpy as np

from jp_speech_eval.dual_ctc_phone_candidate import (
    extract_phoneme_logits,
    normalize_phoneme_vocab,
    validate_immutable_revision,
)


class DualCtcPhoneCandidateTest(unittest.TestCase):
    def test_revision_must_be_immutable(self) -> None:
        for bad in ("", "main", "master", "latest", "abc123"):
            with self.assertRaises(ValueError):
                validate_immutable_revision(bad)
        self.assertEqual(
            validate_immutable_revision("0123456789abcdef0123456789abcdef01234567"),
            "0123456789abcdef0123456789abcdef01234567",
        )

    def test_normalize_phoneme_vocab_uses_explicit_blank(self) -> None:
        vocab, blank_id = normalize_phoneme_vocab({"<blank>": 0, "a": 1, "i": 2, "N": 3, "cl": 4})
        self.assertEqual(blank_id, 0)
        self.assertEqual(vocab["cl"], 4)

    def test_normalize_phoneme_vocab_rejects_ambiguous_blank(self) -> None:
        with self.assertRaises(ValueError):
            normalize_phoneme_vocab({"<blank>": 0, "PAD": 1, "a": 2})

    def test_extract_phoneme_logits_accepts_mapping_and_strips_batch_axis(self) -> None:
        logits = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
        result = extract_phoneme_logits({"phoneme_logits": logits})
        self.assertEqual(result.shape, (4, 6))
        self.assertTrue(np.isfinite(result).all())

    def test_extract_phoneme_logits_rejects_missing_head(self) -> None:
        with self.assertRaises(ValueError):
            extract_phoneme_logits({"kana_logits": np.zeros((1, 3, 4), dtype=np.float32)})


if __name__ == "__main__":
    unittest.main()
