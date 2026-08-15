from __future__ import annotations

import unittest

import numpy as np

from jp_speech_eval.japanese_segmentation_free_norm import (
    METHOD,
    compute_japanese_construct_aware_norm_features,
)


class JapaneseConstructAwareNormTest(unittest.TestCase):
    def test_special_mora_rows_use_canonical_plus_deletion_policy(self) -> None:
        vocab = {"PAD": 0, "k": 1, "a": 2, "N": 3, "cl": 4, "g": 5}
        logits = np.asarray(
            [
                [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 6.0, 0.5, 0.5, 0.5, 1.0],
                [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 6.0, 0.5, 0.5, 0.0],
                [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 6.0, 0.5, 0.0],
                [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        result = compute_japanese_construct_aware_norm_features(
            logits,
            ["k", "a", "N"],
            vocab=vocab,
            blank_id=0,
            model_id="synthetic",
            revision="test",
        )
        self.assertTrue(result.available)
        self.assertEqual(result.method, METHOD)
        self.assertEqual(result.summary["ordinary_segmental_row_count"], 2)
        self.assertEqual(result.summary["special_mora_row_count"], 1)
        self.assertEqual(result.summary["special_mora_indices"], [2])
        self.assertEqual(
            result.summary["row_policies"][2],
            "special_mora_canonical_plus_deletion_only",
        )
        self.assertFalse(
            result.summary["special_mora_normalized_value_directly_comparable_to_ordinary_rows"]
        )
        self.assertFalse(result.summary["cross_role_raw_gop_comparison_allowed"])
        self.assertNotIn("N", result.summary["ordinary_wildcard_phone_inventory"])
        self.assertNotIn("cl", result.summary["ordinary_wildcard_phone_inventory"])

    def test_all_special_mora_target_does_not_require_ordinary_inventory(self) -> None:
        vocab = {"PAD": 0, "N": 1, "cl": 2}
        logits = np.asarray(
            [
                [5.0, 0.0, 0.0],
                [0.0, 6.0, 0.0],
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 6.0],
                [5.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        result = compute_japanese_construct_aware_norm_features(
            logits,
            ["N", "cl"],
            vocab=vocab,
            blank_id=0,
        )
        self.assertTrue(result.available)
        self.assertEqual(result.summary["ordinary_segmental_row_count"], 0)
        self.assertEqual(result.summary["special_mora_row_count"], 2)
        self.assertEqual(result.summary["ordinary_wildcard_phone_inventory"], [])

    def test_control_tokens_never_enter_ordinary_wildcard(self) -> None:
        vocab = {"PAD": 0, "a": 1, "k": 2, "N": 3, "cl": 4, "pau": 5, "sil": 6, "UNK": 7}
        logits = np.zeros((6, len(vocab)), dtype=np.float64)
        result = compute_japanese_construct_aware_norm_features(
            logits,
            ["a"],
            vocab=vocab,
            blank_id=0,
        )
        self.assertTrue(result.available)
        self.assertEqual(set(result.summary["ordinary_wildcard_phone_inventory"]), {"a", "k"})


if __name__ == "__main__":
    unittest.main()
