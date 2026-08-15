import unittest

import numpy as np

from jp_speech_eval.segmentation_free_gop_norm import (
    REFERENCE_METHOD,
    compute_segmentation_free_norm_features,
    sd_norm_alternative_graph_forward,
)


class SegmentationFreeGopNormTest(unittest.TestCase):
    def setUp(self) -> None:
        raw = np.asarray(
            [
                [3, 1, 2, 4, 1, 3, 2, 1],
                [1, 4, 1, 1, 3, 1, 4, 2],
                [2, 1, 4, 1, 2, 4, 1, 3],
                [1, 2, 1, 2, 4, 1, 2, 4],
            ],
            dtype=np.float64,
        )
        # Reference implementation convention is vocab x time; project code is
        # time x vocab. The constants below were frozen from the public
        # taslpro26 normalized-forward recursion on this exact probability grid.
        probs_vocab_time = raw / np.sum(raw, axis=0, keepdims=True)
        self.probs = probs_vocab_time.T

    def test_reference_regression_three_distinct_phones(self) -> None:
        seq = [1, 2, 3]
        expected = [
            (-4.126133797285917, 1.0581338633585355),
            (-4.234785161585473, 1.1607951727947867),
            (-4.0329518725446025, 2.5023448211404795),
        ]
        for index, (expected_lp, expected_occ) in enumerate(expected):
            graph_lp, occ = sd_norm_alternative_graph_forward(
                self.probs,
                seq,
                phone_index=index,
                blank_id=0,
            )
            self.assertAlmostEqual(graph_lp, expected_lp, places=10)
            self.assertAlmostEqual(occ, expected_occ, places=10)

    def test_reference_regression_repeated_phone_context(self) -> None:
        seq = [1, 1, 2]
        expected = [
            (-4.597312853207416, 0.9718420262807377),
            (-4.531635176510009, 1.0751946605839873),
            (-4.781852909025005, 1.894557851622168),
        ]
        for index, (expected_lp, expected_occ) in enumerate(expected):
            graph_lp, occ = sd_norm_alternative_graph_forward(
                self.probs,
                seq,
                phone_index=index,
                blank_id=0,
            )
            self.assertAlmostEqual(graph_lp, expected_lp, places=10)
            self.assertAlmostEqual(occ, expected_occ, places=10)

    def test_one_phone_one_frame_denominator_is_full_frame_mass(self) -> None:
        probs = np.asarray([[0.25, 0.50, 0.25]], dtype=np.float64)
        graph_lp, occ = sd_norm_alternative_graph_forward(
            probs,
            [1],
            phone_index=0,
            blank_id=0,
        )
        self.assertAlmostEqual(graph_lp, 0.0, places=12)
        self.assertAlmostEqual(occ, 0.75, places=12)

    def test_low_level_mask_removes_disallowed_wildcard_mass_without_renormalizing_frames(self) -> None:
        probs = np.asarray([[0.10, 0.40, 0.20, 0.30]], dtype=np.float64)
        unmasked_lp, unmasked_occ = sd_norm_alternative_graph_forward(
            probs,
            [1],
            phone_index=0,
            blank_id=0,
        )
        masked_lp, masked_occ = sd_norm_alternative_graph_forward(
            probs,
            [1],
            phone_index=0,
            blank_id=0,
            wildcard_token_ids=[1, 2],
        )
        self.assertAlmostEqual(unmasked_lp, 0.0, places=12)
        self.assertAlmostEqual(unmasked_occ, 0.90, places=12)
        # Token id 3 is removed only from wildcard alternatives. Its 0.30 mass
        # is not redistributed over allowed phone tokens.
        self.assertAlmostEqual(masked_lp, np.log(0.70), places=12)
        self.assertAlmostEqual(masked_occ, 0.60 / 0.70, places=12)

    def test_occ_is_feature_not_duration_or_score(self) -> None:
        logits = np.log(np.maximum(self.probs, 1e-12))
        result = compute_segmentation_free_norm_features(
            logits,
            ["a", "b", "c"],
            vocab={"<blank>": 0, "a": 1, "b": 2, "c": 3},
            blank_id=0,
            model_id="synthetic",
            revision="test",
        )
        self.assertTrue(result.available)
        self.assertFalse(result.score_mapped)
        self.assertFalse(result.product_calibrated)
        self.assertFalse(result.summary["occ_i_is_physical_phone_duration"])
        self.assertFalse(result.summary["individual_feature_is_pronunciation_decision"])
        self.assertTrue(result.summary["requires_labeled_downstream_validation"])
        self.assertEqual(result.summary["low_level_reference_method"], REFERENCE_METHOD)
        self.assertEqual(
            result.summary["japanese_adaptation"],
            "position_specific_ordinary_vs_special_mora_wildcard_mask",
        )
        self.assertFalse(result.summary["canonical_log_posterior_is_phone_local"])
        self.assertEqual(len(result.evidence), 3)
        self.assertTrue(all(row.occ_i >= 0 for row in result.evidence))

    def test_high_level_japanese_wildcard_excludes_pause_control_and_special_mora_tokens(self) -> None:
        logits = np.asarray(
            [
                [5.0, 2.0, 0.0, 0.0, 3.0, 3.0, 3.0, 1.0],
                [0.0, 5.0, 1.0, 0.5, 4.0, 4.0, 4.0, 1.0],
                [5.0, 1.0, 0.0, 0.0, 3.0, 3.0, 3.0, 1.0],
            ],
            dtype=np.float64,
        )
        vocab = {
            "PAD": 0,
            "a": 1,
            "i": 2,
            "N": 3,
            "pau": 4,
            "sil": 5,
            "UNK": 6,
            "cl": 7,
        }
        result = compute_segmentation_free_norm_features(
            logits,
            ["a"],
            vocab=vocab,
            blank_id=0,
            model_id="synthetic-japanese",
            revision="test",
        )
        self.assertTrue(result.available)
        inventory = set(result.summary["wildcard_phone_inventory"])
        self.assertEqual(inventory, {"a", "i"})
        self.assertNotIn("N", inventory)
        self.assertNotIn("cl", inventory)
        self.assertNotIn("pau", inventory)
        self.assertNotIn("sil", inventory)
        self.assertNotIn("UNK", inventory)
        self.assertTrue(result.summary["wildcard_excludes_nonsegmental_control_pause_tokens"])
        self.assertTrue(result.summary["ordinary_wildcard_excludes_special_mora_tokens"])

    def test_special_mora_position_uses_canonical_wildcard_plus_graph_deletion_only(self) -> None:
        # PAD, a, k, N, cl. The target a-N-k contains an ordinary/special/
        # ordinary sequence so the high-level adapter must switch masks by
        # position without changing the low-level reference recurrence.
        vocab = {"PAD": 0, "a": 1, "k": 2, "N": 3, "cl": 4}
        logits = np.full((13, 5), -7.0, dtype=np.float64)
        logits[:, 0] = 0.0
        logits[0, 0] = 9.0
        logits[1:3, 1] = 10.0
        logits[3:5, 0] = 9.0
        logits[5:7, 3] = 10.0
        logits[7:9, 0] = 9.0
        logits[9:11, 2] = 10.0
        logits[11:, 0] = 9.0

        result = compute_segmentation_free_norm_features(
            logits,
            ["a", "N", "k"],
            vocab=vocab,
            blank_id=0,
            model_id="synthetic-japanese",
            revision="test",
        )
        self.assertTrue(result.available)
        self.assertEqual(result.summary["special_mora_position_count"], 1)
        self.assertEqual(result.summary["ordinary_position_count"], 2)
        self.assertEqual(
            result.summary["special_mora_policy"],
            "canonical_wildcard_plus_graph_deletion_path_only",
        )
        self.assertEqual(result.summary["position_wildcard_size_min"], 1)
        self.assertEqual(result.summary["position_wildcard_size_max"], 2)
        self.assertFalse(result.summary["special_mora_occ_i_is_physical_duration"])

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= np.sum(probs, axis=1, keepdims=True)
        expected_lp, expected_occ = sd_norm_alternative_graph_forward(
            probs,
            [1, 3, 2],
            phone_index=1,
            blank_id=0,
            wildcard_token_ids=[3],
        )
        n_row = result.evidence[1]
        self.assertAlmostEqual(n_row.denominator_graph_log_posterior, expected_lp, places=10)
        self.assertAlmostEqual(n_row.occ_i, expected_occ, places=10)

    def test_all_special_mora_target_does_not_require_ordinary_wildcard_inventory(self) -> None:
        # High-level Japanese semantics should still be defined even if a tiny
        # synthetic vocabulary contains only blank and one special-mora target.
        logits = np.asarray(
            [
                [8.0, 0.0],
                [0.0, 8.0],
                [8.0, 0.0],
            ],
            dtype=np.float64,
        )
        result = compute_segmentation_free_norm_features(
            logits,
            ["N"],
            vocab={"PAD": 0, "N": 1},
            blank_id=0,
            model_id="synthetic-special-only",
            revision="test",
        )
        self.assertTrue(result.available)
        self.assertEqual(result.summary["ordinary_wildcard_phone_inventory"], [])
        self.assertEqual(result.summary["special_mora_position_count"], 1)
        self.assertEqual(result.summary["ordinary_position_count"], 0)

    def test_rejects_blank_inside_canonical_sequence(self) -> None:
        with self.assertRaises(ValueError):
            sd_norm_alternative_graph_forward(
                self.probs,
                [1, 0, 2],
                phone_index=1,
                blank_id=0,
            )


if __name__ == "__main__":
    unittest.main()
