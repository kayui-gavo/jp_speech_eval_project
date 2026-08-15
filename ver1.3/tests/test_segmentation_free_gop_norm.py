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
        self.assertEqual(result.summary["japanese_adaptation"], "phone_only_wildcard_mask")
        self.assertFalse(result.summary["canonical_log_posterior_is_phone_local"])
        self.assertEqual(len(result.evidence), 3)
        self.assertTrue(all(row.occ_i >= 0 for row in result.evidence))

    def test_high_level_japanese_wildcard_excludes_pause_and_control_tokens(self) -> None:
        logits = np.asarray(
            [
                [5.0, 2.0, 0.0, 0.0, 3.0, 3.0, 3.0],
                [0.0, 5.0, 1.0, 0.5, 4.0, 4.0, 4.0],
                [5.0, 1.0, 0.0, 0.0, 3.0, 3.0, 3.0],
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
        self.assertEqual(inventory, {"a", "i", "N"})
        self.assertNotIn("pau", inventory)
        self.assertNotIn("sil", inventory)
        self.assertNotIn("UNK", inventory)
        self.assertTrue(result.summary["wildcard_excludes_nonsegmental_control_pause_tokens"])

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
