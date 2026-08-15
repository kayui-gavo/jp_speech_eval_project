from dataclasses import replace
import unittest

import numpy as np

from jp_speech_eval.phone_criterion_features import (
    build_phone_criterion_feature_bundle,
    compare_shared_suffix_locality,
)
from jp_speech_eval.segmentation_free_gop import compute_enumerated_fgop_sf_sd_features
from jp_speech_eval.segmentation_free_gop_norm import compute_segmentation_free_norm_features


class PhoneCriterionFeaturesTest(unittest.TestCase):
    def _results(self):
        logits = np.asarray(
            [
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0],
                [5.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        vocab = {"<blank>": 0, "a": 1, "b": 2}
        enumerated = compute_enumerated_fgop_sf_sd_features(
            logits,
            ["a", "b"],
            vocab=vocab,
            blank_id=0,
            model_id="synthetic-phone-ctc",
            revision="0123456789abcdef",
        )
        normalized = compute_segmentation_free_norm_features(
            logits,
            ["a", "b"],
            vocab=vocab,
            blank_id=0,
            model_id="synthetic-phone-ctc",
            revision="0123456789abcdef",
        )
        return enumerated, normalized

    def test_bundle_joins_lpr_and_occ_without_score_mapping(self) -> None:
        enumerated, normalized = self._results()
        bundle = build_phone_criterion_feature_bundle(enumerated, normalized)
        self.assertTrue(bundle.available)
        self.assertFalse(bundle.score_mapped)
        self.assertFalse(bundle.product_calibrated)
        self.assertEqual(bundle.canonical_phones, ["a", "b"])
        self.assertEqual(len(bundle.rows), 2)
        self.assertIn("a", bundle.rows[0].substitution_lprs)
        self.assertIn("b", bundle.rows[0].substitution_lprs)
        self.assertGreaterEqual(bundle.rows[0].occ_i, 0.0)
        self.assertFalse(bundle.summary["cross_model_raw_averaging_allowed"])
        self.assertTrue(bundle.summary["requires_labeled_phone_or_human_criterion"])
        self.assertFalse(bundle.summary["product_score_changed"])
        self.assertTrue(bundle.summary["row_field_canonical_log_posterior_is_utterance_sequence_level"])
        self.assertFalse(bundle.summary["row_field_canonical_log_posterior_is_phone_local"])
        self.assertTrue(bundle.summary["row_field_canonical_log_posterior_repeated_across_phone_rows"])
        self.assertAlmostEqual(
            bundle.rows[0].canonical_log_posterior,
            bundle.rows[1].canonical_log_posterior,
        )
        self.assertAlmostEqual(
            bundle.summary["utterance_canonical_sequence_log_posterior"],
            bundle.rows[0].canonical_log_posterior,
        )

    def test_model_revision_mismatch_is_not_silently_joined(self) -> None:
        enumerated, normalized = self._results()
        normalized = replace(normalized, revision="different-revision")
        bundle = build_phone_criterion_feature_bundle(enumerated, normalized)
        self.assertFalse(bundle.available)
        self.assertEqual(bundle.summary["reason"], "model_revision_mismatch")

    def test_unavailable_feature_family_propagates(self) -> None:
        enumerated, normalized = self._results()
        enumerated = replace(enumerated, available=False)
        bundle = build_phone_criterion_feature_bundle(enumerated, normalized)
        self.assertFalse(bundle.available)
        self.assertEqual(bundle.summary["reason"], "enumerated_features_unavailable")

    def test_shared_suffix_locality_aligns_from_the_end(self) -> None:
        enumerated, normalized = self._results()
        right = build_phone_criterion_feature_bundle(enumerated, normalized)
        left = replace(
            right,
            canonical_phones=["x", "b"],
            rows=[replace(right.rows[0], canonical_phone="x"), right.rows[1]],
        )
        diagnostic = compare_shared_suffix_locality(left, right)
        self.assertTrue(diagnostic["available"])
        self.assertEqual(diagnostic["shared_suffix_phone_count"], 1)
        self.assertEqual(diagnostic["shared_suffix_phones"], ["b"])
        self.assertEqual(diagnostic["left_prefix_phone_count"], 1)
        self.assertEqual(diagnostic["right_prefix_phone_count"], 1)
        self.assertAlmostEqual(diagnostic["normalized_graph_gop_abs_delta_max"], 0.0)
        self.assertAlmostEqual(diagnostic["occ_i_abs_delta_max"], 0.0)
        self.assertFalse(diagnostic["product_score_changed"])

    def test_shared_suffix_locality_rejects_cross_model_comparison(self) -> None:
        enumerated, normalized = self._results()
        bundle = build_phone_criterion_feature_bundle(enumerated, normalized)
        other = replace(bundle, model_id="another-model")
        diagnostic = compare_shared_suffix_locality(bundle, other)
        self.assertFalse(diagnostic["available"])
        self.assertEqual(diagnostic["reason"], "model_provenance_mismatch")


if __name__ == "__main__":
    unittest.main()
