from dataclasses import replace
import unittest

import numpy as np

from jp_speech_eval.phone_criterion_features import (
    SCHEMA,
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

    def _special_mora_results(self):
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
        kwargs = dict(
            canonical_phones=["a", "N", "k"],
            vocab=vocab,
            blank_id=0,
            model_id="synthetic-phone-ctc",
            revision="0123456789abcdef",
        )
        enumerated = compute_enumerated_fgop_sf_sd_features(logits, **kwargs)
        normalized = compute_segmentation_free_norm_features(logits, **kwargs)
        return enumerated, normalized

    def test_bundle_joins_lpr_and_occ_without_score_mapping(self) -> None:
        enumerated, normalized = self._results()
        bundle = build_phone_criterion_feature_bundle(enumerated, normalized)
        self.assertTrue(bundle.available)
        self.assertEqual(bundle.schema, SCHEMA)
        self.assertFalse(bundle.score_mapped)
        self.assertFalse(bundle.product_calibrated)
        self.assertEqual(bundle.canonical_phones, ["a", "b"])
        self.assertEqual(len(bundle.rows), 2)
        self.assertEqual(bundle.rows[0].construct_role, "ordinary_segmental_clarity")
        self.assertTrue(bundle.rows[0].ordinary_segmental_clarity_feature_applicable)
        self.assertTrue(bundle.rows[0].substitution_feature_applicable)
        self.assertTrue(bundle.rows[0].deletion_feature_applicable)
        self.assertFalse(bundle.rows[0].normalized_occ_is_physical_duration)
        self.assertIn("a", bundle.rows[0].substitution_lprs)
        self.assertIn("b", bundle.rows[0].substitution_lprs)
        self.assertGreaterEqual(bundle.rows[0].occ_i, 0.0)
        self.assertFalse(bundle.summary["cross_model_raw_averaging_allowed"])
        self.assertTrue(bundle.summary["requires_labeled_phone_or_human_criterion"])
        self.assertFalse(bundle.summary["product_score_changed"])
        self.assertTrue(bundle.summary["construct_specific_feature_selection_required"])
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

    def test_special_mora_row_cannot_be_mistaken_for_ordinary_clarity(self) -> None:
        enumerated, normalized = self._special_mora_results()
        bundle = build_phone_criterion_feature_bundle(enumerated, normalized)
        self.assertTrue(bundle.available)
        n_row = bundle.rows[1]
        self.assertEqual(n_row.canonical_phone, "N")
        self.assertEqual(n_row.construct_role, "special_mora_timing")
        self.assertFalse(n_row.ordinary_segmental_clarity_feature_applicable)
        self.assertFalse(n_row.substitution_feature_applicable)
        self.assertTrue(n_row.deletion_feature_applicable)
        self.assertFalse(n_row.normalized_occ_is_physical_duration)
        self.assertEqual(set(n_row.substitution_lprs), {"N"})
        self.assertEqual(bundle.summary["special_mora_row_count"], 1)
        self.assertEqual(bundle.summary["ordinary_segmental_row_count"], 2)
        self.assertFalse(bundle.summary["special_mora_substitution_feature_applicable"])
        self.assertTrue(bundle.summary["special_mora_deletion_feature_applicable"])
        self.assertFalse(bundle.summary["special_mora_occ_i_is_physical_duration"])
        self.assertNotIn("N", bundle.substitution_phone_inventory)

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
        self.assertEqual(diagnostic["shared_suffix_construct_roles"], ["ordinary_segmental_clarity"])
        self.assertEqual(diagnostic["left_prefix_phone_count"], 1)
        self.assertEqual(diagnostic["right_prefix_phone_count"], 1)
        self.assertAlmostEqual(diagnostic["normalized_graph_gop_abs_delta_max"], 0.0)
        self.assertAlmostEqual(diagnostic["occ_i_abs_delta_max"], 0.0)
        self.assertFalse(diagnostic["product_score_changed"])

    def test_shared_suffix_locality_rejects_construct_role_mismatch(self) -> None:
        enumerated, normalized = self._results()
        right = build_phone_criterion_feature_bundle(enumerated, normalized)
        left = replace(
            right,
            rows=[
                right.rows[0],
                replace(right.rows[1], construct_role="special_mora_timing"),
            ],
        )
        diagnostic = compare_shared_suffix_locality(left, right)
        self.assertFalse(diagnostic["available"])
        self.assertEqual(diagnostic["reason"], "shared_suffix_construct_role_mismatch")

    def test_shared_suffix_locality_rejects_cross_model_comparison(self) -> None:
        enumerated, normalized = self._results()
        bundle = build_phone_criterion_feature_bundle(enumerated, normalized)
        other = replace(bundle, model_id="another-model")
        diagnostic = compare_shared_suffix_locality(bundle, other)
        self.assertFalse(diagnostic["available"])
        self.assertEqual(diagnostic["reason"], "model_provenance_mismatch")


if __name__ == "__main__":
    unittest.main()
