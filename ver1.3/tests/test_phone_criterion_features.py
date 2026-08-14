from dataclasses import replace
import unittest

import numpy as np

from jp_speech_eval.phone_criterion_features import build_phone_criterion_feature_bundle
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


if __name__ == "__main__":
    unittest.main()
