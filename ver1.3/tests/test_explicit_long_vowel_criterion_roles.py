from __future__ import annotations

import unittest

import numpy as np

from jp_speech_eval.japanese_phone_roles import LONG_VOWEL_ROLE, ORDINARY_ROLE
from jp_speech_eval.phone_criterion_features import build_phone_criterion_feature_bundle
from jp_speech_eval.segmentation_free_gop import compute_enumerated_fgop_sf_sd_features
from jp_speech_eval.segmentation_free_gop_norm import compute_segmentation_free_norm_features


class ExplicitLongVowelCriterionRoleTest(unittest.TestCase):
    def _results(self):
        # Synthetic CTC path for two /o/ tokens. The acoustics are irrelevant to
        # construct identity: only target-side metadata can say that the second
        # repeated vowel is the extension mora of a long vowel.
        vocab = {"<blank>": 0, "o": 1, "a": 2}
        logits = np.asarray(
            [
                [7.0, 0.0, 0.0],
                [0.0, 7.0, 0.0],
                [7.0, 0.0, 0.0],
                [0.0, 7.0, 0.0],
                [7.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        kwargs = dict(
            canonical_phones=["o", "o"],
            vocab=vocab,
            blank_id=0,
            model_id="synthetic-phone-ctc",
            revision="0123456789abcdef",
        )
        return (
            compute_enumerated_fgop_sf_sd_features(logits, **kwargs),
            compute_segmentation_free_norm_features(logits, **kwargs),
        )

    def test_explicit_long_vowel_extension_is_not_ordinary_clarity(self) -> None:
        enumerated, normalized = self._results()
        bundle = build_phone_criterion_feature_bundle(
            enumerated,
            normalized,
            construct_roles=[ORDINARY_ROLE, LONG_VOWEL_ROLE],
        )
        self.assertTrue(bundle.available)
        self.assertEqual(bundle.rows[0].construct_role, ORDINARY_ROLE)
        self.assertTrue(bundle.rows[0].ordinary_segmental_clarity_feature_applicable)
        self.assertTrue(bundle.rows[0].substitution_feature_applicable)

        extension = bundle.rows[1]
        self.assertEqual(extension.canonical_phone, "o")
        self.assertEqual(extension.construct_role, LONG_VOWEL_ROLE)
        self.assertFalse(extension.ordinary_segmental_clarity_feature_applicable)
        self.assertFalse(extension.substitution_feature_applicable)
        self.assertTrue(extension.deletion_feature_applicable)
        self.assertFalse(extension.normalized_occ_is_physical_duration)
        self.assertEqual(bundle.summary["ordinary_segmental_row_count"], 1)
        self.assertEqual(bundle.summary["long_vowel_timing_row_count"], 1)
        self.assertEqual(bundle.summary["timing_construct_row_count"], 1)
        self.assertEqual(bundle.summary["construct_role_source"], "explicit_target_metadata")
        self.assertTrue(bundle.summary["explicit_construct_roles_used"])
        self.assertFalse(bundle.summary["long_vowel_substitution_feature_applicable"])
        self.assertTrue(bundle.summary["long_vowel_deletion_feature_applicable"])
        self.assertFalse(bundle.summary["long_vowel_occ_i_is_physical_duration"])

    def test_without_explicit_metadata_same_phone_tokens_remain_ordinary(self) -> None:
        enumerated, normalized = self._results()
        bundle = build_phone_criterion_feature_bundle(enumerated, normalized)
        self.assertTrue(bundle.available)
        self.assertEqual([row.construct_role for row in bundle.rows], [ORDINARY_ROLE, ORDINARY_ROLE])
        self.assertEqual(bundle.summary["construct_role_source"], "safe_phone_token_default")
        self.assertFalse(bundle.summary["explicit_construct_roles_used"])
        self.assertEqual(bundle.summary["long_vowel_timing_row_count"], 0)

    def test_explicit_role_count_mismatch_fails_closed(self) -> None:
        enumerated, normalized = self._results()
        bundle = build_phone_criterion_feature_bundle(
            enumerated,
            normalized,
            construct_roles=[ORDINARY_ROLE],
        )
        self.assertFalse(bundle.available)
        self.assertEqual(bundle.summary["reason"], "construct_role_count_mismatch")


if __name__ == "__main__":
    unittest.main()
