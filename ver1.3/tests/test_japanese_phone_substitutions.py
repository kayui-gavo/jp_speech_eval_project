from __future__ import annotations

import unittest

import numpy as np

from jp_speech_eval.japanese_phone_substitutions import (
    POLICY_NAME,
    canonical_logical_phone,
    restricted_substitution_phones,
    substitution_token_ids_by_position,
)
from jp_speech_eval.restricted_segmentation_free_gop import (
    compare_restricted_vs_unrestricted,
    compute_restricted_fgop_sf_sd_features,
)
from jp_speech_eval.segmentation_free_gop import compute_enumerated_fgop_sf_sd_features


class JapaneseRestrictedSubstitutionPolicyTest(unittest.TestCase):
    def test_high_vowel_allophone_labels_collapse(self) -> None:
        self.assertEqual(canonical_logical_phone("I"), "i")
        self.assertEqual(canonical_logical_phone("U"), "u")
        self.assertEqual(canonical_logical_phone("a"), "a")

    def test_restricted_s_candidates_are_phone_tokens_not_kana(self) -> None:
        result = restricted_substitution_phones(
            "s",
            ["PAD", "s", "z", "sh", "ts", "k", "N", "cl", "pau", "sil"],
        )
        self.assertEqual(result.policy, POLICY_NAME)
        self.assertFalse(result.fallback_used)
        self.assertEqual(set(result.candidates), {"s", "z", "sh", "ts"})
        self.assertNotIn("N", result.candidates)
        self.assertNotIn("cl", result.candidates)
        self.assertNotIn("pau", result.candidates)
        self.assertNotIn("sil", result.candidates)

    def test_special_mora_uses_canonical_plus_deletion_only(self) -> None:
        result = restricted_substitution_phones(
            "cl",
            ["PAD", "k", "s", "cl", "N"],
        )
        self.assertEqual(result.policy, "special_mora_canonical_plus_deletion_only")
        self.assertEqual(result.candidates, ("cl",))
        self.assertFalse(result.fallback_used)

    def test_unknown_ordinary_phone_fallback_is_explicit(self) -> None:
        result = restricted_substitution_phones(
            "q",
            ["PAD", "q", "k", "g", "s"],
        )
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.policy, "unrestricted_segmental_fallback")
        self.assertEqual(set(result.candidates), {"q", "k", "g", "s"})

    def test_position_mapping_preserves_provenance(self) -> None:
        vocab = {"PAD": 0, "s": 1, "z": 2, "sh": 3, "ts": 4, "k": 5, "cl": 6}
        mapping, provenance = substitution_token_ids_by_position(["s", "cl"], vocab)
        self.assertEqual(set(mapping[0]), {1, 2, 3, 4})
        self.assertEqual(mapping[1], [6])
        self.assertEqual(provenance[0]["policy"], POLICY_NAME)
        self.assertEqual(provenance[1]["policy"], "special_mora_canonical_plus_deletion_only")
        self.assertEqual(provenance[0]["candidate_count"], 4)


class RestrictedSegmentationFreeGopTest(unittest.TestCase):
    @staticmethod
    def _logits() -> tuple[np.ndarray, dict[str, int]]:
        vocab = {"PAD": 0, "s": 1, "z": 2, "sh": 3, "ts": 4, "k": 5, "g": 6, "ky": 7}
        # Strong CTC path for s,k with blanks between labels.
        logits = np.asarray(
            [
                [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 6.0, 1.0, 1.2, 1.5, 0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 1.0, 1.0],
                [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        return logits, vocab

    def test_restricted_result_is_available_and_auditable(self) -> None:
        logits, vocab = self._logits()
        result = compute_restricted_fgop_sf_sd_features(
            logits,
            ["s", "k"],
            vocab=vocab,
            blank_id=0,
            model_id="synthetic",
            revision="test",
        )
        self.assertTrue(result.available)
        self.assertFalse(result.score_mapped)
        self.assertFalse(result.product_calibrated)
        self.assertEqual(result.summary["search_space"], "position_specific_restricted_japanese_phonology")
        self.assertEqual(set(result.rows[0].candidate_phones), {"s", "z", "sh", "ts"})
        self.assertEqual(set(result.rows[1].candidate_phones), {"k", "g", "ky"})
        self.assertEqual(result.summary["fallback_position_count"], 0)

    def test_restricted_vs_unrestricted_is_comparison_not_fusion(self) -> None:
        logits, vocab = self._logits()
        restricted = compute_restricted_fgop_sf_sd_features(
            logits,
            ["s", "k"],
            vocab=vocab,
            blank_id=0,
        )
        unrestricted = compute_enumerated_fgop_sf_sd_features(
            logits,
            ["s", "k"],
            vocab=vocab,
            blank_id=0,
        )
        comparison = compare_restricted_vs_unrestricted(restricted, unrestricted)
        self.assertTrue(comparison["available"])
        self.assertTrue(comparison["raw_values_must_not_be_averaged"])
        self.assertFalse(comparison["product_score_changed"])
        self.assertEqual(len(comparison["rows"]), 2)


if __name__ == "__main__":
    unittest.main()
