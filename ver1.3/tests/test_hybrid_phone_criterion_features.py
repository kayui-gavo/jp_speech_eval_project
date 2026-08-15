from __future__ import annotations

from dataclasses import replace
import unittest

from jp_speech_eval.hybrid_phone_criterion_features import build_hybrid_phone_criterion_bundle
from jp_speech_eval.phone_criterion_features import (
    SCHEMA as CRITERION_SCHEMA,
    PhoneCriterionFeatureBundle,
    PhoneCriterionFeatureRow,
)
from jp_speech_eval.phoneme_gop import PhoneGopEvidence, PhoneGopResult


class HybridPhoneCriterionFeatureTest(unittest.TestCase):
    def _frame(self, phone: str = "b") -> PhoneGopResult:
        row = PhoneGopEvidence(
            phone_index=0,
            canonical_phone=phone,
            token_id=1,
            start_frame=2,
            end_frame=3,
            frame_count=1,
            start_sec=0.04,
            end_sec=0.06,
            duration_sec=0.02,
            target_mean_logit=4.0,
            target_max_logit=4.0,
            target_mean_logprob=-0.2,
            best_competitor_phone="p",
            best_competitor_token_id=2,
            best_competitor_mean_logit=1.0,
            best_competitor_max_logit=1.0,
            best_competitor_mean_logprob=-3.2,
            mean_logit_margin=3.0,
            max_logit_margin=3.0,
            posterior_gop_margin=3.0,
            mean_entropy=0.2,
            path_support_mean_logprob=-0.2,
            best_competitor_max_phone="p",
            best_competitor_max_token_id=2,
            ctc_support_duration_sec=0.02,
        )
        return PhoneGopResult(
            available=True,
            backend="synthetic",
            model_id="model@abc123456789",
            method="ctc_viterbi_phone_evidence_v1",
            canonical_phones=[phone],
            evidence=[row],
            summary={},
            warnings=[],
        )

    def _criterion(self, phone: str = "b") -> PhoneCriterionFeatureBundle:
        special = phone in {"N", "cl"}
        row = PhoneCriterionFeatureRow(
            phone_index=0,
            canonical_phone=phone,
            construct_role=("special_mora_timing" if special else "ordinary_segmental_clarity"),
            ordinary_segmental_clarity_feature_applicable=not special,
            substitution_feature_applicable=not special,
            deletion_feature_applicable=True,
            normalized_occ_is_physical_duration=False,
            canonical_log_posterior=-2.0,
            canonical_log_posterior_per_frame=-0.2,
            deletion_lpr=1.5,
            substitution_lprs=({phone: 0.0} if special else {"p": 2.2}),
            enumerated_gop_sf_sd=-0.4,
            normalized_graph_gop_sf_sd=-0.3,
            occ_i=1.1,
            best_noncanonical_type=("deletion" if special else "substitution"),
            best_noncanonical_phone=(None if special else "p"),
            best_noncanonical_lpr=2.2,
        )
        return PhoneCriterionFeatureBundle(
            available=True,
            schema=CRITERION_SCHEMA,
            model_id="model",
            revision="abc123456789",
            canonical_phones=[phone],
            substitution_phone_inventory=["b", "p"],
            rows=[row],
            summary={},
            warnings=[],
        )

    def test_hybrid_bundle_keeps_feature_families_separate(self) -> None:
        bundle = build_hybrid_phone_criterion_bundle(self._frame(), self._criterion())
        self.assertTrue(bundle.available)
        self.assertFalse(bundle.score_mapped)
        self.assertFalse(bundle.product_calibrated)
        self.assertTrue(bundle.summary["alignment_free_construct_metadata_verified"])
        self.assertEqual(bundle.rows[0].construct_role, "ordinary_segmental_clarity")
        self.assertTrue(bundle.rows[0].frame_local_ordinary_clarity_feature_applicable)
        self.assertTrue(bundle.rows[0].alignment_free_substitution_feature_applicable)
        self.assertTrue(bundle.rows[0].alignment_free_deletion_feature_applicable)
        self.assertEqual(bundle.rows[0].frame_local_max_logit_margin, 3.0)
        self.assertEqual(bundle.rows[0].alignment_free_substitution_lprs["p"], 2.2)
        self.assertFalse(bundle.summary["frame_local_support_is_physical_phone_boundary"])
        self.assertFalse(bundle.summary["cross_model_raw_averaging_allowed"])
        self.assertTrue(bundle.summary["feature_family_selection_requires_japanese_l2_criterion"])

    def test_special_mora_row_is_not_silently_treated_as_ordinary_clarity(self) -> None:
        bundle = build_hybrid_phone_criterion_bundle(self._frame("N"), self._criterion("N"))
        self.assertTrue(bundle.available)
        row = bundle.rows[0]
        self.assertEqual(row.construct_role, "special_mora_timing")
        self.assertFalse(row.frame_local_ordinary_clarity_feature_applicable)
        self.assertFalse(row.alignment_free_substitution_feature_applicable)
        self.assertTrue(row.alignment_free_deletion_feature_applicable)
        self.assertEqual(row.frame_local_mean_logit_margin, 3.0)
        self.assertEqual(row.alignment_free_substitution_lprs, {"N": 0.0})
        self.assertEqual(bundle.summary["special_mora_row_count"], 1)
        self.assertEqual(bundle.summary["ordinary_segmental_row_count"], 0)
        self.assertFalse(bundle.summary["special_mora_frame_local_raw_values_are_ordinary_clarity_features"])
        self.assertTrue(bundle.summary["special_mora_requires_construct_specific_feature_selection"])

    def test_construct_role_mismatch_fails_closed(self) -> None:
        criterion = self._criterion("N")
        broken = replace(
            criterion,
            rows=[replace(criterion.rows[0], construct_role="ordinary_segmental_clarity")],
        )
        bundle = build_hybrid_phone_criterion_bundle(self._frame("N"), broken)
        self.assertFalse(bundle.available)
        self.assertEqual(bundle.summary["reason"], "alignment_free_construct_role_mismatch")

    def test_substitution_applicability_mismatch_fails_closed(self) -> None:
        criterion = self._criterion("N")
        broken = replace(
            criterion,
            rows=[replace(criterion.rows[0], substitution_feature_applicable=True)],
        )
        bundle = build_hybrid_phone_criterion_bundle(self._frame("N"), broken)
        self.assertFalse(bundle.available)
        self.assertEqual(bundle.summary["reason"], "alignment_free_substitution_applicability_mismatch")

    def test_occ_duration_semantics_mismatch_fails_closed(self) -> None:
        criterion = self._criterion("b")
        broken = replace(
            criterion,
            rows=[replace(criterion.rows[0], normalized_occ_is_physical_duration=True)],
        )
        bundle = build_hybrid_phone_criterion_bundle(self._frame("b"), broken)
        self.assertFalse(bundle.available)
        self.assertEqual(bundle.summary["reason"], "alignment_free_occ_duration_semantics_mismatch")

    def test_model_provenance_mismatch_fails_closed(self) -> None:
        frame = replace(self._frame(), model_id="other@abc123456789")
        bundle = build_hybrid_phone_criterion_bundle(frame, self._criterion())
        self.assertFalse(bundle.available)
        self.assertEqual(bundle.summary["reason"], "model_provenance_mismatch")

    def test_phone_sequence_mismatch_fails_closed(self) -> None:
        frame = replace(self._frame(), canonical_phones=["p"])
        bundle = build_hybrid_phone_criterion_bundle(frame, self._criterion())
        self.assertFalse(bundle.available)
        self.assertEqual(bundle.summary["reason"], "canonical_phone_sequence_mismatch")


if __name__ == "__main__":
    unittest.main()
