from __future__ import annotations

from dataclasses import replace
import unittest

from jp_speech_eval.hybrid_phone_criterion_features import build_hybrid_phone_criterion_bundle
from jp_speech_eval.japanese_phone_roles import LONG_VOWEL_ROLE
from jp_speech_eval.phone_criterion_features import SCHEMA, PhoneCriterionFeatureBundle, PhoneCriterionFeatureRow
from jp_speech_eval.phoneme_gop import PhoneGopEvidence, PhoneGopResult


def frame_result() -> PhoneGopResult:
    row = PhoneGopEvidence(
        phone_index=0, canonical_phone="o", token_id=1,
        start_frame=2, end_frame=3, frame_count=1,
        start_sec=0.04, end_sec=0.06, duration_sec=0.02,
        target_mean_logit=4.0, target_max_logit=4.0, target_mean_logprob=-0.2,
        best_competitor_phone="a", best_competitor_token_id=2,
        best_competitor_mean_logit=1.0, best_competitor_max_logit=1.0,
        best_competitor_mean_logprob=-3.2, mean_logit_margin=3.0,
        max_logit_margin=3.0, posterior_gop_margin=3.0, mean_entropy=0.2,
        path_support_mean_logprob=-0.2, best_competitor_max_phone="a",
        best_competitor_max_token_id=2, ctc_support_duration_sec=0.02,
    )
    return PhoneGopResult(
        available=True, backend="synthetic", model_id="model@abc123456789",
        method="ctc_viterbi_phone_evidence_v1", canonical_phones=["o"],
        evidence=[row], summary={}, warnings=[],
    )


def criterion_bundle() -> PhoneCriterionFeatureBundle:
    row = PhoneCriterionFeatureRow(
        phone_index=0, canonical_phone="o", construct_role=LONG_VOWEL_ROLE,
        ordinary_segmental_clarity_feature_applicable=False,
        substitution_feature_applicable=False, deletion_feature_applicable=True,
        normalized_occ_is_physical_duration=False,
        canonical_log_posterior=-2.0, canonical_log_posterior_per_frame=-0.2,
        deletion_lpr=1.5, substitution_lprs={"a": 2.2},
        enumerated_gop_sf_sd=-0.4, normalized_graph_gop_sf_sd=-0.3, occ_i=1.1,
        best_noncanonical_type="deletion", best_noncanonical_phone=None,
        best_noncanonical_lpr=1.5,
    )
    return PhoneCriterionFeatureBundle(
        available=True, schema=SCHEMA, model_id="model", revision="abc123456789",
        canonical_phones=["o"], substitution_phone_inventory=["o", "a"],
        rows=[row], summary={"construct_role_source": "explicit_target_metadata"}, warnings=[],
    )


class HybridLongVowelRoleTest(unittest.TestCase):
    def test_long_vowel_extension_is_not_clarity_primary(self) -> None:
        bundle = build_hybrid_phone_criterion_bundle(frame_result(), criterion_bundle())
        self.assertTrue(bundle.available)
        self.assertEqual(bundle.rows[0].construct_role, LONG_VOWEL_ROLE)
        self.assertFalse(bundle.rows[0].frame_local_ordinary_clarity_feature_applicable)
        self.assertFalse(bundle.rows[0].alignment_free_substitution_feature_applicable)
        self.assertTrue(bundle.rows[0].alignment_free_deletion_feature_applicable)
        self.assertEqual(bundle.summary["long_vowel_timing_row_count"], 1)
        self.assertEqual(bundle.summary["ordinary_segmental_row_count"], 0)
        self.assertFalse(bundle.summary["construct_role_rederived_from_phone_token"])

    def test_contradictory_long_vowel_clarity_flag_is_rejected(self) -> None:
        criterion = criterion_bundle()
        broken = replace(
            criterion,
            rows=[replace(criterion.rows[0], ordinary_segmental_clarity_feature_applicable=True)],
        )
        bundle = build_hybrid_phone_criterion_bundle(frame_result(), broken)
        self.assertFalse(bundle.available)
        self.assertEqual(bundle.summary["reason"], "alignment_free_clarity_applicability_mismatch")


if __name__ == "__main__":
    unittest.main()
