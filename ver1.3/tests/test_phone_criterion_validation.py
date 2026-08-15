from __future__ import annotations

import unittest

from jp_speech_eval.phone_criterion_validation import (
    PhoneCriterionExample,
    compare_phone_criterion_feature_families,
    evaluate_phone_criterion_binary,
)


class PhoneCriterionValidationTest(unittest.TestCase):
    def _examples(self):
        rows = []
        # Four speakers; both classes occur in every speaker so the group split
        # has a well-defined synthetic sanity case.
        for speaker_index in range(4):
            speaker = f"spk{speaker_index}"
            rows.extend(
                [
                    PhoneCriterionExample(
                        sample_id=f"{speaker}-good-b",
                        speaker_id=speaker,
                        canonical_phone="b",
                        is_expert_correct=True,
                        features={
                            "margin": 2.0 + 0.1 * speaker_index,
                            "entropy": 0.2,
                            "noise": 0.1 * ((speaker_index % 2) * 2 - 1),
                        },
                    ),
                    PhoneCriterionExample(
                        sample_id=f"{speaker}-bad-b",
                        speaker_id=speaker,
                        canonical_phone="b",
                        is_expert_correct=False,
                        features={
                            "margin": -2.0 - 0.1 * speaker_index,
                            "entropy": 0.8,
                            "noise": -0.1 * ((speaker_index % 2) * 2 - 1),
                        },
                    ),
                    PhoneCriterionExample(
                        sample_id=f"{speaker}-good-k",
                        speaker_id=speaker,
                        canonical_phone="k",
                        is_expert_correct=True,
                        features={
                            "margin": 1.5 + 0.1 * speaker_index,
                            "entropy": 0.3,
                            "noise": 0.05 * ((speaker_index % 2) * 2 - 1),
                        },
                    ),
                    PhoneCriterionExample(
                        sample_id=f"{speaker}-bad-k",
                        speaker_id=speaker,
                        canonical_phone="k",
                        is_expert_correct=False,
                        features={
                            "margin": -1.5 - 0.1 * speaker_index,
                            "entropy": 0.7,
                            "noise": -0.05 * ((speaker_index % 2) * 2 - 1),
                        },
                    ),
                ]
            )
        return rows

    def test_speaker_held_out_pipeline_detects_separable_synthetic_errors(self) -> None:
        result = evaluate_phone_criterion_binary(
            self._examples(),
            feature_names=["margin", "entropy"],
            n_splits=4,
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["split_policy"], "StratifiedGroupKFold_by_speaker")
        self.assertGreater(result["metrics"]["roc_auc_error_detection"], 0.95)
        self.assertGreater(result["metrics"]["balanced_accuracy_at_diagnostic_0_5"], 0.90)
        for fold in result["folds"]:
            self.assertFalse(fold["speaker_overlap"])
            self.assertTrue(set(fold["train_speakers"]).isdisjoint(fold["test_speakers"]))
        self.assertFalse(result["summary"]["diagnostic_0_5_is_product_threshold"])
        self.assertFalse(result["summary"]["score_mapped"])
        self.assertFalse(result["summary"]["product_calibrated"])
        self.assertFalse(result["summary"]["item_or_target_phone_held_out_stress_test_done"])

    def test_feature_family_ablation_reuses_identical_speaker_folds(self) -> None:
        result = compare_phone_criterion_feature_families(
            self._examples(),
            feature_families={
                "logit_margin": ["margin"],
                "uncertainty": ["entropy"],
                "weak_control": ["noise"],
                "hybrid": ["margin", "entropy", "noise"],
            },
            combined_family_name="hybrid",
            n_splits=4,
            random_state=17,
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["family_count"], 4)
        self.assertTrue(result["summary"]["same_speaker_folds_used_for_all_families"])
        self.assertFalse(result["summary"]["feature_family_selected_for_product"])
        self.assertFalse(result["summary"]["statistical_significance_tested"])
        self.assertFalse(result["summary"]["confidence_intervals_estimated"])
        self.assertFalse(result["summary"]["item_or_target_phone_held_out_stress_test_done"])
        comparison = result["combined_family_comparison"]
        self.assertEqual(comparison["combined_family"], "hybrid")
        self.assertFalse(comparison["positive_delta_proves_generalization"])
        self.assertTrue(comparison["requires_external_or_nested_confirmation"])

        signatures = []
        for family in result["family_results"].values():
            signatures.append(
                [
                    (tuple(fold["train_speakers"]), tuple(fold["test_speakers"]))
                    for fold in family["folds"]
                ]
            )
        self.assertTrue(all(signature == signatures[0] for signature in signatures[1:]))

    def test_feature_family_comparison_requires_multiple_families(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two families"):
            compare_phone_criterion_feature_families(
                self._examples(),
                feature_families={"only": ["margin"]},
            )

    def test_combined_family_name_must_exist(self) -> None:
        with self.assertRaisesRegex(ValueError, "combined_family_name"):
            compare_phone_criterion_feature_families(
                self._examples(),
                feature_families={"margin": ["margin"], "entropy": ["entropy"]},
                combined_family_name="hybrid",
            )

    def test_missing_feature_fails_closed(self) -> None:
        rows = self._examples()
        with self.assertRaisesRegex(ValueError, "missing criterion features"):
            evaluate_phone_criterion_binary(rows, feature_names=["does_not_exist"])

    def test_duplicate_feature_name_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_phone_criterion_binary(
                self._examples(),
                feature_names=["margin", "margin"],
            )

    def test_single_speaker_cannot_be_used_for_criterion_validation(self) -> None:
        rows = [
            PhoneCriterionExample("a", "one", "b", True, {"x": 1.0}),
            PhoneCriterionExample("b", "one", "b", False, {"x": -1.0}),
            PhoneCriterionExample("c", "one", "k", True, {"x": 1.2}),
            PhoneCriterionExample("d", "one", "k", False, {"x": -1.2}),
        ]
        with self.assertRaisesRegex(ValueError, "at least two speakers"):
            evaluate_phone_criterion_binary(rows, feature_names=["x"])

    def test_one_class_cannot_be_used_for_criterion_validation(self) -> None:
        rows = [
            PhoneCriterionExample("a", "s1", "b", True, {"x": 1.0}),
            PhoneCriterionExample("b", "s1", "k", True, {"x": 1.2}),
            PhoneCriterionExample("c", "s2", "b", True, {"x": 0.9}),
            PhoneCriterionExample("d", "s2", "k", True, {"x": 1.1}),
        ]
        with self.assertRaisesRegex(ValueError, "both correct and incorrect"):
            evaluate_phone_criterion_binary(rows, feature_names=["x"])


if __name__ == "__main__":
    unittest.main()
