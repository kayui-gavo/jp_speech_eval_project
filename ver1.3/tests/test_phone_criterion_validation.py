from __future__ import annotations

import unittest

from jp_speech_eval.phone_criterion_validation import (
    PhoneCriterionExample,
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
                        features={"margin": 2.0 + 0.1 * speaker_index, "entropy": 0.2},
                    ),
                    PhoneCriterionExample(
                        sample_id=f"{speaker}-bad-b",
                        speaker_id=speaker,
                        canonical_phone="b",
                        is_expert_correct=False,
                        features={"margin": -2.0 - 0.1 * speaker_index, "entropy": 0.8},
                    ),
                    PhoneCriterionExample(
                        sample_id=f"{speaker}-good-k",
                        speaker_id=speaker,
                        canonical_phone="k",
                        is_expert_correct=True,
                        features={"margin": 1.5 + 0.1 * speaker_index, "entropy": 0.3},
                    ),
                    PhoneCriterionExample(
                        sample_id=f"{speaker}-bad-k",
                        speaker_id=speaker,
                        canonical_phone="k",
                        is_expert_correct=False,
                        features={"margin": -1.5 - 0.1 * speaker_index, "entropy": 0.7},
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

    def test_missing_feature_fails_closed(self) -> None:
        rows = self._examples()
        with self.assertRaisesRegex(ValueError, "missing criterion features"):
            evaluate_phone_criterion_binary(rows, feature_names=["does_not_exist"])

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
