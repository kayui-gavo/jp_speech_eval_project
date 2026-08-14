import unittest

from jp_speech_eval.ume_jrf_analysis import (
    aggregate_criterion_units,
    build_validation_plan,
    criterion_coverage_report,
)
from jp_speech_eval.ume_jrf_research import build_ume_jrf_criterion_label


def _label(set_id: str, speaker: str, item: str, rater: str, value: int):
    return build_ume_jrf_criterion_label(
        set_id=set_id,
        speaker_id=speaker,
        item_id=item,
        rater_id=rater,
        raw_label=value,
    )


class UmeJrfAnalysisTest(unittest.TestCase):
    def test_ordinal_units_preserve_raw_labels_and_use_median_without_100(self) -> None:
        labels = [
            _label("D", "s1", "酸っぱい", "r1", 2),
            _label("D", "s1", "酸っぱい", "r2", 3),
            _label("D", "s1", "酸っぱい", "r3", 5),
        ]
        units = aggregate_criterion_units(labels)
        self.assertEqual(len(units), 1)
        unit = units[0]
        self.assertEqual(unit.raw_labels, [2, 3, 5])
        self.assertEqual(unit.ordinal_median, 3.0)
        self.assertIsNone(unit.normalized_100)
        self.assertFalse(unit.product_score_mapped)
        self.assertFalse(unit.commercial_product_use_allowed)

    def test_binary_tie_remains_none_instead_of_inventing_correctness(self) -> None:
        labels = [
            _label("B", "s1", "B001", "r1", 0),
            _label("B", "s1", "B001", "r2", 1),
        ]
        unit = aggregate_criterion_units(labels)[0]
        self.assertEqual(unit.binary_correct_fraction, 0.5)
        self.assertIsNone(unit.binary_majority_label)
        report = criterion_coverage_report([unit])
        self.assertEqual(report["binary_tied_majority_units"], 1)
        self.assertFalse(report["fit_allowed"])

    def test_duplicate_rater_for_same_speaker_item_fails_closed(self) -> None:
        labels = [
            _label("D", "s1", "酸っぱい", "r1", 2),
            _label("D", "s1", "酸っぱい", "r1", 3),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate rater"):
            aggregate_criterion_units(labels)

    def test_construct_mixing_is_rejected(self) -> None:
        labels = [
            _label("B", "s1", "B001", "r1", 1),
            _label("D", "s1", "酸っぱい", "r1", 3),
        ]
        with self.assertRaisesRegex(ValueError, "must not pool"):
            aggregate_criterion_units(labels)

    def test_validation_plan_has_both_speaker_and_item_holdout(self) -> None:
        labels = []
        for speaker, base in (("s1", 2), ("s2", 4)):
            for item in ("酸っぱい", "通信"):
                labels.extend(
                    [
                        _label("D", speaker, item, "r1", base),
                        _label("D", speaker, item, "r2", min(base + 1, 5)),
                    ]
                )
        units = aggregate_criterion_units(labels)
        plan = build_validation_plan(units)
        self.assertTrue(plan["available"])
        self.assertEqual(plan["coverage"]["speaker_count"], 2)
        self.assertEqual(plan["coverage"]["item_count"], 2)
        self.assertEqual(len(plan["speaker_folds"]), 2)
        self.assertEqual(len(plan["item_folds"]), 2)
        for fold in plan["speaker_folds"] + plan["item_folds"]:
            self.assertTrue(fold["train_indices"])
            self.assertTrue(fold["test_indices"])
            self.assertFalse(fold["product_score_mapping_allowed"])
        self.assertFalse(plan["commercial_product_training_allowed_without_separate_permission"])


if __name__ == "__main__":
    unittest.main()
