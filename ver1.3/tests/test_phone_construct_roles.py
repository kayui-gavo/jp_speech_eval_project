from __future__ import annotations

import unittest

from jp_speech_eval.phone_construct_roles import (
    ORDINARY_SEGMENTAL_CLARITY,
    SPECIAL_MORA_TIMING_SUPPORT,
    build_construct_role_view,
    clarity_primary_rows,
    construct_role_for_phone,
    special_mora_support_rows,
)
from jp_speech_eval.phone_criterion_features import (
    PhoneCriterionFeatureBundle,
    PhoneCriterionFeatureRow,
)


def _row(index: int, phone: str) -> PhoneCriterionFeatureRow:
    return PhoneCriterionFeatureRow(
        phone_index=index,
        canonical_phone=phone,
        canonical_log_posterior=-10.0,
        canonical_log_posterior_per_frame=-0.1,
        deletion_lpr=2.0,
        substitution_lprs={"x": 1.0},
        enumerated_gop_sf_sd=-0.3,
        normalized_graph_gop_sf_sd=-0.2,
        occ_i=1.1,
        best_noncanonical_type="substitution",
        best_noncanonical_phone="x",
        best_noncanonical_lpr=1.0,
    )


class PhoneConstructRoleTest(unittest.TestCase):
    def test_role_policy(self) -> None:
        self.assertEqual(construct_role_for_phone("k"), ORDINARY_SEGMENTAL_CLARITY)
        self.assertEqual(construct_role_for_phone("N"), SPECIAL_MORA_TIMING_SUPPORT)
        self.assertEqual(construct_role_for_phone("cl"), SPECIAL_MORA_TIMING_SUPPORT)

    def test_special_mora_rows_drop_ordinary_clarity_primary_features(self) -> None:
        bundle = PhoneCriterionFeatureBundle(
            available=True,
            schema="phone_criterion_feature_bundle_v1",
            model_id="synthetic",
            revision="test",
            canonical_phones=["k", "N", "cl"],
            substitution_phone_inventory=["k", "N", "cl", "g"],
            rows=[_row(0, "k"), _row(1, "N"), _row(2, "cl")],
            summary={},
            warnings=[],
        )
        view = build_construct_role_view(bundle)
        self.assertTrue(view["available"])
        self.assertEqual(view["summary"]["ordinary_segmental_count"], 1)
        self.assertEqual(view["summary"]["special_mora_count"], 2)
        self.assertFalse(view["summary"]["special_mora_used_as_ordinary_clarity_primary"])

        ordinary = clarity_primary_rows(view)
        special = special_mora_support_rows(view)
        self.assertEqual([row["canonical_phone"] for row in ordinary], ["k"])
        self.assertEqual([row["canonical_phone"] for row in special], ["N", "cl"])
        for row in special:
            self.assertFalse(row["clarity_primary_eligible"])
            self.assertTrue(row["timing_context_required"])
            self.assertEqual(row["substitution_lprs"], {})
            self.assertIsNone(row["enumerated_gop_sf_sd"])
            self.assertIsNone(row["normalized_graph_gop_sf_sd"])
            self.assertIsNone(row["occ_i"])
            self.assertEqual(row["deletion_lpr"], 2.0)

    def test_unavailable_criterion_bundle_fails_closed(self) -> None:
        bundle = PhoneCriterionFeatureBundle(
            available=False,
            schema="phone_criterion_feature_bundle_v1",
            model_id="synthetic",
            revision="test",
            canonical_phones=[],
            substitution_phone_inventory=[],
            rows=[],
            summary={"reason": "no_data"},
            warnings=["no_data"],
        )
        view = build_construct_role_view(bundle)
        self.assertFalse(view["available"])
        self.assertEqual(clarity_primary_rows(view), [])
        self.assertEqual(special_mora_support_rows(view), [])


if __name__ == "__main__":
    unittest.main()
