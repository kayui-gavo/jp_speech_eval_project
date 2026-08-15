from __future__ import annotations

import unittest

from jp_speech_eval.phone_construct_roles import (
    LONG_VOWEL_TIMING_SUPPORT,
    ORDINARY_SEGMENTAL_CLARITY,
    SPECIAL_MORA_TIMING_SUPPORT,
    build_construct_role_view,
    clarity_primary_rows,
    construct_role_for_phone,
    long_vowel_support_rows,
    special_mora_support_rows,
)
from jp_speech_eval.phone_criterion_features import (
    SCHEMA as CRITERION_SCHEMA,
    PhoneCriterionFeatureBundle,
    PhoneCriterionFeatureRow,
)


def _row(index: int, phone: str, role: str | None = None) -> PhoneCriterionFeatureRow:
    if role is None:
        role = SPECIAL_MORA_TIMING_SUPPORT if phone in {"N", "cl"} else ORDINARY_SEGMENTAL_CLARITY
    ordinary = role == ORDINARY_SEGMENTAL_CLARITY
    return PhoneCriterionFeatureRow(
        phone_index=index,
        canonical_phone=phone,
        construct_role=role,
        ordinary_segmental_clarity_feature_applicable=ordinary,
        substitution_feature_applicable=ordinary,
        deletion_feature_applicable=True,
        normalized_occ_is_physical_duration=False,
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
    def test_role_policy_is_only_a_phone_token_fallback(self) -> None:
        self.assertEqual(construct_role_for_phone("k"), ORDINARY_SEGMENTAL_CLARITY)
        self.assertEqual(construct_role_for_phone("N"), SPECIAL_MORA_TIMING_SUPPORT)
        self.assertEqual(construct_role_for_phone("cl"), SPECIAL_MORA_TIMING_SUPPORT)
        # A repeated /o/ token alone cannot reveal whether it is a base vowel or
        # a long-vowel extension; explicit target metadata is required.
        self.assertEqual(construct_role_for_phone("o"), ORDINARY_SEGMENTAL_CLARITY)

    def test_special_mora_rows_drop_ordinary_clarity_primary_features(self) -> None:
        bundle = PhoneCriterionFeatureBundle(
            available=True,
            schema=CRITERION_SCHEMA,
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
        self.assertEqual(view["summary"]["long_vowel_timing_count"], 0)
        self.assertFalse(view["summary"]["special_mora_used_as_ordinary_clarity_primary"])
        self.assertFalse(view["summary"]["construct_role_rederived_from_phone_token"])

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

    def test_explicit_long_vowel_role_is_not_lost_by_phone_token_rederivation(self) -> None:
        bundle = PhoneCriterionFeatureBundle(
            available=True,
            schema=CRITERION_SCHEMA,
            model_id="synthetic",
            revision="test",
            canonical_phones=["k", "o", "o"],
            substitution_phone_inventory=["k", "o"],
            rows=[
                _row(0, "k"),
                _row(1, "o"),
                _row(2, "o", LONG_VOWEL_TIMING_SUPPORT),
            ],
            summary={},
            warnings=[],
        )
        view = build_construct_role_view(bundle)
        self.assertTrue(view["available"])
        self.assertEqual(view["summary"]["ordinary_segmental_count"], 2)
        self.assertEqual(view["summary"]["long_vowel_timing_count"], 1)
        self.assertFalse(view["summary"]["long_vowel_used_as_ordinary_clarity_primary"])
        self.assertEqual([row["phone_index"] for row in clarity_primary_rows(view)], [0, 1])
        long_rows = long_vowel_support_rows(view)
        self.assertEqual([row["phone_index"] for row in long_rows], [2])
        self.assertFalse(long_rows[0]["clarity_primary_eligible"])
        self.assertTrue(long_rows[0]["timing_context_required"])
        self.assertEqual(long_rows[0]["substitution_lprs"], {})
        self.assertEqual(long_rows[0]["deletion_lpr"], 2.0)

    def test_contradictory_role_applicability_fails_closed(self) -> None:
        broken = _row(0, "o", LONG_VOWEL_TIMING_SUPPORT)
        broken = PhoneCriterionFeatureRow(
            **{
                **broken.to_dict(),
                "ordinary_segmental_clarity_feature_applicable": True,
            }
        )
        bundle = PhoneCriterionFeatureBundle(
            available=True,
            schema=CRITERION_SCHEMA,
            model_id="synthetic",
            revision="test",
            canonical_phones=["o"],
            substitution_phone_inventory=["o"],
            rows=[broken],
            summary={},
            warnings=[],
        )
        view = build_construct_role_view(bundle)
        self.assertFalse(view["available"])
        self.assertEqual(view["reason"], "clarity_applicability_role_mismatch")

    def test_unavailable_criterion_bundle_fails_closed(self) -> None:
        bundle = PhoneCriterionFeatureBundle(
            available=False,
            schema=CRITERION_SCHEMA,
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
        self.assertEqual(long_vowel_support_rows(view), [])


if __name__ == "__main__":
    unittest.main()
