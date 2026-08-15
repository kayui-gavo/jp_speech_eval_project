from __future__ import annotations

import unittest

from jp_speech_eval.japanese_phone_roles import (
    LONG_VOWEL_ROLE,
    ORDINARY_ROLE,
    SPECIAL_MORA_ROLE,
    infer_phone_construct_roles,
)


class JapanesePhoneRoleInferenceTest(unittest.TestCase):
    def test_gakko_long_vowel_and_sokuon_are_separated(self) -> None:
        result = infer_phone_construct_roles(
            "ガッコー",
            ["g", "a", "cl", "k", "o", "o"],
        )
        self.assertTrue(result.available)
        self.assertEqual(result.moras, ["ガ", "ッ", "コ", "ー"])
        self.assertEqual(
            result.roles,
            [
                ORDINARY_ROLE,
                ORDINARY_ROLE,
                SPECIAL_MORA_ROLE,
                ORDINARY_ROLE,
                ORDINARY_ROLE,
                LONG_VOWEL_ROLE,
            ],
        )
        self.assertEqual(result.phone_to_mora_index, [0, 0, 1, 2, 2, 3])
        self.assertEqual(result.summary["long_vowel_timing_phone_count"], 1)
        self.assertEqual(result.summary["special_mora_phone_count"], 1)

    def test_byouin_long_vowel_and_n_are_separated(self) -> None:
        result = infer_phone_construct_roles(
            "ビョーイン",
            ["by", "o", "o", "i", "N"],
        )
        self.assertTrue(result.available)
        self.assertEqual(result.moras, ["ビョ", "ー", "イ", "ン"])
        self.assertEqual(
            result.roles,
            [ORDINARY_ROLE, ORDINARY_ROLE, LONG_VOWEL_ROLE, ORDINARY_ROLE, SPECIAL_MORA_ROLE],
        )
        self.assertFalse(result.summary["orthographic_u_i_long_vowel_guessing_used"])

    def test_high_vowel_allophone_label_is_structurally_a_vowel(self) -> None:
        result = infer_phone_construct_roles("スシ", ["s", "U", "sh", "i"])
        self.assertTrue(result.available)
        self.assertEqual(result.roles, [ORDINARY_ROLE] * 4)

    def test_plain_u_is_not_guessed_as_long_vowel_without_long_mark(self) -> None:
        # A reviewed/manual reading such as ヨウ is ambiguous at the construct
        # level without additional lexical annotation. The conservative
        # classifier does not silently reinterpret ウ as a long-vowel extension.
        result = infer_phone_construct_roles("ヨウ", ["y", "o", "u"])
        self.assertTrue(result.available)
        self.assertEqual(result.roles, [ORDINARY_ROLE, ORDINARY_ROLE, ORDINARY_ROLE])
        self.assertEqual(result.summary["long_vowel_timing_phone_count"], 0)

    def test_alignment_mismatch_fails_closed(self) -> None:
        result = infer_phone_construct_roles("ガッコー", ["g", "a", "k", "o", "o"])
        self.assertFalse(result.available)
        self.assertIn("expected_cl", result.summary["reason"])
        self.assertEqual(result.roles, [])
        self.assertFalse(result.summary["score_mapped"])


if __name__ == "__main__":
    unittest.main()
