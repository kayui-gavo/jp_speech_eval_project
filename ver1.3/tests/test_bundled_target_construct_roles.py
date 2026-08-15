from __future__ import annotations

import unittest

from jp_speech_eval.japanese_phone_roles import LONG_VOWEL_ROLE, infer_phone_construct_roles
from jp_speech_eval.japanese_phoneme_gop import sanitize_canonical_phones
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence


class BundledTargetConstructRoleTest(unittest.TestCase):
    def _roles_for(self, text: str):
        target = build_japanese_target_evidence(text)
        phones, _dropped = sanitize_canonical_phones(target.phones)
        result = infer_phone_construct_roles(target.reading_kana, phones)
        self.assertTrue(result.available, result.summary)
        self.assertEqual(result.phones, phones)
        self.assertEqual(len(result.roles), len(phones))
        return target, result

    def test_ramen_target_marks_long_vowel_extension(self) -> None:
        target, result = self._roles_for("ラーメンをください。")
        self.assertIn("ー", target.reading_kana)
        self.assertGreaterEqual(result.summary["long_vowel_timing_phone_count"], 1)
        self.assertIn(LONG_VOWEL_ROLE, result.roles)

    def test_coffee_target_marks_long_vowel_extensions(self) -> None:
        target, result = self._roles_for("コーヒーをください。")
        self.assertIn("ー", target.reading_kana)
        self.assertGreaterEqual(result.summary["long_vowel_timing_phone_count"], 2)
        self.assertGreaterEqual(result.roles.count(LONG_VOWEL_ROLE), 2)


if __name__ == "__main__":
    unittest.main()
