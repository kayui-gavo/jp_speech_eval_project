from __future__ import annotations

import unittest

from jp_speech_eval.mora_evidence import build_mora_evidence
from jp_speech_eval.phonology import classify_mora_sequence, mora_vowel
from jp_speech_eval.scoring import score_pronunciation_rhythm
from jp_speech_eval.text_frontend import split_mora


class PhonologyTest(unittest.TestCase):
    def test_mora_vowel_handles_compound_mora(self) -> None:
        self.assertEqual(mora_vowel("キョ"), "o")
        self.assertEqual(mora_vowel("リュ"), "u")
        self.assertEqual(mora_vowel("シェ"), "e")

    def test_classifies_strong_and_weak_duration_roles(self) -> None:
        rows = classify_mora_sequence(split_mora("ラーメンヲクダサイ"))
        by_mora = [(row.mora, row.mora_type, row.strength) for row in rows]
        self.assertIn(("ー", "explicit_long_vowel", "strong"), by_mora)
        self.assertIn(("ン", "nasal", "strong"), by_mora)

        arigatou = classify_mora_sequence(split_mora("アリガトウ"))
        self.assertEqual(arigatou[-1].mora, "ウ")
        self.assertEqual(arigatou[-1].mora_type, "vowel_lengthening_candidate")
        self.assertEqual(arigatou[-1].strength, "weak")

        sensei = classify_mora_sequence(split_mora("センセイ"))
        self.assertEqual(sensei[-1].mora_type, "vowel_lengthening_candidate")

    def test_weak_long_vowel_candidate_is_light_penalty(self) -> None:
        moras = split_mora("アリガトウ")
        boundaries = [(0.00, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 0.86)]
        score, feedback, details = score_pronunciation_rhythm(moras, boundaries)
        self.assertLess(score, 100)
        self.assertTrue(any("长音感" in item for item in feedback))
        self.assertEqual(details["special_mora_diagnostics"][0]["strength"], "weak")
        self.assertLess(details["special_mora_diagnostics"][0]["penalty"], 10.0)

    def test_mora_evidence_logs_weak_and_strong_counts(self) -> None:
        moras = split_mora("ラーメンアリガトウ")
        boundaries = [(i * 0.12, (i + 1) * 0.12) for i in range(len(moras))]
        rows, summary = build_mora_evidence(
            moras=moras,
            boundaries=boundaries,
            f0_times=[],
            f0_hz=[],
            y_speech=[0.1] * 16000,
            sr=16000,
        )
        self.assertGreaterEqual(summary["strong_special_mora_count"], 2)
        self.assertGreaterEqual(summary["weak_special_mora_count"], 1)
        self.assertTrue(any(row["special_strength"] == "weak" for row in rows))


if __name__ == "__main__":
    unittest.main()
