from __future__ import annotations

import unittest

from jp_speech_eval.transcript_sanity import check_asr_transcript_sanity


class TranscriptSanityTests(unittest.TestCase):
    def test_accepts_normal_japanese_sentence(self) -> None:
        result = check_asr_transcript_sanity("私は東京大学の一年生です")
        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "ok")

    def test_rejects_too_short_noise_like_text(self) -> None:
        result = check_asr_transcript_sanity("あ")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "too_short_for_pseudo_reference")

    def test_rejects_repetitive_shout(self) -> None:
        result = check_asr_transcript_sanity("ああああああああ")
        self.assertFalse(result.ok)
        self.assertIn(result.reason, {"repetitive_or_shouted_transcript", "low_information_repetition"})

    def test_rejects_non_japanese_text(self) -> None:
        result = check_asr_transcript_sanity("hello hello")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "not_enough_japanese_content")


if __name__ == "__main__":
    unittest.main()
