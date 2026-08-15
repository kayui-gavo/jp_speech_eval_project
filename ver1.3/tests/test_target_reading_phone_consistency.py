from __future__ import annotations

import unittest

from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence


class TargetReadingPhoneConsistencyTest(unittest.TestCase):
    def test_manual_reading_override_controls_ambiguous_meiou_phones(self) -> None:
        # Surface-only OpenJTalk analysis has been observed to interpret 明王 as
        # a personal-name reading. The reviewed override must force みょうおう.
        result = build_japanese_target_evidence(
            "明王",
            reading_override="みょうおう",
        )
        self.assertEqual(result.reading_source, "manual_override")
        self.assertIn("resolved_reading_drives_phone_sequence", result.warnings)
        self.assertIn("my", result.phones)
        self.assertNotIn("r", result.phones)
        self.assertNotIn("k", result.phones)

    def test_resolved_reading_and_phone_sequence_are_internally_consistent(self) -> None:
        reading = "また、とうじのように、ごだいみょうおうとよばれる、しゅようなみょうおうのちゅうおうにはいされることもおおい。"
        result = build_japanese_target_evidence(
            "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。",
            reading_override=reading,
        )
        # Two 明王 occurrences should both contribute /my/ rather than an
        # independent kanji re-analysis.
        self.assertEqual(result.phones.count("my"), 2)
        self.assertGreaterEqual(result.phones.count("o"), 6)
        self.assertEqual(result.frontend_input, result.reading_kana)


if __name__ == "__main__":
    unittest.main()
