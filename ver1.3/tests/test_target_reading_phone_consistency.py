from __future__ import annotations

import unittest

import pyopenjtalk

from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence


def _phones(text: str) -> list[str]:
    output = pyopenjtalk.g2p(text, kana=False, join=True)
    return output.split() if isinstance(output, str) else [str(phone) for phone in output]


class TargetReadingPhoneConsistencyTest(unittest.TestCase):
    def test_manual_reading_override_controls_ambiguous_meiou_phones(self) -> None:
        # Surface-only OpenJTalk analysis has been observed to interpret 明王 as
        # a personal-name reading. The reviewed override must instead drive the
        # phone frontend exactly.
        result = build_japanese_target_evidence(
            "明王",
            reading_override="みょうおう",
        )
        self.assertEqual(result.reading_source, "manual_override")
        self.assertIn("resolved_reading_drives_phone_sequence", result.warnings)
        self.assertEqual(result.phones, _phones(result.reading_kana))
        self.assertNotEqual(result.phones, _phones("明王"))

    def test_full_reviewed_reading_drives_phone_sequence_without_brittle_phone_count_assumption(self) -> None:
        surface = "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。"
        reading = "また、とうじのように、ごだいみょうおうとよばれる、しゅようなみょうおうのちゅうおうにはいされることもおおい。"
        result = build_japanese_target_evidence(surface, reading_override=reading)

        # The invariant we actually need is source provenance: every target
        # phone must come from the reviewed reading, regardless of how the
        # pinned frontend tokenizes a palatalized sequence internally.
        self.assertEqual(result.phones, _phones(result.reading_kana))
        self.assertEqual(result.frontend_input, result.reading_kana)
        self.assertNotEqual(result.phones, _phones(surface))

    def test_ordinary_automatic_target_keeps_surface_context_phone_contract(self) -> None:
        text = "それは、かっこです。"
        result = build_japanese_target_evidence(text)
        self.assertEqual(result.reading_source, "pyopenjtalk_g2p")
        self.assertIn("automatic_surface_context_drives_phone_sequence", result.warnings)
        self.assertEqual(result.phones, _phones(text))


if __name__ == "__main__":
    unittest.main()
