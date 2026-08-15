from __future__ import annotations

import unittest

import pyopenjtalk

from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence


def _phones(text: str) -> list[str]:
    output = pyopenjtalk.g2p(text, kana=False, join=True)
    return output.split() if isinstance(output, str) else [str(phone) for phone in output]


class TargetReadingPhoneConsistencyTest(unittest.TestCase):
    def test_manual_reading_override_avoids_surface_kanji_analysis_but_is_still_g2p_derived(self) -> None:
        result = build_japanese_target_evidence(
            "明王",
            reading_override="みょうおう",
        )
        self.assertEqual(result.reading_source, "manual_override")
        self.assertIn("resolved_reading_drives_phone_sequence", result.warnings)
        self.assertNotIn("manual_phone_override_drives_phone_sequence", result.warnings)
        self.assertEqual(result.phones, _phones(result.reading_kana))
        self.assertNotEqual(result.phones, _phones("明王"))

    def test_explicit_phone_override_is_authoritative_over_kana_reanalysis(self) -> None:
        surface = "明王"
        reading = "みょうおう"
        reviewed = ["my", "o", "o", "o", "o"]
        result = build_japanese_target_evidence(
            surface,
            reading_override=reading,
            phones_override=reviewed,
        )
        self.assertEqual(result.phones, reviewed)
        self.assertIn("manual_phone_override_drives_phone_sequence", result.warnings)
        self.assertIn("phone_override_is_authoritative_over_text_frontend_g2p", result.warnings)

    def test_full_reviewed_anchor_can_override_context_dependent_kana_g2p(self) -> None:
        surface = "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。"
        reading = "また、とうじのように、ごだいみょうおうとよばれる、しゅようなみょうおうのちゅうおうにはいされることもおおい。"
        reviewed = ["m", "a", "t", "a", "my", "o", "o", "o", "o"]
        result = build_japanese_target_evidence(
            surface,
            reading_override=reading,
            phones_override=reviewed,
        )
        # Deliberately tiny synthetic reviewed sequence: the contract under test
        # is that explicit phones are not overwritten by either surface or kana
        # text analysis. Inventory validation happens later in backend preflight.
        self.assertEqual(result.phones, reviewed)
        self.assertEqual(result.frontend_input, result.reading_kana)

    def test_phone_override_rejects_control_pause_and_empty_tokens(self) -> None:
        for bad in (["a", "pau"], ["a", "sil"], ["a", "PAD"], ["a", ""], []):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    build_japanese_target_evidence(
                        "てすと",
                        reading_override="てすと",
                        phones_override=bad,
                    )

    def test_special_morae_are_valid_in_explicit_phone_override(self) -> None:
        result = build_japanese_target_evidence(
            "みんな、かっこ。",
            reading_override="みんな、かっこ。",
            phones_override=["m", "i", "N", "n", "a", "k", "a", "cl", "k", "o"],
        )
        self.assertIn("N", result.phones)
        self.assertIn("cl", result.phones)

    def test_ordinary_automatic_target_keeps_surface_context_phone_contract(self) -> None:
        text = "それは、かっこです。"
        result = build_japanese_target_evidence(text)
        self.assertEqual(result.reading_source, "pyopenjtalk_g2p")
        self.assertIn("automatic_surface_context_drives_phone_sequence", result.warnings)
        self.assertEqual(result.phones, _phones(text))


if __name__ == "__main__":
    unittest.main()
