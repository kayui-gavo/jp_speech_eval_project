from __future__ import annotations

import unittest

from jp_speech_eval.japanese_phone_inventory import (
    acoustic_phone_token_ids,
    acoustic_phone_tokens,
    inventory_semantics,
)
from jp_speech_eval.japanese_phoneme_gop import segmental_competitor_ids


class JapanesePhoneInventoryPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = {
            "PAD": 0,
            "a": 1,
            "k": 2,
            "N": 3,
            "cl": 4,
            "pau": 5,
            "sil": 6,
            "UNK": 7,
        }

    def test_acoustic_inventory_includes_special_morae(self) -> None:
        ids = acoustic_phone_token_ids(self.vocab, blank_id=0)
        self.assertEqual(ids, [1, 2, 3, 4])
        self.assertEqual(acoustic_phone_tokens(self.vocab, blank_id=0), ["a", "k", "N", "cl"])

    def test_clarity_competitors_are_stricter_than_acoustic_inventory(self) -> None:
        acoustic = set(acoustic_phone_token_ids(self.vocab, blank_id=0))
        clarity = set(segmental_competitor_ids(self.vocab, blank_id=0))
        self.assertEqual(clarity, {1, 2})
        self.assertEqual(acoustic - clarity, {3, 4})

    def test_inventory_semantics_are_machine_readable(self) -> None:
        result = inventory_semantics(self.vocab, blank_id=0)
        self.assertTrue(result["includes_special_mora_N"])
        self.assertTrue(result["includes_special_mora_cl"])
        self.assertTrue(result["excludes_pause_and_control_tokens"])
        self.assertFalse(result["ordinary_clarity_competitor_inventory"])
        self.assertEqual(result["token_count"], 4)

    def test_empty_acoustic_inventory_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "inventory is empty"):
            acoustic_phone_token_ids({"PAD": 0, "pau": 1, "sil": 2}, blank_id=0)


if __name__ == "__main__":
    unittest.main()
