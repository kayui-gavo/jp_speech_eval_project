from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jp_speech_eval.target_specs import (
    build_verified_target_entry,
    parse_ints,
    parse_pitch_labels,
    validate_target_spec,
)
from jp_speech_eval.text_frontend import build_text_info


class TargetSpecsTest(unittest.TestCase):
    def test_parse_pitch_labels_accepts_common_formats(self) -> None:
        self.assertEqual(parse_pitch_labels("L H H L"), ["L", "H", "H", "L"])
        self.assertEqual(parse_pitch_labels("L,H,H,L"), ["L", "H", "H", "L"])
        self.assertEqual(parse_pitch_labels("LHHL"), ["L", "H", "H", "L"])

    def test_parse_ints_accepts_japanese_comma(self) -> None:
        self.assertEqual(parse_ints("2，3,4"), [2, 3, 4])

    def test_build_verified_target_entry_contains_schema_and_special_mora(self) -> None:
        entry = build_verified_target_entry(
            text="ラーメンをください",
            kana="ラーメンヲクダサイ",
            target_pitch=["L", "H", "H", "H", "H", "H", "L", "L", "L"],
            source="manual",
            phrase_lengths=[9],
            accent_positions=[6],
            note="unit test",
            verified_by="tester",
        )
        validation = validate_target_spec(entry)
        self.assertTrue(validation.ok, validation.errors)
        self.assertEqual(entry["schema_version"], "target_pronunciation_spec_v1")
        self.assertEqual(entry["special_mora"]["by_type"]["explicit_long_vowel"], [2])
        self.assertEqual(entry["special_mora"]["by_type"]["nasal"], [4])
        self.assertEqual(entry["accent_phrases"][0]["accent_position"], 6)

    def test_invalid_pitch_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_verified_target_entry(
                text="ラーメンをください",
                kana="ラーメンヲクダサイ",
                target_pitch=["L", "H"],
                source="manual",
                phrase_lengths=[9],
                accent_positions=[0],
            )

    def test_build_text_info_uses_verified_targets_env_path(self) -> None:
        entry = build_verified_target_entry(
            text="テスト",
            kana="テスト",
            target_pitch=["H", "L", "L"],
            source="manual",
            phrase_lengths=[3],
            accent_positions=[1],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "verified.json"
            path.write_text('{"テスト": ' + __import__("json").dumps(entry, ensure_ascii=False) + "}", encoding="utf-8")
            old = __import__("os").environ.get("JP_SPEECH_EVAL_VERIFIED_TARGETS")
            __import__("os").environ["JP_SPEECH_EVAL_VERIFIED_TARGETS"] = str(path)
            try:
                info = build_text_info("テスト")
            finally:
                if old is None:
                    __import__("os").environ.pop("JP_SPEECH_EVAL_VERIFIED_TARGETS", None)
                else:
                    __import__("os").environ["JP_SPEECH_EVAL_VERIFIED_TARGETS"] = old
        self.assertEqual(info.pitch_target_source, "manual")
        self.assertEqual(info.target_pitch, ["H", "L", "L"])


if __name__ == "__main__":
    unittest.main()
