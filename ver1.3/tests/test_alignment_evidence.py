from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jp_speech_eval.alignment_evidence import build_alignment_evidence
from jp_speech_eval.alignment_evidence.phone_mora_mapper import map_phones_to_moras
from jp_speech_eval.alignment_evidence.textgrid_parser import parse_textgrid_phone_segments
from jp_speech_eval.alignment_evidence.base import PhoneSegment
from jp_speech_eval.special_mora_alignment_features import extract_special_mora_alignment_features


class AlignmentEvidenceTest(unittest.TestCase):
    def test_equal_fallback_not_usable_for_special_mora(self) -> None:
        ev = build_alignment_evidence(
            backend="equal",
            utterance_id="utt",
            target_text="ラーメン",
            moras=["ラ", "ー", "メ", "ン"],
            mora_table=[{"start_sec": 0.0, "end_sec": 0.1}] * 4,
        )
        self.assertFalse(ev.usable_for_special_mora_feedback)
        self.assertFalse(ev.usable_for_pitch_feedback)

    def test_mfa_unavailable_does_not_crash_when_skipped(self) -> None:
        with patch("jp_speech_eval.alignment_evidence.mfa_available", return_value=False):
            ev = build_alignment_evidence(
                backend="mfa",
                utterance_id="utt",
                target_text="ラーメン",
                moras=["ラ", "ー", "メ", "ン"],
                mora_table=[{"start_sec": 0.0, "end_sec": 0.1}] * 4,
            )
        self.assertEqual(ev.method, "mfa_japanese_skipped")
        self.assertIn("mfa_unavailable", ev.warning_flags)

    def test_textgrid_parser_handles_simple_phone_segments(self) -> None:
        body = '''
File type = "ooTextFile"
Object class = "TextGrid"
item [1]:
    class = "IntervalTier"
    name = "phones"
    intervals [1]:
        xmin = 0
        xmax = 0.1
        text = "r"
    intervals [2]:
        xmin = 0.1
        xmax = 0.2
        text = "a"
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "utt.TextGrid"
            path.write_text(body, encoding="utf-8")
            rows = parse_textgrid_phone_segments(path)
        self.assertEqual([row.phone for row in rows], ["r", "a"])

    def test_phone_to_mora_mapper_warns_instead_of_crashing(self) -> None:
        segments, info = map_phones_to_moras([PhoneSegment("a", 0.0, 0.1)], ["ラ", "ー"])
        self.assertFalse(info["mapping_success"])
        self.assertIsInstance(segments, list)
        self.assertTrue(info["mapping_warning_flags"])

    def test_feature_extractor_returns_uncertain_for_unreliable_alignment(self) -> None:
        ev = build_alignment_evidence(
            backend="equal",
            utterance_id="utt",
            target_text="ラーメン",
            moras=["ラ", "ー", "メ", "ン"],
            mora_table=[
                {"start_sec": 0.0, "end_sec": 0.1},
                {"start_sec": 0.1, "end_sec": 0.2},
                {"start_sec": 0.2, "end_sec": 0.3},
                {"start_sec": 0.3, "end_sec": 0.4},
            ],
        )
        rows = extract_special_mora_alignment_features(ev)
        self.assertTrue(rows)
        self.assertTrue(all(row["uncertain"] for row in rows))

    def test_auto_backend_falls_back_gracefully(self) -> None:
        with patch("jp_speech_eval.alignment_evidence.mfa_available", return_value=False):
            ev = build_alignment_evidence(
                backend="auto",
                utterance_id="utt",
                target_text="ラーメン",
                moras=["ラ", "ー", "メ", "ン"],
                mora_table=[{"start_sec": 0.0, "end_sec": 0.1}] * 4,
            )
        self.assertIn(ev.method, {"mfcc_dtw", "mfa_japanese_skipped", "equal_fallback"})


if __name__ == "__main__":
    unittest.main()
