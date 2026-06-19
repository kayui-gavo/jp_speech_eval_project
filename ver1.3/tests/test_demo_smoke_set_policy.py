from __future__ import annotations

import csv
import unittest
from pathlib import Path

from scripts.audit_demo_smoke_set import build_rows


ROOT = Path(__file__).resolve().parents[1]


class DemoSmokeSetPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = build_rows(
            ROOT / "data/demo_smoke_set_plan.csv",
            ROOT / "results/calibration/weak_reference_native_likeness_audit.csv",
            ROOT / "results/calibration/weak_reference_content_guardrail_audit.csv",
        )
        cls.by_case = {row["case_id"]: row for row in cls.rows}

    def test_plan_contains_required_ten_cases(self) -> None:
        self.assertEqual(len(self.rows), 10)
        self.assertTrue(all(row["pass_or_fail"] == "PASS" for row in self.rows))

    def test_english_latin_short_and_mismatch_do_not_show_scores(self) -> None:
        for case_id in (
            "random_english_latin",
            "latin_dominant_transcript",
            "very_short_japanese",
            "wrong_japanese_or_content_mismatch",
        ):
            self.assertEqual(self.by_case[case_id]["observed_score_visibility"], "no scores", case_id)

    def test_native_is_high_and_pitch_controls_are_lower(self) -> None:
        native = self.by_case["native_japanese_normal"]["pitch_mean"]
        flat = self.by_case["flat_pitch_control"]["pitch_mean"]
        random = self.by_case["random_pitch_control"]["pitch_mean"]
        self.assertGreaterEqual(native, 85)
        self.assertLess(flat, native)
        self.assertLess(random, native)

    def test_low_f0_is_unavailable_or_capped(self) -> None:
        row = self.by_case["low_f0_coverage"]
        self.assertEqual(row["observed_pitch_behavior"], "unavailable")
        self.assertIn("capped", row["observed_score_visibility"])

    def test_ui_uses_practice_wording_not_strict_pitch_accent(self) -> None:
        html = (ROOT / "debug_ui/index.html").read_text(encoding="utf-8")
        self.assertIn('pitch_score: "音高变化（自然度参考）"', html)
        self.assertNotIn('pitch_score: "音调（F0高低）"', html)
        self.assertNotIn('pitch_score: "アクセント（F0高低）"', html)

    def test_tone_score_is_not_a_core_dimension(self) -> None:
        html = (ROOT / "debug_ui/index.html").read_text(encoding="utf-8")
        score_key_block = html.split("const scoreKeys = [", 1)[1].split("];", 1)[0]
        self.assertNotIn('"tone_score"', score_key_block)

    def test_verified_fixed_reference_evidence_still_exists(self) -> None:
        report = (ROOT / "reports/jvs_verified_pitch_demo_summary.md").read_text(encoding="utf-8")
        self.assertIn("reference_audio_f0_cache", report)
        self.assertIn("cannot be directly used for the packaged", report)

    def test_generated_audit_schema_has_no_raw_debug_score(self) -> None:
        rows = list(csv.DictReader((ROOT / "results/calibration/demo_smoke_set_audit.csv").open(encoding="utf-8")))
        self.assertEqual(len(rows), 10)
        forbidden = {"raw_prosody_score", "tone_score", "raw_total_score"}
        self.assertFalse(forbidden.intersection(rows[0]))


if __name__ == "__main__":
    unittest.main()
