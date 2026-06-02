from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.generate_demo_api_examples import build_examples
from scripts.run_demo_flow_smoke_tests import run as run_smoke_tests
from scripts.run_demo_readiness_check import collect_checks, readiness_status


ROOT = Path(__file__).resolve().parents[1]


class DemoReadinessPackTest(unittest.TestCase):
    def test_required_demo_readiness_reports_exist(self) -> None:
        required = [
            ROOT / "reports" / "demo_presentation_script.md",
            ROOT / "reports" / "demo_api_response_examples.md",
            ROOT / "reports" / "demo_known_limitations.md",
            ROOT / "reports" / "minimal_human_validation_plan.md",
            ROOT / "reports" / "demo_freeze_checklist.md",
            ROOT / "reports" / "demo_readiness_report.md",
        ]
        for path in required:
            self.assertTrue(path.exists(), str(path))

    def test_demo_readiness_check_detects_required_files(self) -> None:
        rows = collect_checks()
        by_check = {row["check"]: row for row in rows}
        self.assertEqual(by_check["file:demo_fixed_targets"]["status"], "ready")
        self.assertEqual(by_check["file:user_facing_messages"]["status"], "ready")
        self.assertEqual(by_check["file:api_contract"]["status"], "ready")
        self.assertNotEqual(readiness_status(rows), "blocked")

    def test_demo_api_examples_contain_user_facing(self) -> None:
        examples = build_examples()
        self.assertGreaterEqual(len(examples), 8)
        for item in examples:
            self.assertIn("user_facing", item["response"])
            self.assertIn("debug_summary", item["response"])

    def test_demo_api_examples_do_not_expose_forbidden_raw_fields(self) -> None:
        forbidden = ("debug_total_score", "prosody_score", "raw DTW", "raw F0", "threshold_low", "special_mora_decisions")
        for item in build_examples():
            user_facing = item["response"]["user_facing"]
            learner = {key: value for key, value in user_facing.items() if key != "debug"}
            text = json.dumps(learner, ensure_ascii=False)
            for token in forbidden:
                self.assertNotIn(token, text)

    def test_known_limitations_include_required_claim_limits(self) -> None:
        text = (ROOT / "reports" / "demo_known_limitations.md").read_text(encoding="utf-8")
        self.assertIn("practice_score` is demo guidance, not validated pronunciation ability", text)
        self.assertIn("Kanade is playback reference only", text)
        self.assertIn("ASR-generated reference requires user confirmation", text)
        self.assertIn("JANON trend is not ground truth", text)

    def test_existing_smoke_tests_still_pass(self) -> None:
        rows = run_smoke_tests()
        self.assertTrue(rows)
        self.assertTrue(all(row["passed"] for row in rows))
        self.assertTrue(all(row["weak_reference_conservative"] for row in rows))


if __name__ == "__main__":
    unittest.main()
