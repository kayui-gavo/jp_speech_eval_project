from __future__ import annotations

import csv
import unittest
from pathlib import Path

from jp_speech_eval.provider_adapters import TTSProviderResult
from scripts.audit_asr_weak_reference_combos import audit_rows as audit_asr_combos
from scripts.audit_tts_pseudo_reference_combos import audit_rows as audit_tts_combos


ROOT = Path(__file__).resolve().parents[1]


class ASRTTSModeSpecificBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.asr_rows = audit_asr_combos(
            ROOT / "data/asr_weak_reference_combo_plan.csv",
            ROOT / "results/calibration/asr_bottleneck_audit.csv",
        )
        cls.tts_rows = audit_tts_combos(
            ROOT / "data/tts_pseudo_reference_combo_plan.csv",
            ROOT / "results/calibration/tts_reference_quality_audit.csv",
            ROOT / "results/calibration/jvs_verified_pitch_demo_summary.csv",
        )

    def test_audits_are_offline_and_require_no_api_key(self) -> None:
        rows = self.asr_rows + self.tts_rows
        self.assertTrue(rows)
        self.assertTrue(all(not row["external_provider_called"] for row in rows))
        self.assertTrue(all(not row["api_key_required"] for row in rows))

    def test_weak_reference_benchmark_holds_tts_fixed(self) -> None:
        self.assertTrue(all(row["benchmark_mode"] == "asr_confirmed_weak_reference" for row in self.asr_rows))
        self.assertEqual({row["tts_setting"] for row in self.asr_rows}, {"current_demo_pseudo_reference"})
        self.assertTrue(all(not row["tts_is_primary_scoring_variable"] for row in self.asr_rows))

    def test_fixed_pseudo_reference_benchmark_varies_tts(self) -> None:
        self.assertTrue(all(row["benchmark_mode"] == "fixed_reference_pseudo_reference" for row in self.tts_rows))
        self.assertTrue(all(row["tts_is_primary_scoring_variable"] for row in self.tts_rows))
        self.assertEqual(len({row["combo_id"] for row in self.tts_rows}), 3)

    def test_combo_metadata_preserves_provider_model_and_provenance(self) -> None:
        for row in self.asr_rows + self.tts_rows:
            self.assertTrue(row["provider_name"])
            self.assertTrue(row["model_name"])
        self.assertIn("oracle_benchmark_only", {row["provider_provenance"] for row in self.asr_rows})
        self.assertIn("verified_native", {row["provenance"] for row in self.tts_rows})

    def test_synthetic_tts_is_never_promoted_to_reliable(self) -> None:
        synthetic = [row for row in self.tts_rows if row["provenance"] == "synthetic_tts"]
        self.assertTrue(synthetic)
        self.assertTrue(all(row["pitch_target_reliability"] == "weak" for row in synthetic))
        self.assertTrue(all(not row["strong_pitch_reference_allowed"] for row in synthetic))
        result = TTSProviderResult(
            audio_path="offline.wav",
            sample_rate=24000,
            provider_name="candidate",
            model_name="offline",
        )
        self.assertEqual(result.pitch_target_reliability, "weak")

    def test_missing_candidate_tts_is_reported_not_measured(self) -> None:
        candidate = next(row for row in self.tts_rows if row["combo_id"] == "best_candidate_tts_pseudo_reference")
        self.assertEqual(candidate["fixture_status"], "not_supplied")
        self.assertEqual(candidate["score_evidence_status"], "not_measured")
        self.assertIsNone(candidate["native_against_reference_prosody_score"])

    def test_verified_oracle_is_test_only_and_scores_are_not_fabricated(self) -> None:
        oracle = next(row for row in self.tts_rows if row["combo_id"] == "verified_native_reference_oracle")
        self.assertEqual(oracle["fixture_status"], "test_only_fixture")
        self.assertEqual(oracle["native_against_reference_prosody_score"], 85.75)
        current = next(row for row in self.tts_rows if row["combo_id"] == "current_tts_pseudo_reference")
        self.assertIsNone(current["native_against_reference_prosody_score"])

    def test_oracle_transcript_is_benchmark_only(self) -> None:
        plan = list(csv.DictReader((ROOT / "data/asr_weak_reference_combo_plan.csv").open(encoding="utf-8")))
        oracle = next(row for row in plan if row["combo_id"] == "oracle_transcript")
        self.assertEqual(oracle["provenance"], "oracle_benchmark_only")
        for relative in (
            "src/jp_speech_eval/asr.py",
            "src/jp_speech_eval/eval_modes.py",
            "src/jp_speech_eval/evaluator.py",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("provider_adapters", text, relative)
            self.assertNotIn("oracle_transcript", text, relative)

    def test_default_runtime_and_core_four_contract_are_unchanged(self) -> None:
        html = (ROOT / "debug_ui/index.html").read_text(encoding="utf-8")
        score_key_block = html.split("const scoreKeys = [", 1)[1].split("];", 1)[0]
        self.assertNotIn('"tone_score"', score_key_block)
        path_report = (ROOT / "reports/asr_tts_mode_specific_path_audit.md").read_text(encoding="utf-8")
        self.assertIn("not a strict pitch target", path_report)
        self.assertIn("Synthetic TTS provenance remains weak", path_report)


if __name__ == "__main__":
    unittest.main()
