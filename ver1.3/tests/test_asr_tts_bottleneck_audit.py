from __future__ import annotations

import unittest
from pathlib import Path

from scripts.audit_asr_bottleneck import _provider_summary, audit_rows as audit_asr_rows
from scripts.audit_tts_reference_quality import audit_rows as audit_tts_rows
from jp_speech_eval.provider_adapters import (
    ASRProviderResult,
    TTSProviderResult,
    load_provider_result_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "provider_results"


class ASRTTSBottleneckAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.asr_rows = audit_asr_rows(ROOT / "data/asr_benchmark_plan.csv")
        cls.tts_rows = audit_tts_rows(ROOT / "data/tts_benchmark_plan.csv")

    def test_asr_provider_fixture_is_offline(self) -> None:
        result = load_provider_result_fixture(FIXTURES / "asr_offline_example.json")
        self.assertIsInstance(result, ASRProviderResult)
        self.assertEqual(result.result_source, "offline_fixture")
        self.assertFalse(result.raw_metadata["network_called"])

    def test_tts_provider_fixture_is_offline(self) -> None:
        result = load_provider_result_fixture(FIXTURES / "tts_offline_example.json")
        self.assertIsInstance(result, TTSProviderResult)
        self.assertFalse(result.raw_metadata["network_called"])
        self.assertEqual(result.provenance, "synthetic_tts")

    def test_oracle_transcript_improves_good_japanese_content_gate(self) -> None:
        summary = {item["provider"]: item for item in _provider_summary(self.asr_rows)}
        self.assertGreater(
            summary["oracle_transcript"]["good_japanese_accept_rate"],
            summary["current_asr"]["good_japanese_accept_rate"],
        )
        self.assertGreaterEqual(summary["oracle_transcript"]["good_japanese_mean_kana_similarity"], 0.99)

    def test_hallucinated_japanese_candidate_is_not_scored_without_confirmation(self) -> None:
        row = next(
            item for item in self.asr_rows
            if item["case_id"] == "asr_hallucinated_japanese" and item["provider_slot"] == "current_asr"
        )
        self.assertTrue(row["hallucination_risk"])
        self.assertFalse(row["weak_score_visibility_now"])
        self.assertTrue(row["weak_score_visibility_if_user_confirms_unchanged"])

    def test_random_english_remains_rejected(self) -> None:
        rows = [item for item in self.asr_rows if item["case_id"] == "random_english"]
        self.assertTrue(rows)
        self.assertTrue(all(not item["weak_score_visibility_now"] for item in rows))
        self.assertTrue(all(item["content_gate_status"] != "pass" for item in rows))

    def test_tts_result_is_always_synthetic_and_weak(self) -> None:
        result = TTSProviderResult(
            audio_path="candidate.wav",
            sample_rate=24000,
            provider_name="candidate",
            model_name="candidate-model",
        )
        self.assertEqual(result.provenance, "synthetic_tts")
        self.assertEqual(result.pitch_target_reliability, "weak")
        self.assertFalse(result.strong_pitch_reference_allowed)
        with self.assertRaises(ValueError):
            TTSProviderResult(
                audio_path="candidate.wav",
                sample_rate=24000,
                provider_name="candidate",
                model_name="candidate-model",
                provenance="verified_native",
            )

    def test_adapter_metadata_preserves_provider_model_and_timestamps(self) -> None:
        result = load_provider_result_fixture(FIXTURES / "asr_offline_example.json")
        self.assertEqual(result.provider_name, "offline_replay")
        self.assertEqual(result.model_name, "candidate-model")
        self.assertTrue(result.word_timestamps)

    def test_no_external_provider_was_called_by_audits(self) -> None:
        self.assertTrue(all(not row["external_provider_called"] for row in self.asr_rows))
        self.assertTrue(all(not row["external_provider_called"] for row in self.tts_rows))

    def test_tts_slots_never_become_strong_pitch_reference(self) -> None:
        synthetic_rows = [row for row in self.tts_rows if row["provider_slot"] != "human_native_reference_oracle"]
        self.assertTrue(all(row["provenance"] == "synthetic_tts" for row in synthetic_rows))
        self.assertTrue(all(not row["strong_pitch_reference_allowed"] for row in synthetic_rows))
        self.assertTrue(all(row["sidecar_reliability"] == "weak" for row in synthetic_rows))

    def test_default_runtime_does_not_import_provider_adapter(self) -> None:
        for relative in (
            "src/jp_speech_eval/asr.py",
            "src/jp_speech_eval/eval_modes.py",
            "src/jp_speech_eval/evaluator.py",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("provider_adapters", text, relative)

    def test_tone_score_is_not_restored_to_core_four(self) -> None:
        html = (ROOT / "debug_ui/index.html").read_text(encoding="utf-8")
        score_key_block = html.split("const scoreKeys = [", 1)[1].split("];", 1)[0]
        self.assertNotIn('"tone_score"', score_key_block)


if __name__ == "__main__":
    unittest.main()
