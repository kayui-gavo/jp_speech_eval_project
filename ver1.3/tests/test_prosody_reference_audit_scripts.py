from __future__ import annotations

import csv
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_cross_speaker_prosody_reference import (  # noqa: E402
    f0_coverage,
    flat_f0,
    low_f0_coverage,
    shuffled_f0,
    wrong_drop_f0,
)
from audit_fixed_reference_prosody_targets import inventory_rows  # noqa: E402
from build_test_jvs_prosody_reference_cache import build_test_jvs_cache  # noqa: E402
from diagnose_wrong_drop_sensitivity import diagnose  # noqa: E402
from jp_speech_eval.evaluator import evaluate_utterance  # noqa: E402
from jp_speech_eval.prosody_reference_cache import prosody_reference_cache_path  # noqa: E402


class ProsodyReferenceAuditScriptTests(unittest.TestCase):
    def test_packaged_fixed_reference_inventory_has_no_verified_strong_pitch_target(self) -> None:
        rows = inventory_rows(
            cache_dir=ROOT / "cache",
            manifest_path=ROOT / "data" / "demo_fixed_targets.json",
            min_f0_coverage=0.50,
        )

        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(
            [row for row in rows if row.get("can_be_strong_pitch_reference") == "yes"],
            [],
        )

        ramen = next(row for row in rows if row.get("target_id") == "ramen_kudasai")
        self.assertEqual(ramen["reference_source"], "pyopenjtalk_tts_pseudo_reference")
        self.assertEqual(ramen["pitch_target_source"], "tts_reference_weak")
        self.assertEqual(ramen["pitch_target_reliability"], "weak")
        self.assertEqual(ramen["has_prosody_ref_sidecar"], "no")
        self.assertEqual(ramen["manifest_reference_audio_exists"], "yes")
        self.assertEqual(ramen["manifest_verified_level"], "auto_pyopenjtalk")
        self.assertEqual(ramen["manifest_pitch_reference_status"], "weak_tts_pseudo_reference")
        self.assertEqual(ramen["can_be_strong_pitch_reference"], "no")
        self.assertIn("untrusted_reference_source", ramen["reason_if_not"])
        self.assertIn("weak_tts_pseudo_reference", ramen["reason_if_not"])

    def test_manifest_claims_human_checked_missing_reference_path_is_not_strong(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest = tmp / "targets.json"
            manifest.write_text(json.dumps([{
                "target_id": "broken_target",
                "target_text": "ラーメンをください",
                "kana": "ラーメンヲクダサイ",
                "reference_audio": "missing/reference.wav",
                "verified_level": "human_checked",
                "pitch_target_source": "human_checked",
            }], ensure_ascii=False), encoding="utf-8")
            rows = inventory_rows(cache_dir=tmp / "cache", manifest_path=manifest, min_f0_coverage=0.50)
        broken = rows[0]
        self.assertEqual(broken["can_be_strong_pitch_reference"], "no")
        self.assertIn("missing_sentence_cache", broken["reason_if_not"])
        self.assertIn("manifest_reference_audio_missing", broken["reason_if_not"])

    def test_cross_speaker_sanity_artifact_keeps_native_above_counterfactuals(self) -> None:
        csv_path = ROOT / "results" / "calibration" / "cross_speaker_prosody_reference_sanity.csv"
        self.assertTrue(csv_path.exists(), "run scripts/audit_cross_speaker_prosody_reference.py first")
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        self.assertGreaterEqual(len(rows), 6)

        def mean_for(mode: str) -> float:
            values = [float(row["prosody_score"]) for row in rows if row.get("mode") == mode and row.get("prosody_score")]
            self.assertGreater(values, [], mode)
            return statistics.fmean(values)

        native_reference = mean_for("native_cross_speaker_reference_audio_f0_cache")
        openjtalk = mean_for("native_cross_speaker_openjtalk_target")
        flat = mean_for("flat_pitch_correct_content")
        shuffled = mean_for("shuffled_random_pitch_correct_content")
        wrong_drop = mean_for("wrong_accent_drop_correct_content")
        low_coverage = mean_for("low_f0_coverage_correct_content")

        self.assertGreater(native_reference, openjtalk)
        self.assertGreater(native_reference, flat)
        self.assertGreater(native_reference, shuffled)
        self.assertGreater(native_reference, wrong_drop)
        self.assertGreater(native_reference, low_coverage)
        self.assertLess(wrong_drop - flat, native_reference - flat)

    def test_counterfactual_helpers_preserve_shape_and_reduce_pitch_evidence(self) -> None:
        values = [100.0, 126.0, 148.0, 142.0, 108.0, 103.0, 99.0, 96.0, 92.0]
        accent_phrases = [{"moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"], "accent_position": 4}]

        flat = flat_f0(values)
        shuffled = shuffled_f0(values, seed=4)
        wrong_drop = wrong_drop_f0(values, values, accent_phrases)
        sparse = low_f0_coverage(values)

        self.assertEqual(len(flat), len(values))
        self.assertEqual(len(shuffled), len(values))
        self.assertEqual(len(wrong_drop), len(values))
        self.assertEqual(len(sparse), len(values))
        self.assertNotEqual(shuffled, values)
        self.assertGreater(wrong_drop[4], wrong_drop[3])
        self.assertLess(f0_coverage(sparse), 0.50)

    def test_reports_explicitly_keep_calibration_inactive(self) -> None:
        inventory_report = (ROOT / "reports" / "fixed_reference_prosody_target_inventory.md").read_text(encoding="utf-8")
        cross_report = (ROOT / "reports" / "cross_speaker_prosody_reference_sanity.md").read_text(encoding="utf-8")

        self.assertIn("No packaged fixed-reference target currently has a verified reliable human/native reference F0 sidecar.", inventory_report)
        self.assertIn("This audit does not make calibration active.", cross_report)
        self.assertIn("Not ready.", cross_report)

    def test_test_only_jvs_verified_sidecar_reaches_evaluator_path_when_jvs_available(self) -> None:
        jvs_root = ROOT.parent / "JVS"
        if not (jvs_root / "jvs001").exists() or not (jvs_root / "jvs002").exists():
            self.skipTest("JVS cross-speaker fixture not available")
        with tempfile.TemporaryDirectory() as tmp_name:
            prefix = Path(tmp_name) / "jvs001_voiceactress100_001"
            cache = build_test_jvs_cache(
                jvs_root=jvs_root,
                speaker_id="jvs001",
                utterance_id="VOICEACTRESS100_001",
                out_prefix=prefix,
                write_sidecar=True,
            )
            sidecar = json.loads(prosody_reference_cache_path(prefix).read_text(encoding="utf-8"))
            result = evaluate_utterance(
                wav_path=jvs_root / "jvs002" / "parallel100" / "wav24kHz16bit" / "VOICEACTRESS100_001.wav",
                alignment_mode="cached_dtw",
                cache_path=cache.prefix,
                use_content_match=True,
            ).to_dict()
        self.assertTrue(sidecar["reliable"])
        self.assertEqual(sidecar["reference_provenance_status"], "trusted")
        self.assertEqual(result["details"]["pitch_target_source"], "reference_audio_f0_cache")
        self.assertEqual(result["details"]["pitch_target_reliability"], "reliable")
        self.assertEqual(result["details"]["content_match"]["status"], "pass")

    def test_wrong_drop_diagnostic_confirms_current_weak_separation_when_jvs_available(self) -> None:
        jvs_root = ROOT.parent / "JVS"
        if not (jvs_root / "jvs001").exists() or not (jvs_root / "jvs002").exists():
            self.skipTest("JVS cross-speaker fixture not available")
        rows = diagnose(jvs_root, max_speakers=2, max_pairs=3, sample_rate=16000)
        self.assertGreaterEqual(len(rows), 1)
        deltas = [float(row["score_delta_native_minus_wrong"]) for row in rows]
        self.assertLess(statistics.fmean(deltas), 12.0)
        self.assertTrue(any(row["diagnosis"] == "drop_changed_but_no_explicit_drop_component" for row in rows))


if __name__ == "__main__":
    unittest.main()
