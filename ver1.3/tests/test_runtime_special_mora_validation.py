from __future__ import annotations

import argparse
import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.validate_runtime_special_mora_shadow import run
from scripts.analyze_special_mora_false_alarms import run as run_false_alarm_analysis
from scripts.validate_runtime_special_mora_profile import run as run_profile_validation
from scripts.build_special_mora_manual_inspection_pack import run as run_manual_pack
from scripts.build_special_mora_manual_review_viewer import build_annotation_template, build_review_viewer
from scripts.summarize_special_mora_manual_annotations import summarize, write_outputs
from scripts.evaluate_special_mora_rollout_gate import evaluate_gate


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RuntimeSpecialMoraValidationTest(unittest.TestCase):
    def test_validation_outputs_reports_and_keeps_sokuon_yoon_blocked(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            thresholds = root / "thresholds.json"
            thresholds.write_text(json.dumps({
                "thresholds": {
                    "long_vowel": {"status": "active", "low_ratio": 0.5, "high_ratio": 1.5, "sample_count": 40, "source_dataset": "JVS"},
                    "moraic_nasal": {"status": "active", "low_ratio": 0.5, "high_ratio": 1.5, "sample_count": 40, "source_dataset": "JVS"},
                    "sokuon": {"status": "insufficient", "low_ratio": None, "high_ratio": None, "sample_count": 12, "source_dataset": "JVS"},
                    "yoon": {"status": "debug_only", "low_ratio": 0.8, "high_ratio": 2.2, "sample_count": 66, "source_dataset": "JVS"},
                }
            }), encoding="utf-8")
            jvs = root / "jvs.csv"
            janon = root / "janon.csv"
            rows = [
                {"dataset": "jvs", "speaker_id": "s1", "utterance_id": "u1", "audio_path": "jvs/u1.wav", "transcript": "ラーメン", "special_mora_type": "long_vowel", "surface_mora": "ー", "mora_index": "2", "long_vowel_ratio_to_avg_mora": "1.0", "evidence_confidence": "1.0", "mapping_success": "True", "alignment_method": "existing_label", "alignment_fallback": "False"},
                {"dataset": "jvs", "speaker_id": "s1", "utterance_id": "u2", "audio_path": "jvs/u2.wav", "transcript": "ほん", "special_mora_type": "moraic_nasal", "surface_mora": "ン", "mora_index": "2", "nasal_ratio_to_avg_mora": "0.4", "evidence_confidence": "1.0", "mapping_success": "True", "alignment_method": "existing_label", "alignment_fallback": "False"},
                {"dataset": "jvs", "speaker_id": "s1", "utterance_id": "u3", "audio_path": "jvs/u3.wav", "transcript": "きって", "special_mora_type": "sokuon", "surface_mora": "ッ", "mora_index": "2", "closure_ratio_to_neighbor_mora": "0.2", "evidence_confidence": "1.0", "mapping_success": "True", "alignment_method": "existing_label", "alignment_fallback": "False"},
                {"dataset": "jvs", "speaker_id": "s1", "utterance_id": "u4", "audio_path": "jvs/u4.wav", "transcript": "きゃ", "special_mora_type": "yoon", "surface_mora": "キャ", "mora_index": "1", "ratio_to_avg_mora": "0.3", "evidence_confidence": "1.0", "mapping_success": "True", "alignment_method": "existing_label", "alignment_fallback": "False"},
            ]
            _write_csv(jvs, rows)
            _write_csv(janon, [dict(row, dataset="janon") for row in rows])
            out = root / "out"
            result = run(argparse.Namespace(
                threshold_path=thresholds,
                jvs_sample_audit=jvs,
                janon_sample_audit=janon,
                special_mora_types="long_vowel,moraic_nasal,sokuon,yoon",
                jvs_limit=0,
                output_dir=out,
                report_dir=root / "reports",
            ))
            self.assertTrue((out / "jvs_shadow_decisions.csv").exists())
            self.assertTrue((out / "counterfactual_feature_sensitivity.csv").exists())
            self.assertTrue((out / "jvs_false_alarm_by_type.csv").exists())
            self.assertGreater(result["counterfactual_rows"], 0)

            report_dir = root / "reports"
            readiness = (report_dir / "special_mora_user_facing_readiness.md").read_text(encoding="utf-8")
            janon_report = (report_dir / "runtime_special_mora_janon_shadow_trend.md").read_text(encoding="utf-8")
            self.assertIn("status: blocked", readiness)
            self.assertIn("status: blocked/debug_only", readiness)
            self.assertIn("trend-only", janon_report)

    def test_false_alarm_analysis_generates_v2_without_overwriting_v1(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            thresholds = root / "thresholds.json"
            original = {
                "thresholds": {
                    "long_vowel": {"type": "long_vowel", "status": "active", "low_ratio": 0.5, "high_ratio": 1.5, "sample_count": 40, "source_dataset": "JVS"},
                    "moraic_nasal": {"type": "moraic_nasal", "status": "active", "low_ratio": 0.5, "high_ratio": 1.5, "sample_count": 40, "source_dataset": "JVS"},
                    "sokuon": {"type": "sokuon", "status": "insufficient", "low_ratio": None, "high_ratio": None, "sample_count": 12, "source_dataset": "JVS"},
                    "yoon": {"type": "yoon", "status": "debug_only", "low_ratio": 0.8, "high_ratio": 2.2, "sample_count": 66, "source_dataset": "JVS"},
                }
            }
            thresholds.write_text(json.dumps(original), encoding="utf-8")
            sample = root / "sample.csv"
            rows = [
                {"dataset": "jvs", "speaker_id": "s1", "utterance_id": f"u{i}", "audio_path": f"jvs/u{i}.wav", "transcript": "ラーメン", "special_mora_type": "long_vowel", "surface_mora": "ー", "mora_index": "2", "long_vowel_ratio_to_avg_mora": str(0.35 + i * 0.05), "evidence_confidence": "1.0", "mapping_success": "True", "alignment_method": "existing_label", "alignment_fallback": "False", "expected_mora_sequence": "ラ ー メ ン", "avg_mora_duration": "0.2", "neighbor_prev_duration": "0.2", "neighbor_next_duration": "0.2"}
                for i in range(10)
            ] + [
                {"dataset": "jvs", "speaker_id": "s1", "utterance_id": f"n{i}", "audio_path": f"jvs/n{i}.wav", "transcript": "ほん", "special_mora_type": "moraic_nasal", "surface_mora": "ン", "mora_index": "2", "nasal_ratio_to_avg_mora": str(0.35 + i * 0.05), "evidence_confidence": "1.0", "mapping_success": "True", "alignment_method": "existing_label", "alignment_fallback": "False", "expected_mora_sequence": "ホ ン", "avg_mora_duration": "0.2", "neighbor_prev_duration": "0.2", "neighbor_next_duration": "0.2"}
                for i in range(10)
            ] + [
                {"dataset": "jvs", "speaker_id": "s1", "utterance_id": "s1", "audio_path": "jvs/s.wav", "transcript": "きって", "special_mora_type": "sokuon", "surface_mora": "ッ", "mora_index": "2", "closure_ratio_to_neighbor_mora": "0.2", "evidence_confidence": "1.0", "mapping_success": "True", "alignment_method": "existing_label", "alignment_fallback": "False", "expected_mora_sequence": "キ ッ テ", "avg_mora_duration": "0.2"},
                {"dataset": "jvs", "speaker_id": "s1", "utterance_id": "y1", "audio_path": "jvs/y.wav", "transcript": "きゃ", "special_mora_type": "yoon", "surface_mora": "キャ", "mora_index": "1", "ratio_to_avg_mora": "0.2", "evidence_confidence": "1.0", "mapping_success": "True", "alignment_method": "existing_label", "alignment_fallback": "False", "expected_mora_sequence": "キャ", "avg_mora_duration": "0.2"},
            ]
            _write_csv(sample, rows)
            janon = root / "janon.csv"
            _write_csv(janon, [dict(row, dataset="janon") for row in rows[:2]])
            out = root / "out"
            result = run_false_alarm_analysis(argparse.Namespace(
                shadow_decisions=out / "missing.csv",
                sample_audit=sample,
                janon_sample_audit=janon,
                threshold_path=thresholds,
                output_dir=out,
                report_dir=root / "reports",
            ))
            self.assertTrue((out / "special_mora_false_alarm_cases.csv").exists())
            self.assertTrue((out / "special_mora_threshold_sweep_v2.csv").exists())
            v2 = out / "special_mora_thresholds_v2.json"
            self.assertTrue(v2.exists())
            self.assertEqual(json.loads(thresholds.read_text(encoding="utf-8")), original)
            readiness = (root / "reports" / "special_mora_user_facing_readiness.md").read_text(encoding="utf-8")
            janon_report = (root / "reports" / "runtime_special_mora_janon_shadow_trend_v2.md").read_text(encoding="utf-8")
            self.assertIn("sokuon", readiness)
            self.assertIn("debug_only", readiness)
            self.assertIn("v2 trend is not scoring validation", janon_report)
            self.assertGreaterEqual(result["sweep_rows"], 1)

    def test_profile_validation_and_manual_pack_are_generated(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            reports = root / "reports"
            rows = run_profile_validation(argparse.Namespace(output_dir=out, report_dir=reports))
            self.assertTrue((out / "profile_validation_summary.csv").exists())
            self.assertTrue((reports / "runtime_special_mora_profile_validation.md").exists())
            by_profile = {(row["profile_name"], row["flag_enabled"]): row for row in rows}
            self.assertEqual(by_profile[("default_safe", False)]["candidate_count"], 0)
            self.assertEqual(by_profile[("v2_shadow", False)]["candidate_count"], 0)
            self.assertGreaterEqual(by_profile[("v2_limited_candidate", True)]["candidate_count"], 1)
            self.assertEqual(by_profile[("v2_limited_candidate", True)]["sokuon_yoon_leakage"], 0)
            self.assertEqual(by_profile[("v2_limited_candidate", True)]["too_long_leakage"], 0)

            false_cases = out / "false.csv"
            v2_jvs = out / "v2.csv"
            janon = out / "janon.csv"
            counter = out / "counter.csv"
            sample = out / "sample.csv"
            base = [{"dataset": "jvs", "speaker_id": "s", "utterance_id": "u", "audio_path": "a.wav", "transcript": "ラーメン", "special_mora_type": "long_vowel", "surface_mora": "ー", "decision": "too_short", "feature_value": "0.2", "user_low": "0.23", "near_boundary": "False", "evidence_confidence": "1.0", "phone_sequence_for_mora": "a"}]
            _write_csv(false_cases, base)
            _write_csv(v2_jvs, [dict(base[0], user_feedback_allowed="True")])
            _write_csv(janon, base)
            _write_csv(counter, [dict(base[0], too_short="True")])
            _write_csv(sample, base)
            items = run_manual_pack(argparse.Namespace(
                false_alarm_cases=false_cases,
                v2_jvs_decisions=v2_jvs,
                janon_v2=janon,
                counterfactual=counter,
                sample_audit=sample,
                output_csv=out / "manual.csv",
                report=reports / "manual.md",
            ))
            self.assertTrue((out / "manual.csv").exists())
            self.assertTrue((reports / "manual.md").exists())
            self.assertGreaterEqual(len(items), 1)

    def test_manual_review_workflow_and_rollout_gate_block_without_annotations(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            items = root / "items.csv"
            rows = [{
                "item_id": "i1",
                "source": "JVS_false_alarm",
                "dataset": "jvs",
                "speaker_id": "s",
                "utterance_id": "u",
                "audio_path": str(root / "missing.wav"),
                "transcript": "ラーメン",
                "special_mora_type": "long_vowel",
                "surface_mora": "ー",
                "decision": "too_short",
                "feature_value": "0.2",
                "threshold_user_low": "0.23",
                "near_boundary": "False",
                "evidence_confidence": "1.0",
                "phone_sequence_for_mora": "a",
            }]
            _write_csv(items, rows)
            template = root / "annotations_template.csv"
            self.assertTrue(build_annotation_template(items, template))
            before = template.read_text(encoding="utf-8")
            self.assertFalse(build_annotation_template(items, template))
            self.assertEqual(template.read_text(encoding="utf-8"), before)
            self.assertTrue(build_annotation_template(items, template, overwrite=True))
            html = root / "viewer.html"
            self.assertEqual(build_review_viewer(items, html), 1)
            html_text = html.read_text(encoding="utf-8")
            self.assertIn("特殊拍人工复核", html_text)
            self.assertIn("找不到音频文件", html_text)
            self.assertIn("missing.wav", html_text)

            audio = root / "exists.wav"
            audio.write_bytes(b"RIFFxxxxWAVE")
            _write_csv(items, [dict(rows[0], audio_path=str(audio))])
            self.assertEqual(build_review_viewer(items, html), 1)
            html_text = html.read_text(encoding="utf-8")
            self.assertIn("<audio controls", html_text)
            self.assertIn("exists.wav", html_text)

            missing_summary = summarize(root / "missing_annotations.csv")
            self.assertEqual(missing_summary["total_annotated"], 0)
            empty = root / "empty.csv"
            empty.write_text("", encoding="utf-8")
            self.assertEqual(summarize(empty)["total_annotated"], 0)
            summary_csv = root / "summary.csv"
            report = root / "summary.md"
            write_outputs(missing_summary, summary_csv, report)
            self.assertIn("no human annotations yet", report.read_text(encoding="utf-8"))

            profile_csv = root / "profile.csv"
            _write_csv(profile_csv, [{
                "profile_name": "v2_limited_candidate",
                "flag_enabled": "True",
                "sokuon_yoon_leakage": "0",
                "too_long_leakage": "0",
                "near_boundary_leakage": "0",
            }])
            false_csv = root / "false.csv"
            _write_csv(false_csv, [
                {"special_mora_type": "long_vowel", "false_alarm_proxy_rate_all": "0.034"},
                {"special_mora_type": "moraic_nasal", "false_alarm_proxy_rate_all": "0.045"},
            ])
            gate = evaluate_gate(profile_csv, false_csv, summary_csv)
            self.assertEqual(gate["decisions"]["long_vowel"]["rollout_status"], "blocked_pending_manual_inspection")
            self.assertEqual(gate["decisions"]["moraic_nasal"]["rollout_status"], "blocked_pending_manual_inspection")
            self.assertEqual(gate["decisions"]["sokuon"]["rollout_status"], "blocked_insufficient_native_evidence")
            self.assertEqual(gate["decisions"]["yoon"]["rollout_status"], "blocked_debug_only_duration_not_valid")


if __name__ == "__main__":
    unittest.main()
