from __future__ import annotations

import argparse
import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.validate_runtime_special_mora_shadow import run


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


if __name__ == "__main__":
    unittest.main()
