from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path

from jp_speech_eval.calibration_bench import read_manifest, summarize_audit_rows, write_csv


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_calibration_snapshot.py"
_spec = importlib.util.spec_from_file_location("run_calibration_snapshot", SCRIPT_PATH)
assert _spec and _spec.loader
snapshot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(snapshot)


class CalibrationBenchTest(unittest.TestCase):
    def test_summarize_audit_rows_groups_native_and_l2(self) -> None:
        rows = [
            {"dataset": "jvs", "split": "native", "score_total": 90, "speech_rate_mora_per_sec": 5.0, "score_is_diagnostic": False},
            {"dataset": "jvs", "split": "native", "score_total": 80, "speech_rate_mora_per_sec": 6.0, "score_is_diagnostic": False},
            {"dataset": "janon", "split": "l2", "score_total": 60, "speech_rate_mora_per_sec": 7.5, "score_is_diagnostic": True},
        ]
        summary = summarize_audit_rows(rows, numeric_fields=["score_total", "speech_rate_mora_per_sec"])
        by_group = {(row["dataset"], row["split"]): row for row in summary}
        self.assertEqual(by_group[("jvs", "native")]["n_utterances"], 2)
        self.assertEqual(by_group[("jvs", "native")]["score_total_p50"], 85.0)
        self.assertEqual(by_group[("janon", "l2")]["diagnostic_rate"], 1.0)

    def test_read_manifest_requires_audio_and_text_or_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            path.write_text(
                "audio_path,text,dataset,split,cache_path\n"
                "a.wav,ラーメン,jvs,native,\n"
                "b.wav,,jvs,native,cache/b\n"
                "c.wav,,jvs,native,\n",
                encoding="utf-8",
            )
            rows = read_manifest(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["dataset"], "jvs")

    def test_write_csv_handles_empty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            write_csv(path, [])
            self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_insufficient_special_mora_coverage_does_not_update_threshold(self) -> None:
        thresholds = snapshot._build_special_thresholds(
            [{"special_type": "sokuon", "ratio_to_avg_mora": 0.8, "uncertain": False, "alignment_fallback": False}],
            min_coverage=5,
        )
        self.assertFalse(thresholds["thresholds"]["sokuon"]["sufficient_evidence"])
        self.assertIsNone(thresholds["thresholds"]["sokuon"]["low_ratio"])

    def test_janon_special_report_says_trend_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "janon.md"
            snapshot._write_janon_special_report(path, [{"special_type": "long_vowel", "count": 1}])
            text = path.read_text(encoding="utf-8")
        self.assertIn("trend only", text)
        self.assertIn("do not prove scoring correctness", text)

    def test_threshold_metadata_includes_feature_definition(self) -> None:
        thresholds = snapshot._build_special_thresholds(
            [{"special_type": "long_vowel", "ratio_to_avg_mora": 0.8, "uncertain": False, "alignment_unsafe_for_threshold": False}],
            min_coverage=1,
        )
        entry = thresholds["thresholds"]["long_vowel"]
        self.assertIn("feature_definition", entry)
        self.assertIn("denominator", entry)

    def test_yoon_threshold_is_debug_only_by_default(self) -> None:
        rows = [
            {"special_type": "yoon", "ratio_to_avg_mora": 1.0 + i * 0.01, "uncertain": False, "alignment_unsafe_for_threshold": False}
            for i in range(35)
        ]
        thresholds = snapshot._build_special_thresholds(rows, min_coverage=30)
        self.assertEqual(thresholds["thresholds"]["yoon"]["status"], "debug_only")

    def test_threshold_updater_warns_on_suspicious_distribution(self) -> None:
        rows = [
            {"special_type": "long_vowel", "ratio_to_avg_mora": 0.01 if i == 0 else 1.0, "uncertain": False, "alignment_unsafe_for_threshold": False}
            for i in range(35)
        ]
        thresholds = snapshot._build_special_thresholds(rows, min_coverage=30)
        self.assertTrue(thresholds["thresholds"]["long_vowel"]["warnings"])

    def test_apply_special_decisions_includes_evidence_confidence(self) -> None:
        thresholds = snapshot._build_special_thresholds(
            [{"special_type": "long_vowel", "ratio_to_avg_mora": 1.0, "uncertain": False, "alignment_unsafe_for_threshold": False}],
            min_coverage=1,
        )
        rows = snapshot._apply_special_decisions(
            [{"special_type": "long_vowel", "ratio_to_avg_mora": 1.0, "evidence_confidence": 0.9, "uncertain": False}],
            thresholds,
        )
        self.assertEqual(rows[0]["decision"], "ok")
        self.assertEqual(rows[0]["evidence_confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
