from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jp_speech_eval.calibration_bench import read_manifest, summarize_audit_rows, write_csv


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


if __name__ == "__main__":
    unittest.main()
