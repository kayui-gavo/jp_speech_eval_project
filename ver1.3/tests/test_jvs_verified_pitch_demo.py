from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_fixed_reference_prosody_targets import inventory_rows
from jp_speech_eval.feedback_renderer import render_user_facing_result


SUMMARY_CSV = ROOT / "results" / "calibration" / "jvs_verified_pitch_demo_summary.csv"
FIXTURE_DIR = ROOT / "results" / "test_fixtures" / "jvs_verified_pitch_demo"


def _rows() -> list[dict[str, str]]:
    if not SUMMARY_CSV.exists():
        raise AssertionError("run scripts/build_jvs_verified_pitch_demo.py first")
    return list(csv.DictReader(SUMMARY_CSV.open(encoding="utf-8")))


def _scores(case_name: str) -> list[float]:
    out: list[float] = []
    for row in _rows():
        if row.get("case_name") == case_name and row.get("prosody_score"):
            out.append(float(row["prosody_score"]))
    return out


def _mean(case_name: str) -> float:
    values = _scores(case_name)
    if not values:
        raise AssertionError(f"missing scores for {case_name}")
    return sum(values) / len(values)


class JvsVerifiedPitchDemoTests(unittest.TestCase):
    def test_test_only_jvs_reference_cache_not_in_packaged_demo_inventory(self) -> None:
        inventory = inventory_rows(
            cache_dir=ROOT / "cache",
            manifest_path=ROOT / "data" / "demo_fixed_targets.json",
            min_f0_coverage=0.50,
        )
        target_ids = {row["target_id"] for row in inventory}
        self.assertFalse(any(str(target).startswith("jvs") for target in target_ids))
        self.assertEqual([row for row in inventory if row["can_be_strong_pitch_reference"] == "yes"], [])

    def test_sidecars_are_test_only_jvs_reliable_reference_audio_cache(self) -> None:
        sidecars = sorted(FIXTURE_DIR.glob("*.prosody_ref.json"))
        self.assertGreaterEqual(len(sidecars), 3)
        for sidecar_path in sidecars:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertTrue(sidecar["reliable"])
            self.assertEqual(sidecar["pitch_target_source"], "reference_audio_f0_cache")
            self.assertEqual(sidecar["pitch_target_reliability"], "reliable")
            self.assertEqual(sidecar["reference_source"], "test_only_jvs_native_reference")

    def test_evaluator_reads_reference_audio_cache_and_openjtalk_does_not_override_it(self) -> None:
        fixed_rows = [row for row in _rows() if row.get("case_name") == "verified_jvs_sidecar_fixed_reference"]
        baseline_rows = [row for row in _rows() if row.get("case_name") == "same_audio_text_openjtalk_evaluator_baseline"]
        self.assertGreaterEqual(len(fixed_rows), 3)
        self.assertEqual(len(fixed_rows), len(baseline_rows))
        self.assertTrue(all(row["pitch_target_source"] == "reference_audio_f0_cache" for row in fixed_rows))
        self.assertTrue(all(row["pitch_target_reliability"] == "reliable" for row in fixed_rows))
        self.assertTrue(all(row["pitch_target_source"] == "openjtalk_accent_phrase_chain" for row in baseline_rows))
        self.assertTrue(all("fallback_alignment" in row.get("suppressed_reasons", "") for row in fixed_rows))

    def test_reference_audio_cache_scores_above_flat_and_random_controls(self) -> None:
        reference = _mean("native_cross_speaker_reference_audio_f0_cache")
        flat = _mean("flat_pitch_correct_content")
        random = _mean("shuffled_random_pitch_correct_content")
        self.assertGreater(reference, flat)
        self.assertGreater(reference, random)

    def test_reference_audio_cache_scores_above_openjtalk_target(self) -> None:
        self.assertGreater(
            _mean("native_cross_speaker_reference_audio_f0_cache"),
            _mean("native_cross_speaker_openjtalk_target"),
        )

    def test_low_f0_control_is_marked_insufficient(self) -> None:
        rows = [row for row in _rows() if row.get("case_name") == "low_f0_coverage_correct_content"]
        self.assertGreaterEqual(len(rows), 3)
        self.assertTrue(all(row.get("low_f0_unavailable") == "yes" for row in rows))
        self.assertTrue(all(row.get("unavailable_reason") == "insufficient_valid_mora_f0" for row in rows))

    def test_wrong_drop_remains_known_weak_separation(self) -> None:
        reference = _mean("native_cross_speaker_reference_audio_f0_cache")
        wrong = _mean("wrong_accent_drop_correct_content")
        self.assertGreater(reference, wrong)
        self.assertLess(reference - wrong, 10.0)

    def test_tone_score_not_in_core_four_and_content_gate_unchanged(self) -> None:
        rows = _rows()
        self.assertTrue(rows)
        self.assertTrue(all(row.get("tone_score_in_core_four") == "no" for row in rows if row.get("tone_score_in_core_four")))
        result = {
            "target_text": "ラーメンをください",
            "kana": "ラーメンヲクダサイ",
            "moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"],
            "mora_table": [],
            "total_score": 90,
            "pronunciation_score": 90,
            "prosody_score": 90,
            "fluency_score": 90,
            "tone_score": 99,
            "feedback": [],
            "alignment_mode": "cached_dtw",
            "details": {
                "mode": "reference_based",
                "verified_level": "human_checked",
                "pitch_target_source": "reference_audio_f0_cache",
                "pitch_target_reliability": "reliable",
                "reliability": {"level": "high", "overall": 0.95, "alignment": 0.95, "f0_coverage": 0.95},
                "recording_quality": {"score": 0.95},
                "content_match": {
                    "status": "pass",
                    "method": "asr_kana_match+mfcc_dtw_reference_gate",
                    "transcript": "please give me ramen",
                    "transcript_kana": "プリーズギブミーラーメン",
                    "target_kana": "ラーメンヲクダサイ",
                    "kana_similarity": 0.2,
                },
                "alignment": {"mode": "cached_dtw"},
                "prosody": {
                    "pitch_target_source": "reference_audio_f0_cache",
                    "pitch_target_reliability": "reliable",
                },
            },
        }
        rendered = render_user_facing_result(result)
        self.assertIsNone(rendered["display_score"])
        self.assertIn("content_mismatch_veto", rendered["suppressed_reasons"])
        self.assertEqual(rendered["debug"]["expression_proxy_score"], 99)


if __name__ == "__main__":
    unittest.main()
