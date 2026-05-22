from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from jp_speech_eval.app_core.calibration import build_voice_profile
from jp_speech_eval.app_core.personalized_scorer import compare_to_profile
from jp_speech_eval.app_core.practice_modes import compute_reference_dependency_gap
from jp_speech_eval.app_core.progress_tracker import (
    ProgressRecord,
    append_progress_record,
    latest_record,
    load_progress_records,
)
from jp_speech_eval.app_core.user_profile import CalibrationSample


def _sample(total: float, rate: float, feedback: list[str] | None = None) -> CalibrationSample:
    return CalibrationSample(
        text="ラーメンをください",
        audio_path="dummy.wav",
        kana="ラーメンヲクダサイ",
        mora_count=9,
        scores={
            "total": total,
            "pronunciation": total,
            "prosody": total,
            "fluency": total,
            "expression": total,
        },
        features={
            "f0_median_hz": 180.0,
            "f0_range_log": 0.5,
            "mora_rate": rate,
            "avg_mora_duration_sec": 1.0 / rate,
            "pause_ratio": 0.05,
            "intensity_avg": 0.1,
        },
        reliability={"overall": 0.9, "level": "high"},
        feedback=feedback or [],
    )


class AppCoreMvpTest(unittest.TestCase):
    def test_build_voice_profile_aggregates_baseline(self) -> None:
        profile = build_voice_profile(
            "u1",
            [_sample(70, 5.0, ["语速有点快"]), _sample(80, 4.0, ["语速有点快"])],
        )
        self.assertEqual(profile.user_id, "u1")
        self.assertAlmostEqual(profile.baseline_scores["total"], 75.0)
        self.assertAlmostEqual(profile.mora_rate_avg or 0.0, 4.5)
        self.assertEqual(profile.common_issues, ["语速有点快"])

    def test_personalized_feedback_compares_previous(self) -> None:
        profile = build_voice_profile("u1", [_sample(70, 5.0)])
        result = SimpleNamespace(
            total_score=78,
            pronunciation_score=78,
            prosody_score=77,
            fluency_score=80,
            tone_score=65,
            feedback=[],
            pause_info={"pause_ratio": 0.04},
            details={
                "reliability": {"level": "high"},
                "fluency": {"speech_rate_mora_per_sec": 4.4, "avg_mora_duration_sec": 0.227},
                "tone": {"pitch_range_log": 0.52},
            },
        )
        previous = {"scores": {"total": 72}, "features": {"mora_rate": 5.1}}
        comparison = compare_to_profile(result, profile=profile, previous_record=previous)
        self.assertGreater(comparison.progress_delta["total_vs_previous"] or 0.0, 0.0)
        self.assertTrue(any("上次" in item for item in comparison.feedback))

    def test_progress_jsonl_roundtrip_and_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            append_progress_record(
                path,
                ProgressRecord(
                    user_id="u1",
                    item_id="item",
                    step=1,
                    target_text="x",
                    audio_path="a.wav",
                    scores={"total": 80},
                    features={},
                ),
            )
            append_progress_record(
                path,
                ProgressRecord(
                    user_id="u1",
                    item_id="item",
                    step=3,
                    target_text="x",
                    audio_path="b.wav",
                    scores={"total": 67},
                    features={},
                ),
            )
            rows = load_progress_records(path, user_id="u1", item_id="item")
            self.assertEqual(len(rows), 2)
            self.assertEqual(latest_record(rows, step=3)["audio_path"], "b.wav")

    def test_reference_dependency_gap(self) -> None:
        records = [
            {"item_id": "item", "step": 1, "scores": {"total": 88}},
            {"item_id": "item", "step": 3, "scores": {"total": 70}},
        ]
        gap = compute_reference_dependency_gap(records, item_id="item")
        self.assertEqual(gap.gap, 18.0)
        self.assertTrue(any("跟读" in item for item in gap.feedback))


if __name__ == "__main__":
    unittest.main()
