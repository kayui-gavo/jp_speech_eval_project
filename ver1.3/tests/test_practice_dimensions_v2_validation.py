from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rows() -> dict[str, dict[str, str]]:
    path = ROOT / "data/calibration_candidates/practice_dimensions_v2_summary.csv"
    with path.open(encoding="utf-8") as stream:
        return {row["metric"]: row for row in csv.DictReader(stream)}


def test_clarity_proxy_responds_to_recording_degradation() -> None:
    rows = _rows()
    assert float(rows["pronunciation_clarity_normal"]["mean"]) > float(rows["pronunciation_clarity_5db_noise"]["mean"]) + 10
    assert float(rows["pronunciation_clarity_normal"]["mean"]) > float(rows["pronunciation_clarity_low_gain"]["mean"]) + 5


def test_rhythm_separates_jitter_and_does_not_reward_equal_fallback() -> None:
    rows = _rows()
    assert float(rows["rhythm_normal"]["mean"]) > float(rows["rhythm_timing_jitter"]["mean"]) + 20
    assert float(rows["rhythm_equal_fallback"]["mean"]) < 80
    assert float(rows["paired_rhythm_normal_vs_rhythm_timing_jitter"]["roc_auc"]) >= 0.95


def test_fluency_has_resolution_and_separates_controls() -> None:
    rows = _rows()
    normal = float(rows["fluency_normal"]["mean"])
    assert normal > float(rows["fluency_slow"]["mean"]) + 8
    assert normal > float(rows["fluency_fast"]["mean"]) + 8
    assert normal > float(rows["fluency_hesitation"]["mean"]) + 25
    assert float(rows["janon_learner_fluency_external"]["ceiling_rate"]) == 0.0


def test_report_keeps_scientific_limits_explicit() -> None:
    report = (ROOT / "reports/practice_dimensions_v2_validation.md").read_text(encoding="utf-8")
    assert "not phone-level GOP or phoneme correctness" in report
    assert "Special-mora user feedback is safe by default" in report
    assert "not strict pitch-accent correctness" in report
