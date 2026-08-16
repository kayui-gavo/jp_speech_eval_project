from __future__ import annotations

import json

from scripts.analyze_partial_evidence_aggregate import analyze_batch


def _row(sample_id: str, fluency: float, current: float, *, task: str = "spontaneous"):
    return {
        "sample_id": sample_id,
        "status": "ok",
        "metadata": {
            "task_mode": task,
            "speaker_group": "learner",
            "channel_condition": "clean",
        },
        "user_score": {
            "display_score": current,
            "component_scores": {
                "clarity": {"value": 70, "evidence_state": "neutral_prior"},
                "mora_timing": {"value": 70, "evidence_state": "neutral_prior"},
                "delivery_fluency": {"value": fluency, "evidence_state": "broad_proxy"},
                "intonation": {"value": 70, "evidence_state": "neutral_prior"},
            },
        },
    }


def test_analysis_recomputes_candidate_and_uses_latest_attempt(tmp_path) -> None:
    path = tmp_path / "batch.jsonl"
    rows = [
        _row("s1", 55, 66),
        {"sample_id": "s2", "status": "error", "error": "old failure"},
        _row("s2", 90, 75, task="dialogue_response"),
        # Latest retry for s1 wins and should replace the earlier 55-fluency row.
        _row("s1", 80, 72),
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    report = analyze_batch(path)
    assert report["raw_attempt_count"] == 4
    assert report["latest_sample_count"] == 2
    assert report["superseded_attempt_count"] == 2
    assert report["ok_latest_count"] == 2
    assert report["paired_score_count"] == 2
    assert report["candidate_available_rate"] == 1.0
    assert report["neutral_prior_count"]["mean"] == 3
    assert report["available_component_count"]["mean"] == 1
    assert report["current_display"]["n"] == 2
    assert report["partial_evidence_candidate"]["n"] == 2
    assert set(report["by_task_mode"]) == {"dialogue_response", "spontaneous"}
    assert report["decision"] == "telemetry_only_no_product_promotion"
    assert report["user_facing"] is False
    assert report["product_score_changed"] is False


def test_analysis_keeps_candidate_unavailable_when_every_dimension_is_placeholder(tmp_path) -> None:
    path = tmp_path / "batch.jsonl"
    row = {
        "sample_id": "s1",
        "status": "ok",
        "metadata": {"task_mode": "spontaneous", "speaker_group": "learner", "channel_condition": "clean"},
        "user_score": {
            "display_score": 70,
            "component_scores": {
                "clarity": {"value": 70, "evidence_state": "neutral_prior"},
                "mora_timing": {"value": 70, "evidence_state": "neutral_prior"},
                "delivery_fluency": {"value": None, "evidence_state": "unavailable"},
                "intonation": {"value": 70, "evidence_state": "neutral_prior"},
            },
        },
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    report = analyze_batch(path)
    assert report["ok_latest_count"] == 1
    assert report["paired_score_count"] == 0
    assert report["candidate_available_rate"] == 0.0
    assert report["partial_evidence_candidate"]["n"] == 0
