from __future__ import annotations

import json
from pathlib import Path

import scripts.run_free_speech_promotion_readiness_v10 as runner


def _patch_machine_stage(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "validate_manifest_file", lambda path: {"ok": True, "schema": "free_speech_sample_manifest_v1"})
    batch = tmp_path / "machine.jsonl"
    batch.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "run_acceptance",
        lambda *args, **kwargs: {
            "schema": "free_speech_acceptance_v8",
            "batch_jsonl": str(batch),
            "frozen_product_conditions": {"transcript": None, "scoring_used_gold_transcript": False},
        },
    )
    monkeypatch.setattr(
        runner,
        "export_v10_evidence",
        lambda input_jsonl, output_csv: {
            "schema": "free_speech_v10_evidence_export_v1",
            "sample_count": 12,
            "gold_transcript_used_for_duration_bucket": False,
        },
    )


def test_runner_stops_cleanly_at_human_rating_boundary(tmp_path: Path, monkeypatch) -> None:
    _patch_machine_stage(monkeypatch, tmp_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("placeholder\n", encoding="utf-8")

    report = runner.run(manifest, tmp_path / "out")

    assert report["stage"] == "awaiting_human_ratings"
    assert report["promotion_analysis"] is None
    assert report["decision"] is None
    assert report["product_score_changed"] is False
    assert report["score_contract_changed"] is False
    assert report["gold_transcript_used_for_scoring"] is False
    saved = json.loads(Path(report["summary_json"]).read_text(encoding="utf-8"))
    assert saved["stage"] == "awaiting_human_ratings"


def test_runner_only_reports_promotion_after_normalized_human_ratings(tmp_path: Path, monkeypatch) -> None:
    _patch_machine_stage(monkeypatch, tmp_path)
    manifest = tmp_path / "manifest.csv"
    ratings = tmp_path / "ratings.csv"
    manifest.write_text("placeholder\n", encoding="utf-8")
    ratings.write_text("placeholder\n", encoding="utf-8")

    def fake_normalize(ratings_csv, manifest_csv, output_path):
        Path(output_path).write_text("sample_id,criterion,human_rating\n", encoding="utf-8")
        return {"normalized_rating_count": 100}

    monkeypatch.setattr(runner, "normalize_file", fake_normalize)
    monkeypatch.setattr(
        runner,
        "analyze_v10",
        lambda *args, **kwargs: {
            "schema": "free_speech_promotion_analysis_v10",
            "overall_v10_promotion_readiness": "pass",
            "product_score_changed": False,
        },
    )

    report = runner.run(manifest, tmp_path / "out", completed_ratings_csv=ratings)

    assert report["stage"] == "promotion_readiness_evaluated"
    assert report["decision"] == "pass"
    assert report["promotion_analysis"]["product_score_changed"] is False
    assert report["score_contract_changed"] is False
