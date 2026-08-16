from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.run_free_speech_validation_batch import run_batch
from scripts.validate_free_speech_sample_manifest import REQUIRED_COLUMNS


def _manifest_row(sample_id: str):
    return {
        "sample_id": sample_id,
        "audio_path": f"audio/{sample_id}.wav",
        "speaker_id": f"spk_{sample_id}",
        "speaker_group": "learner",
        "l1": "zh",
        "task_mode": "spontaneous",
        "prompt_id": f"prompt_{sample_id}",
        "split": "held",
        "expected_language": "ja",
        "channel_condition": "clean",
        "channel_pair_id": "",
        "source_recording_id": f"source_{sample_id}",
        "context_type": "prompt",
        "context_id": f"ctx_{sample_id}",
        "context_text": "最近あった出来事について話してください。",
        "context_audio_path": "",
        "source_note": "test",
    }


def _write_manifest(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def test_batch_never_passes_oracle_transcript_and_is_resumable(tmp_path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "batch.jsonl"
    _write_manifest(manifest, [_manifest_row("s1"), _manifest_row("s2")])
    calls = []

    def fake_evaluator(wav_path, transcript=None, asr_model="small", asr_provider="auto"):
        calls.append({"wav": str(wav_path), "transcript": transcript, "model": asr_model, "provider": asr_provider})
        return {
            "fluency_score": 75,
            "details": {
                "mode": "transcript_assisted_light",
                "language_gate": {"eligible": True, "reason": "ja"},
                "recording_quality": {"score": 0.9},
                "shadow": {},
            },
        }

    def fake_policy(raw, mode="transcript_assisted_light"):
        return {
            "score_available": True,
            "display_score": 74,
            "score_contract_version": "consumer_four_score_v2",
            "component_scores": {
                "clarity": {"value": 70, "evidence_state": "neutral_prior"},
                "mora_timing": {"value": 70, "evidence_state": "neutral_prior"},
                "delivery_fluency": {"value": 82, "evidence_state": "broad_proxy"},
                "intonation": {"value": 70, "evidence_state": "neutral_prior"},
            },
        }

    first = run_batch(
        manifest,
        output,
        audio_root=tmp_path,
        evaluator=fake_evaluator,
        user_policy=fake_policy,
    )
    assert first["written_ok"] == 2
    assert first["scoring_used_gold_transcript"] is False
    assert first["partial_evidence_candidate_user_facing"] is False
    assert first["partial_evidence_candidate_product_score_changed"] is False
    assert len(calls) == 2
    assert all(call["transcript"] is None for call in calls)

    second = run_batch(
        manifest,
        output,
        audio_root=tmp_path,
        evaluator=fake_evaluator,
        user_policy=fake_policy,
    )
    assert second["written_ok"] == 0
    assert second["skipped_completed"] == 2
    assert len(calls) == 2

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert all(row["scoring_used_gold_transcript"] is False for row in rows)
    assert all(row["metadata"]["speaker_group"] == "learner" for row in rows)
    assert {row["metadata"]["source_recording_id"] for row in rows} == {"source_s1", "source_s2"}
    for row in rows:
        candidate = row["score_candidates"]["partial_evidence_aggregate"]
        assert candidate["available"] is True
        assert candidate["user_facing"] is False
        assert candidate["product_score_changed"] is False
        assert candidate["neutral_prior_count"] == 3
        assert candidate["available_component_count"] == 1
        assert candidate["candidate_display_score"] != row["user_score"]["display_score"]


def test_batch_preserves_per_sample_failure_instead_of_aborting(tmp_path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "batch.jsonl"
    _write_manifest(manifest, [_manifest_row("s1"), _manifest_row("s2")])

    def fake_evaluator(wav_path, **kwargs):
        if str(wav_path).endswith("s1.wav"):
            raise RuntimeError("decode failed")
        return {"fluency_score": 70, "details": {"language_gate": {"eligible": True}, "recording_quality": {}, "shadow": {}}}

    report = run_batch(
        manifest,
        output,
        audio_root=tmp_path,
        evaluator=fake_evaluator,
        user_policy=lambda raw, mode: {"score_available": True},
        resume=False,
    )
    assert report["written_ok"] == 1
    assert report["failed"] == 1
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = next(row for row in rows if row["status"] == "ok")
    assert ok["score_candidates"]["partial_evidence_aggregate"]["available"] is False
    failed = next(row for row in rows if row["status"] == "error")
    assert failed["sample_id"] == "s1"
    assert failed["error_type"] == "RuntimeError"