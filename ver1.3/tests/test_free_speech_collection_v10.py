from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.materialize_free_speech_manifest_v10 import materialize
from scripts.prepare_free_speech_collection_v10 import build_assignments


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_collection_plan_generates_balanced_disjoint_assignments() -> None:
    plan = json.loads(Path("data/human_eval/free_speech_v10_minimum_collection_plan.json").read_text())
    assignments, speakers, report = build_assignments(plan)

    assert report["assignment_count"] == 80
    assert report["speaker_count"] == 18
    assert report["development_speaker_count"] == 6
    assert report["held_speaker_count"] == 12
    assert report["speaker_split_overlap_count"] == 0
    assert report["planned_duration_bucket_counts"] == {"short": 44, "long": 36}
    assert report["task_mode_counts"] == {"spontaneous": 44, "controlled_dialogue": 36}
    assert report["participant_metadata_fabricated"] is False
    assert report["response_transcript_requested"] is False

    development_ids = {row["speaker_id"] for row in speakers if row["split"] == "development"}
    held_ids = {row["speaker_id"] for row in speakers if row["split"] == "held"}
    assert development_ids.isdisjoint(held_ids)
    assert all("transcript" not in row for row in assignments)
    assert all(row["audio_relpath"].endswith(".wav") for row in assignments)


def test_materializer_emits_only_real_files_and_keeps_gold_transcript_out(tmp_path: Path) -> None:
    assignments = [
        {
            "sample_id": "HL01_01",
            "speaker_id": "HL01",
            "speaker_group": "learner",
            "split": "held",
            "task_mode": "spontaneous",
            "planned_duration_bucket": "short",
            "prompt_id": "short_spontaneous_p01",
            "context_type": "prompt",
            "context_text": "今日の気分を一言で教えてください。",
            "collection_instruction": "answer naturally",
            "audio_relpath": "held/learner/HL01/HL01_01.wav",
            "status": "recorded",
            "assignment_schema": "free_speech_collection_assignment_v10",
        },
        {
            "sample_id": "HN01_01",
            "speaker_id": "HN01",
            "speaker_group": "native",
            "split": "held",
            "task_mode": "controlled_dialogue",
            "planned_duration_bucket": "long",
            "prompt_id": "dlg_followup_long",
            "context_type": "preceding_turn",
            "context_text": "どうしてそう思ったんですか。",
            "collection_instruction": "answer naturally with detail",
            "audio_relpath": "held/native/HN01/HN01_01.wav",
            "status": "needed",
            "assignment_schema": "free_speech_collection_assignment_v10",
        },
    ]
    assignment_csv = tmp_path / "assignments.csv"
    _write_csv(assignment_csv, assignments)

    metadata = [
        {
            "speaker_id": "HL01", "speaker_group": "learner", "split": "held", "l1": "zh",
            "japanese_proficiency": "N2", "recording_device": "phone",
            "consent_or_dataset_provenance": "consented_test", "notes": "", "metadata_schema": "v10",
        },
        {
            "speaker_id": "HN01", "speaker_group": "native", "split": "held", "l1": "ja",
            "japanese_proficiency": "", "recording_device": "phone",
            "consent_or_dataset_provenance": "consented_test", "notes": "", "metadata_schema": "v10",
        },
    ]
    metadata_csv = tmp_path / "speakers.csv"
    _write_csv(metadata_csv, metadata)

    audio_root = tmp_path / "audio"
    first_audio = audio_root / assignments[0]["audio_relpath"]
    first_audio.parent.mkdir(parents=True, exist_ok=True)
    first_audio.write_bytes(b"RIFFfixture")

    manifest = tmp_path / "manifest.csv"
    report = materialize(
        assignment_csv,
        audio_root,
        manifest,
        speaker_metadata_csv=metadata_csv,
        require_all=False,
    )

    assert report["materialized_sample_count"] == 1
    assert report["missing_recording_count"] == 1
    assert report["missing_sample_ids"] == ["HN01_01"]
    assert report["gold_transcript_added"] is False
    assert report["manifest_validation"]["ok"] is True

    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    assert rows[0]["audio_path"] == assignments[0]["audio_relpath"]
    assert rows[0]["source_recording_id"] == "HL01_01"
    assert rows[0]["expected_language"] == "ja"
    assert "transcript" not in rows[0]


def test_materializer_can_fail_closed_when_collection_is_incomplete(tmp_path: Path) -> None:
    assignments = [
        {
            "sample_id": "HL01_01", "speaker_id": "HL01", "speaker_group": "learner", "split": "held",
            "task_mode": "spontaneous", "planned_duration_bucket": "short", "prompt_id": "p1",
            "context_type": "prompt", "context_text": "質問です。", "collection_instruction": "",
            "audio_relpath": "held/learner/HL01/HL01_01.wav", "status": "needed", "assignment_schema": "v10",
        }
    ]
    assignment_csv = tmp_path / "assignments.csv"
    _write_csv(assignment_csv, assignments)

    with pytest.raises(FileNotFoundError):
        materialize(assignment_csv, tmp_path / "audio", tmp_path / "manifest.csv", require_all=True)
