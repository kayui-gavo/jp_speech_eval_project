from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.free_speech_collection_server_v10 import CollectionStore, _safe_relpath


def _write_assignments(path: Path, *, relpath: str = "held/learner/HL01/HL01_01.wav") -> None:
    rows = [
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
            "collection_instruction": "短く自然に答えてください。",
            "audio_relpath": relpath,
            "status": "needed",
            "assignment_schema": "free_speech_collection_assignment_v10",
        }
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _wav_fixture() -> bytes:
    # Minimal RIFF/WAVE-shaped payload accepted by the local collection boundary.
    payload = bytearray(44)
    payload[0:4] = b"RIFF"
    payload[8:12] = b"WAVE"
    return bytes(payload)


def test_safe_relpath_rejects_absolute_traversal_and_non_wav() -> None:
    assert _safe_relpath("held/learner/HL01/HL01_01.wav") == Path("held/learner/HL01/HL01_01.wav")
    for bad in ("../escape.wav", "/tmp/escape.wav", "held/../../escape.wav", "held/file.webm"):
        with pytest.raises(ValueError):
            _safe_relpath(bad)


def test_participant_payload_hides_split_group_and_filesystem_path(tmp_path: Path) -> None:
    assignments = tmp_path / "assignments.csv"
    _write_assignments(assignments)
    store = CollectionStore(assignments, tmp_path / "audio")

    payload = store.speaker_payload("HL01")

    assert payload["speaker_id"] == "HL01"
    assert payload["recorded_count"] == 0
    assert payload["analysis_labels_exposed"] is False
    assert payload["response_transcript_requested"] is False
    item = payload["assignments"][0]
    assert item["sample_id"] == "HL01_01"
    assert "speaker_group" not in item
    assert "split" not in item
    assert "audio_relpath" not in item
    assert "response_transcript" not in item


def test_store_only_writes_preassigned_wav_and_refuses_silent_overwrite(tmp_path: Path) -> None:
    assignments = tmp_path / "assignments.csv"
    _write_assignments(assignments)
    root = tmp_path / "audio"
    store = CollectionStore(assignments, root)

    target = store.save_wav("HL01_01", _wav_fixture())

    assert target == (root / "held/learner/HL01/HL01_01.wav").resolve()
    assert target.read_bytes() == _wav_fixture()
    assert store.speaker_payload("HL01")["recorded_count"] == 1
    with pytest.raises(FileExistsError):
        store.save_wav("HL01_01", _wav_fixture())
    with pytest.raises(KeyError):
        store.save_wav("UNKNOWN", _wav_fixture())


def test_store_rejects_non_wav_payload(tmp_path: Path) -> None:
    assignments = tmp_path / "assignments.csv"
    _write_assignments(assignments)
    store = CollectionStore(assignments, tmp_path / "audio")

    with pytest.raises(ValueError):
        store.save_wav("HL01_01", b"not a wav" * 8)


def test_assignment_path_cannot_escape_audio_root(tmp_path: Path) -> None:
    assignments = tmp_path / "assignments.csv"
    _write_assignments(assignments, relpath="../../escape.wav")
    with pytest.raises(ValueError):
        CollectionStore(assignments, tmp_path / "audio")
