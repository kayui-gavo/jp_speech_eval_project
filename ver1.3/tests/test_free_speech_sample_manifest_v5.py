from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_free_speech_listener_pack_v3 import build_listener_pack
from scripts.validate_free_speech_sample_manifest import REQUIRED_COLUMNS, validate_manifest_rows


def _row(sample_id: str = "s1", **overrides):
    row = {
        "sample_id": sample_id,
        "audio_path": f"audio/{sample_id}.wav",
        "speaker_id": "spk1",
        "speaker_group": "learner",
        "l1": "zh",
        "task_mode": "controlled_dialogue",
        "prompt_id": "p1",
        "split": "held",
        "expected_language": "ja",
        "channel_condition": "clean",
        "channel_pair_id": "",
        "context_type": "preceding_turn",
        "context_id": "ctx1",
        "context_text": "昨日はどこへ行きましたか。",
        "context_audio_path": "",
        "source_note": "private provenance",
    }
    row.update(overrides)
    return row


def _write_manifest(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_rejects_speaker_split_leakage():
    rows = [
        _row("s1", split="development"),
        _row("s2", split="held"),
    ]
    report = validate_manifest_rows(rows, list(REQUIRED_COLUMNS))
    assert report["ok"] is False
    assert any(item.startswith("speaker_split_leakage:spk1") for item in report["errors"])


def test_channel_pairs_must_hold_learner_content_constant():
    rows = [
        _row("s1", channel_pair_id="pair1", channel_condition="clean"),
        _row("s2", channel_pair_id="pair1", channel_condition="moderate_noise", prompt_id="different"),
    ]
    report = validate_manifest_rows(rows, list(REQUIRED_COLUMNS))
    assert report["ok"] is False
    assert "channel_pair:pair1:conflicting_prompt_id" in report["errors"]


def test_listener_pack_separates_isolated_and_contextual_ratings_and_blinds_metadata(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [_row()])
    listener_rows, asset_rows, report = build_listener_pack(
        manifest,
        rater_ids=["r1", "r2", "r3"],
        ratings_per_presentation=2,
    )
    assert report["japanese_sample_count"] == 1
    assert len(listener_rows) == 4
    isolated = [row for row in listener_rows if row["presentation_variant"] == "isolated"]
    contextual = [row for row in listener_rows if row["presentation_variant"] == "contextual"]
    assert len(isolated) == 2
    assert len(contextual) == 2
    assert all(row["context_text"] == "" for row in isolated)
    assert all(row["context_presented_yes_no"] == "no" for row in isolated)
    assert all(row["assigned_constructs"].endswith("intonation_utterance_naturalness") for row in isolated)
    assert all(row["context_text"] == "昨日はどこへ行きましたか。" for row in contextual)
    assert all(row["assigned_constructs"] == "intonation_contextual_appropriateness" for row in contextual)
    for row in listener_rows:
        assert "speaker_group" not in row
        assert "l1" not in row
        assert "channel_condition" not in row
        assert "source_note" not in row
        assert "source_audio_path" not in row
    assert asset_rows[0]["private_only"] == "true"
    assert asset_rows[0]["source_audio_path"] == "audio/s1.wav"
