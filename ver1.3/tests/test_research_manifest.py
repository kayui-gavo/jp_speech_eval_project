from __future__ import annotations

from pathlib import Path

from scripts.validate_research_manifest import REQUIRED_COLUMNS, validate_manifest_rows


def _row(**overrides):
    row = {
        "sample_id": "s1",
        "audio_path": "audio/s1.wav",
        "speaker_id": "spk1",
        "l1": "zh",
        "proficiency": "intermediate",
        "target_id": "t1",
        "target_text": "ラーメンをください",
        "subset": "sentence",
        "task": "fixed_reading",
        "criterion": "pronunciation_accuracy",
        "rater_id": "r1",
        "human_rating": "5",
        "condition": "clean",
    }
    row.update(overrides)
    return row


def _fields():
    return list(_row().keys())


def test_manifest_validator_accepts_multiple_raters_for_same_sample():
    report = validate_manifest_rows(
        [_row(), _row(rater_id="r2", human_rating="6")],
        _fields(),
    )
    assert report["ok"] is True
    assert report["row_count"] == 2
    assert report["unique_sample_count"] == 1
    assert report["criteria"] == {"pronunciation_accuracy": 2}


def test_manifest_validator_rejects_duplicate_sample_criterion_rater():
    report = validate_manifest_rows([_row(), _row()], _fields())
    assert report["ok"] is False
    assert any("duplicate_sample_criterion_rater" in error for error in report["errors"])


def test_manifest_validator_rejects_missing_column_and_nonfinite_rating():
    missing = validate_manifest_rows([_row()], [c for c in _fields() if c != "criterion"])
    assert missing["ok"] is False
    assert "criterion" in missing["missing_required_columns"]

    bad_rating = validate_manifest_rows([_row(human_rating="nan")], _fields())
    assert bad_rating["ok"] is False
    assert any("invalid_human_rating" in error for error in bad_rating["errors"])


def test_manifest_audio_existence_check_is_optional(tmp_path: Path):
    row = _row(audio_path="audio/s1.wav")
    unchecked = validate_manifest_rows([row], _fields(), manifest_dir=tmp_path)
    assert unchecked["ok"] is True

    checked_missing = validate_manifest_rows(
        [row], _fields(), manifest_dir=tmp_path, check_audio_exists=True
    )
    assert checked_missing["ok"] is False

    audio = tmp_path / "audio" / "s1.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"not-decoded-by-validator")
    checked = validate_manifest_rows(
        [row], _fields(), manifest_dir=tmp_path, check_audio_exists=True
    )
    assert checked["ok"] is True


def test_required_schema_keeps_construct_and_target_identifiers():
    assert "criterion" in REQUIRED_COLUMNS
    assert "speaker_id" in REQUIRED_COLUMNS
    assert "target_id" in REQUIRED_COLUMNS
