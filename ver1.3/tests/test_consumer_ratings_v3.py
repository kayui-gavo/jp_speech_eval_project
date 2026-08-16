from __future__ import annotations

import csv
from pathlib import Path

from scripts.normalize_consumer_ratings_v3 import normalize_file
from scripts.validate_consumer_ratings_v3 import REQUIRED_COLUMNS, validate_rating_rows
from scripts.validate_free_speech_sample_manifest import REQUIRED_COLUMNS as MANIFEST_COLUMNS


def _rating_row(**overrides):
    row = {
        "rater_id": "r1",
        "sample_id": "s1",
        "presentation_id": "p1",
        "presentation_variant": "isolated",
        "task_mode": "spontaneous",
        "audio_asset_id": "audio_x",
        "context_type": "none",
        "context_id": "",
        "context_text": "",
        "context_audio_asset_id": "",
        "assigned_constructs": "clarity_comprehensibility|fluency|rhythm_naturalness|intonation_utterance_naturalness",
        "analyzable_yes_no": "yes",
        "clarity_comprehensibility_1to7": "6",
        "fluency_1to7": "5",
        "rhythm_naturalness_1to7": "5",
        "intonation_utterance_naturalness_1to7": "4",
        "intonation_contextual_appropriateness_1to7": "",
        "context_presented_yes_no": "no",
        "timestamp": "2026-08-16T13:00:00+09:00",
    }
    row.update(overrides)
    return row


def _manifest_row(sample_id: str = "s1", **overrides):
    row = {
        "sample_id": sample_id,
        "audio_path": f"{sample_id}.wav",
        "speaker_id": "spk_hidden",
        "speaker_group": "learner",
        "l1": "zh",
        "task_mode": "spontaneous",
        "prompt_id": "prompt1",
        "split": "held",
        "expected_language": "ja",
        "channel_condition": "clean",
        "channel_pair_id": "pair1",
        "context_type": "prompt",
        "context_id": "ctx1",
        "context_text": "週末は何をしましたか。",
        "context_audio_path": "",
        "source_note": "private",
    }
    row.update(overrides)
    return row


def _write(path: Path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def test_contextual_intonation_cannot_be_collected_on_isolated_presentation():
    row = _rating_row(
        assigned_constructs="intonation_contextual_appropriateness",
        clarity_comprehensibility_1to7="",
        fluency_1to7="",
        rhythm_naturalness_1to7="",
        intonation_utterance_naturalness_1to7="",
        intonation_contextual_appropriateness_1to7="5",
    )
    report = validate_rating_rows([row], list(REQUIRED_COLUMNS))
    assert report["ok"] is False
    assert any("contextual_construct_on_isolated_presentation" in item for item in report["errors"])


def test_contextual_presentation_requires_actual_context_payload():
    row = _rating_row(
        presentation_variant="contextual",
        assigned_constructs="intonation_contextual_appropriateness",
        clarity_comprehensibility_1to7="",
        fluency_1to7="",
        rhythm_naturalness_1to7="",
        intonation_utterance_naturalness_1to7="",
        intonation_contextual_appropriateness_1to7="5",
        context_presented_yes_no="yes",
        context_type="preceding_turn",
    )
    report = validate_rating_rows([row], list(REQUIRED_COLUMNS))
    assert report["ok"] is False
    assert any("requires_actual_context_payload" in item for item in report["errors"])


def test_normalizer_reattaches_private_metadata_only_after_collection(tmp_path):
    ratings = tmp_path / "ratings.csv"
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "long.csv"
    _write(ratings, REQUIRED_COLUMNS, [_rating_row()])
    # A channel-pair id represents an actual same-content channel control.  The
    # manifest therefore contains both the clean anchor and one channel variant,
    # even though only s1 is rated in this unit test.
    _write(
        manifest,
        MANIFEST_COLUMNS,
        [
            _manifest_row("s1", channel_condition="clean"),
            _manifest_row("s2", channel_condition="low_level"),
        ],
    )
    report = normalize_file(ratings, manifest, output)
    assert report["normalized_rating_count"] == 4
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["criterion"] for row in rows} == {
        "clarity_comprehensibility",
        "fluency",
        "rhythm_naturalness",
        "intonation_utterance_naturalness",
    }
    assert all(row["speaker_id"] == "spk_hidden" for row in rows)
    assert all(row["speaker_group"] == "learner" for row in rows)
    assert all(row["condition"] == "clean" for row in rows)
    assert all(row["channel_pair_id"] == "pair1" for row in rows)
