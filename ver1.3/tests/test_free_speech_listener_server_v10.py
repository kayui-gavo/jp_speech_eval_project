from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.free_speech_listener_server_v10 import ListenerStore, _safe_relpath


LISTENER_FIELDS = [
    "rater_id", "sample_id", "presentation_id", "presentation_variant", "task_mode",
    "audio_asset_id", "context_type", "context_id", "context_text", "context_audio_asset_id",
    "assigned_constructs", "analyzable_yes_no", "clarity_comprehensibility_1to7",
    "fluency_1to7", "rhythm_naturalness_1to7", "intonation_utterance_naturalness_1to7",
    "intonation_contextual_appropriateness_1to7", "context_presented_yes_no", "timestamp", "pack_schema",
]


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[ListenerStore, Path]:
    listener = tmp_path / "listener.csv"
    assets = tmp_path / "assets.csv"
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    (audio_root / "held.wav").write_bytes(b"RIFFxxxxWAVEfixture")
    rows = [
        {
            "rater_id": "r01", "sample_id": "secret_sample", "presentation_id": "pres_iso",
            "presentation_variant": "isolated", "task_mode": "spontaneous", "audio_asset_id": "audio_a",
            "context_type": "none", "context_id": "", "context_text": "", "context_audio_asset_id": "",
            "assigned_constructs": "clarity_comprehensibility|fluency|rhythm_naturalness|intonation_utterance_naturalness",
            "analyzable_yes_no": "", "clarity_comprehensibility_1to7": "", "fluency_1to7": "",
            "rhythm_naturalness_1to7": "", "intonation_utterance_naturalness_1to7": "",
            "intonation_contextual_appropriateness_1to7": "", "context_presented_yes_no": "no",
            "timestamp": "", "pack_schema": "consumer_listener_pack_v3",
        },
        {
            "rater_id": "r01", "sample_id": "secret_sample", "presentation_id": "pres_ctx",
            "presentation_variant": "contextual", "task_mode": "controlled_dialogue", "audio_asset_id": "audio_a",
            "context_type": "preceding_turn", "context_id": "ctx1", "context_text": "どうしてそう思ったんですか。",
            "context_audio_asset_id": "", "assigned_constructs": "intonation_contextual_appropriateness",
            "analyzable_yes_no": "", "clarity_comprehensibility_1to7": "", "fluency_1to7": "",
            "rhythm_naturalness_1to7": "", "intonation_utterance_naturalness_1to7": "",
            "intonation_contextual_appropriateness_1to7": "", "context_presented_yes_no": "yes",
            "timestamp": "", "pack_schema": "consumer_listener_pack_v3",
        },
    ]
    _write_csv(listener, rows, LISTENER_FIELDS)
    _write_csv(
        assets,
        [{
            "sample_id": "secret_sample", "audio_asset_id": "audio_a", "source_audio_path": "held.wav",
            "context_audio_asset_id": "", "source_context_audio_path": "", "private_only": "true",
        }],
    )
    store = ListenerStore(listener, assets, audio_root, tmp_path / "responses")
    return store, listener


def test_participant_payload_stays_blinded_and_uses_schema_instructions(tmp_path: Path) -> None:
    store, _ = _fixture(tmp_path)
    payload = store.participant_payload("r01")

    assert payload["analysis_metadata_exposed"] is False
    assert payload["target_transcript_exposed"] is False
    assert payload["source_paths_exposed"] is False
    assert payload["constructs"]["clarity_comprehensibility"]["instruction_ja"].startswith("この発話は")
    assert payload["scale"]["1"]
    item = payload["presentations"][0]
    assert "sample_id" not in item
    assert "speaker_group" not in item
    assert "l1" not in item
    assert "source_audio_path" not in item
    assert item["context_presented"] is False
    contextual = payload["presentations"][1]
    assert contextual["context_presented"] is True
    assert contextual["context_text"] == "どうしてそう思ったんですか。"


def test_analyzable_response_requires_exact_assigned_1to7_ratings(tmp_path: Path) -> None:
    store, _ = _fixture(tmp_path)
    with pytest.raises(ValueError):
        store.save_response("pres_iso", {"analyzable_yes_no": "yes", "ratings": {"fluency": 5}})
    with pytest.raises(ValueError):
        store.save_response(
            "pres_iso",
            {
                "analyzable_yes_no": "yes",
                "ratings": {
                    "clarity_comprehensibility": 5,
                    "fluency": 5,
                    "rhythm_naturalness": 8,
                    "intonation_utterance_naturalness": 5,
                },
            },
        )


def test_unanalyzable_response_must_not_carry_construct_scores(tmp_path: Path) -> None:
    store, _ = _fixture(tmp_path)
    with pytest.raises(ValueError):
        store.save_response("pres_ctx", {"analyzable_yes_no": "no", "ratings": {"intonation_contextual_appropriateness": 4}})
    store.save_response("pres_ctx", {"analyzable_yes_no": "no", "ratings": {}})
    assert store.completed("pres_ctx") is True


def test_export_round_trips_into_existing_v3_rating_validator(tmp_path: Path) -> None:
    store, _ = _fixture(tmp_path)
    store.save_response(
        "pres_iso",
        {
            "analyzable_yes_no": "yes",
            "ratings": {
                "clarity_comprehensibility": 5,
                "fluency": 4,
                "rhythm_naturalness": 6,
                "intonation_utterance_naturalness": 5,
            },
        },
    )
    store.save_response(
        "pres_ctx",
        {"analyzable_yes_no": "yes", "ratings": {"intonation_contextual_appropriateness": 6}},
    )
    out = tmp_path / "completed.csv"
    report = store.export_completed(out, require_complete=True)

    assert report["completed_presentation_count"] == 2
    assert report["missing_presentation_count"] == 0
    assert report["validation"]["ok"] is True
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    isolated = next(row for row in rows if row["presentation_id"] == "pres_iso")
    contextual = next(row for row in rows if row["presentation_id"] == "pres_ctx")
    assert isolated["clarity_comprehensibility_1to7"] == "5"
    assert isolated["intonation_contextual_appropriateness_1to7"] == ""
    assert contextual["intonation_contextual_appropriateness_1to7"] == "6"
    assert contextual["clarity_comprehensibility_1to7"] == ""


def test_export_can_report_partial_progress_without_fabricating_missing_rows(tmp_path: Path) -> None:
    store, _ = _fixture(tmp_path)
    store.save_response("pres_ctx", {"analyzable_yes_no": "no", "ratings": {}})
    out = tmp_path / "partial.csv"
    report = store.export_completed(out, require_complete=False)
    assert report["completed_presentation_count"] == 1
    assert report["missing_presentation_count"] == 1
    with pytest.raises(ValueError):
        store.export_completed(tmp_path / "final.csv", require_complete=True)


def test_listener_asset_paths_reject_traversal() -> None:
    assert _safe_relpath("held/learner/a.wav") == Path("held/learner/a.wav")
    for bad in ("../secret.wav", "/tmp/secret.wav", "held/../../secret.wav"):
        with pytest.raises(ValueError):
            _safe_relpath(bad)
