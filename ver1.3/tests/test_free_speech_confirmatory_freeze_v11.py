from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from free_speech_confirmatory_freeze_v11 import build_freeze, verify_freeze


FIELDS = [
    "sample_id",
    "audio_path",
    "speaker_id",
    "speaker_group",
    "l1",
    "task_mode",
    "prompt_id",
    "split",
    "expected_language",
    "channel_condition",
    "channel_pair_id",
    "source_recording_id",
    "context_type",
    "context_id",
    "context_text",
    "context_audio_path",
    "source_note",
]


def _row(sample_id: str, audio: str, speaker: str, split: str, source: str) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "audio_path": audio,
        "speaker_id": speaker,
        "speaker_group": "learner",
        "l1": "zh",
        "task_mode": "spontaneous",
        "prompt_id": f"prompt_{sample_id}",
        "split": split,
        "expected_language": "ja",
        "channel_condition": "clean",
        "channel_pair_id": "",
        "source_recording_id": source,
        "context_type": "prompt",
        "context_id": f"ctx_{sample_id}",
        "context_text": "日本語で最近の出来事を話してください。",
        "context_audio_path": "",
        "source_note": "self_collected",
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _protocols(tmp_path: Path) -> list[Path]:
    p1 = tmp_path / "free_speech_v5_promotion_protocol.json"
    p2 = tmp_path / "free_speech_v10_consumer_promotion_protocol.json"
    p1.write_text(json.dumps({"schema": "v5", "rho": 0.3}), encoding="utf-8")
    p2.write_text(json.dumps({"schema": "v10", "learner_pairs": 30}), encoding="utf-8")
    return [p1, p2]


def test_freeze_is_reproducible_and_verifies(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "dev.wav").write_bytes(b"RIFF-development-audio")
    (audio / "held.wav").write_bytes(b"RIFF-held-audio")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _row("dev_1", "dev.wav", "learner_dev", "development", "src_dev"),
            _row("held_1", "held.wav", "learner_held", "held", "src_held"),
        ],
    )
    protocols = _protocols(tmp_path)

    frozen = build_freeze(manifest, audio_root=audio, protocol_paths=protocols)
    assert frozen["freeze_ok"] is True
    assert frozen["scientific_lock"]["held_set_must_not_be_used_for_threshold_tuning"] is True
    assert len(frozen["freeze_fingerprint_sha256"]) == 64

    freeze_json = tmp_path / "freeze.json"
    freeze_json.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    verified = verify_freeze(freeze_json)
    assert verified["verification_ok"] is True
    assert verified["drift_detected"] is False


def test_audio_byte_change_is_detected_as_drift(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "dev.wav").write_bytes(b"dev-audio")
    (audio / "held.wav").write_bytes(b"held-audio-v1")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _row("dev_1", "dev.wav", "speaker_dev", "development", "src_dev"),
            _row("held_1", "held.wav", "speaker_held", "held", "src_held"),
        ],
    )
    protocols = _protocols(tmp_path)
    frozen = build_freeze(manifest, audio_root=audio, protocol_paths=protocols)
    freeze_json = tmp_path / "freeze.json"
    freeze_json.write_text(json.dumps(frozen), encoding="utf-8")

    (audio / "held.wav").write_bytes(b"held-audio-v2")
    verified = verify_freeze(freeze_json)
    assert verified["verification_ok"] is False
    assert verified["drift_detected"] is True


def test_identical_audio_bytes_across_development_and_held_fail_closed(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    duplicate = b"same-underlying-audio"
    (audio / "dev.wav").write_bytes(duplicate)
    (audio / "held.wav").write_bytes(duplicate)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _row("dev_1", "dev.wav", "speaker_dev", "development", "src_dev"),
            _row("held_1", "held.wav", "speaker_held", "held", "src_held"),
        ],
    )

    frozen = build_freeze(manifest, audio_root=audio, protocol_paths=_protocols(tmp_path))
    assert frozen["freeze_ok"] is False
    assert "duplicate_audio_bytes_across_splits" in frozen["integrity"]["problems"]


def test_source_recording_id_must_not_cross_split_even_with_different_files(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "dev.wav").write_bytes(b"dev-version")
    (audio / "held.wav").write_bytes(b"held-version")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _row("dev_1", "dev.wav", "speaker_dev", "development", "shared_source"),
            _row("held_1", "held.wav", "speaker_held", "held", "shared_source"),
        ],
    )

    frozen = build_freeze(manifest, audio_root=audio, protocol_paths=_protocols(tmp_path))
    assert frozen["freeze_ok"] is False
    assert frozen["integrity"]["source_recording_split_leakage"] == ["shared_source"]
