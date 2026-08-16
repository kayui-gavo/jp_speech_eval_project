from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.audit_free_speech_audio_dataset_v11 import audit


FIELDS = [
    "sample_id", "audio_path", "speaker_id", "speaker_group", "l1", "task_mode",
    "prompt_id", "split", "expected_language", "channel_condition", "channel_pair_id",
    "source_recording_id", "context_type", "context_id", "context_text",
    "context_audio_path", "source_note",
]


def _row(
    sample_id: str,
    audio_path: str,
    speaker_id: str,
    split: str,
    *,
    source_recording_id: str | None = None,
    context_type: str = "none",
    context_id: str = "",
    context_text: str = "",
    context_audio_path: str = "",
    speaker_group: str = "learner",
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "audio_path": audio_path,
        "speaker_id": speaker_id,
        "speaker_group": speaker_group,
        "l1": "zh" if speaker_group == "learner" else "ja",
        "task_mode": "spontaneous",
        "prompt_id": "p01",
        "split": split,
        "expected_language": "ja",
        "channel_condition": "clean",
        "channel_pair_id": "",
        "source_recording_id": source_recording_id or sample_id,
        "context_type": context_type,
        "context_id": context_id,
        "context_text": context_text,
        "context_audio_path": context_audio_path,
        "source_note": "v11_fixture",
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _wav(path: Path, *, sr: int = 16000, channels: int = 1, freq: float = 220.0, amp: float = 0.1, duration: float = 1.0) -> None:
    t = np.arange(max(1, int(sr * duration)), dtype=np.float32) / float(sr)
    mono = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    data = mono if channels == 1 else np.stack([mono, mono * 0.8], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sr, subtype="PCM_16")


def test_exact_response_audio_across_development_and_held_is_hard_block(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _wav(audio / "dev.wav")
    (audio / "held.wav").write_bytes((audio / "dev.wav").read_bytes())
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _row("dev1", "dev.wav", "DL01", "development", source_recording_id="src_dev"),
            _row("held1", "held.wav", "HL01", "held", source_recording_id="src_held"),
        ],
    )

    report = audit(manifest, audio)

    assert report["decision"] == "collection_blocked"
    assert report["hard_blocker_code_counts"]["development_held_exact_audio_hash_overlap"] == 1
    assert report["automatic_sample_exclusion_performed"] is False
    assert report["product_score_changed"] is False


def test_same_source_recording_id_across_splits_is_hard_block_even_when_audio_differs(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _wav(audio / "dev.wav", freq=220.0)
    _wav(audio / "held.wav", freq=330.0)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _row("dev1", "dev.wav", "DL01", "development", source_recording_id="same_source"),
            _row("held1", "held.wav", "HL01", "held", source_recording_id="same_source"),
        ],
    )

    report = audit(manifest, audio)

    assert report["decision"] == "collection_blocked"
    assert report["hard_blocker_code_counts"]["development_held_source_recording_overlap"] == 1


def test_missing_and_undecodable_response_audio_are_hard_blockers(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "broken.wav").write_bytes(b"not-a-wave-file")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _row("missing", "missing.wav", "HL01", "held"),
            _row("broken", "broken.wav", "HL02", "held"),
        ],
    )

    report = audit(manifest, audio)

    assert report["decision"] == "collection_blocked"
    assert report["hard_blocker_code_counts"]["missing_response_audio"] == 1
    assert report["hard_blocker_code_counts"]["undecodable_response_audio"] == 1


def test_context_audio_reuse_across_splits_is_not_response_leakage(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _wav(audio / "dev_response.wav", freq=200.0)
    _wav(audio / "held_response.wav", freq=320.0)
    _wav(audio / "shared_context.wav", freq=440.0)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _row(
                "dev1", "dev_response.wav", "DL01", "development",
                context_type="preceding_turn", context_id="ctx_shared", context_audio_path="shared_context.wav",
            ),
            _row(
                "held1", "held_response.wav", "HL01", "held",
                context_type="preceding_turn", context_id="ctx_shared", context_audio_path="shared_context.wav",
            ),
        ],
    )

    report = audit(manifest, audio)

    assert report["decision"] in {"collection_ready", "collection_ready_with_review"}
    assert report["hard_blocker_count"] == 0
    assert report["context_audio_hashed_for_response_leakage"] is False


def test_stereo_mixed_sample_rates_and_clipping_are_review_only(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _wav(audio / "a.wav", sr=16000, channels=2, freq=200.0, amp=1.0)
    _wav(audio / "b.wav", sr=22050, channels=1, freq=330.0, amp=0.1)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _row("a", "a.wav", "HL01", "held"),
            _row("b", "b.wav", "HL02", "held"),
        ],
    )

    report = audit(manifest, audio)

    assert report["decision"] == "collection_ready_with_review"
    assert report["hard_blocker_count"] == 0
    assert report["review_code_counts"]["non_mono_response_audio"] == 1
    assert report["review_code_counts"]["mixed_sample_rates"] == 1
    assert report["review_code_counts"]["clipping_fraction_high"] >= 1
    assert report["format_summary"]["sample_rates"] == [16000, 22050]
    assert report["automatic_sample_exclusion_performed"] is False


def test_undeclared_byte_identical_samples_inside_one_split_are_blocked(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _wav(audio / "a.wav")
    (audio / "b.wav").write_bytes((audio / "a.wav").read_bytes())
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _row("a", "a.wav", "HL01", "held", source_recording_id="src_a"),
            _row("b", "b.wav", "HL02", "held", source_recording_id="src_b"),
        ],
    )

    report = audit(manifest, audio)

    assert report["decision"] == "collection_blocked"
    assert report["hard_blocker_code_counts"]["undeclared_exact_duplicate_response"] == 1


def test_declared_shared_source_identical_audio_is_review_not_silent_independence(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _wav(audio / "a.wav")
    (audio / "b.wav").write_bytes((audio / "a.wav").read_bytes())
    manifest = tmp_path / "manifest.csv"
    rows = [
        _row("a", "a.wav", "HL01", "held", source_recording_id="src_shared"),
        _row("b", "b.wav", "HL01", "held", source_recording_id="src_shared"),
    ]
    rows[0]["channel_pair_id"] = "pair1"
    rows[1]["channel_pair_id"] = "pair1"
    rows[1]["channel_condition"] = "derived_codec"
    _write_manifest(manifest, rows)

    report = audit(manifest, audio)

    assert "undeclared_exact_duplicate_response" not in report["hard_blocker_code_counts"]
    assert report["review_code_counts"]["declared_shared_source_has_byte_identical_audio"] == 1
    assert report["automatic_sample_exclusion_performed"] is False
