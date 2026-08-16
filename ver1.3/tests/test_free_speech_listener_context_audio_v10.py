from __future__ import annotations

import csv
from pathlib import Path

from scripts.free_speech_listener_server_v10 import HTML, ListenerStore


LISTENER_FIELDS = [
    "rater_id", "sample_id", "presentation_id", "presentation_variant", "task_mode",
    "audio_asset_id", "context_type", "context_id", "context_text", "context_audio_asset_id",
    "assigned_constructs", "analyzable_yes_no", "clarity_comprehensibility_1to7",
    "fluency_1to7", "rhythm_naturalness_1to7", "intonation_utterance_naturalness_1to7",
    "intonation_contextual_appropriateness_1to7", "context_presented_yes_no", "timestamp", "pack_schema",
]


def _write(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_contextual_presentation_exposes_anonymous_context_asset_and_ui_player(tmp_path: Path) -> None:
    listener = tmp_path / "listener.csv"
    assets = tmp_path / "assets.csv"
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    (audio_root / "target.wav").write_bytes(b"RIFFxxxxWAVEtarget")
    (audio_root / "context.wav").write_bytes(b"RIFFxxxxWAVEcontext")

    _write(
        listener,
        [{
            "rater_id": "r01", "sample_id": "hidden_sample", "presentation_id": "ctx_p1",
            "presentation_variant": "contextual", "task_mode": "controlled_dialogue",
            "audio_asset_id": "target_asset", "context_type": "preceding_turn", "context_id": "ctx1",
            "context_text": "", "context_audio_asset_id": "context_asset",
            "assigned_constructs": "intonation_contextual_appropriateness", "analyzable_yes_no": "",
            "clarity_comprehensibility_1to7": "", "fluency_1to7": "", "rhythm_naturalness_1to7": "",
            "intonation_utterance_naturalness_1to7": "", "intonation_contextual_appropriateness_1to7": "",
            "context_presented_yes_no": "yes", "timestamp": "", "pack_schema": "consumer_listener_pack_v3",
        }],
        LISTENER_FIELDS,
    )
    _write(
        assets,
        [{
            "sample_id": "hidden_sample", "audio_asset_id": "target_asset", "source_audio_path": "target.wav",
            "context_audio_asset_id": "context_asset", "source_context_audio_path": "context.wav", "private_only": "true",
        }],
        ["sample_id", "audio_asset_id", "source_audio_path", "context_audio_asset_id", "source_context_audio_path", "private_only"],
    )

    store = ListenerStore(listener, assets, audio_root, tmp_path / "responses")
    payload = store.participant_payload("r01")
    item = payload["presentations"][0]

    assert item["context_presented"] is True
    assert item["context_text"] == ""
    assert item["context_audio_asset_id"] == "context_asset"
    assert store.audio_path("context_asset") == (audio_root / "context.wav").resolve()
    assert "source_context_audio_path" not in item

    assert 'id="contextAudio"' in HTML
    assert 'current.context_audio_asset_id' in HTML
    assert '/api/audio/${encodeURIComponent(current.context_audio_asset_id)}' in HTML
    assert '$("contextAudio").classList.add("hidden")' in HTML
