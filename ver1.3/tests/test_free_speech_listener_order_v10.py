from __future__ import annotations

import csv
from pathlib import Path

from scripts.free_speech_listener_server_v10 import ListenerStore


FIELDS = [
    "rater_id", "sample_id", "presentation_id", "presentation_variant", "task_mode",
    "audio_asset_id", "context_type", "context_id", "context_text", "context_audio_asset_id",
    "assigned_constructs", "analyzable_yes_no", "clarity_comprehensibility_1to7",
    "fluency_1to7", "rhythm_naturalness_1to7", "intonation_utterance_naturalness_1to7",
    "intonation_contextual_appropriateness_1to7", "context_presented_yes_no", "timestamp", "pack_schema",
]


def _listener_row(presentation_id: str, variant: str, audio_asset_id: str) -> dict[str, str]:
    contextual = variant == "contextual"
    return {
        "rater_id": "r01",
        "sample_id": f"hidden_{presentation_id}",
        "presentation_id": presentation_id,
        "presentation_variant": variant,
        "task_mode": "controlled_dialogue" if contextual else "spontaneous",
        "audio_asset_id": audio_asset_id,
        "context_type": "preceding_turn" if contextual else "none",
        "context_id": "ctx" if contextual else "",
        "context_text": "どうしてそう思ったんですか。" if contextual else "",
        "context_audio_asset_id": "",
        "assigned_constructs": "intonation_contextual_appropriateness" if contextual else "clarity_comprehensibility|fluency|rhythm_naturalness|intonation_utterance_naturalness",
        "analyzable_yes_no": "",
        "clarity_comprehensibility_1to7": "",
        "fluency_1to7": "",
        "rhythm_naturalness_1to7": "",
        "intonation_utterance_naturalness_1to7": "",
        "intonation_contextual_appropriateness_1to7": "",
        "context_presented_yes_no": "yes" if contextual else "no",
        "timestamp": "",
        "pack_schema": "consumer_listener_pack_v3",
    }


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_listener_order_ignores_csv_order_and_keeps_context_after_isolated(tmp_path: Path) -> None:
    listener = tmp_path / "listener.csv"
    assets = tmp_path / "assets.csv"
    audio_root = tmp_path / "audio"
    audio_root.mkdir()

    # Deliberately adversarial CSV order: contextual first and isolated rows not randomized.
    rows = [
        _listener_row("ctx_first", "contextual", "a3"),
        _listener_row("iso_z", "isolated", "a1"),
        _listener_row("iso_a", "isolated", "a2"),
    ]
    _write_csv(listener, rows, FIELDS)
    asset_rows = []
    for asset_id in ("a1", "a2", "a3"):
        filename = f"{asset_id}.wav"
        (audio_root / filename).write_bytes(b"RIFFxxxxWAVEfixture")
        asset_rows.append(
            {
                "sample_id": f"hidden_{asset_id}",
                "audio_asset_id": asset_id,
                "source_audio_path": filename,
                "context_audio_asset_id": "",
                "source_context_audio_path": "",
                "private_only": "true",
            }
        )
    _write_csv(
        assets,
        asset_rows,
        ["sample_id", "audio_asset_id", "source_audio_path", "context_audio_asset_id", "source_context_audio_path", "private_only"],
    )

    store = ListenerStore(listener, assets, audio_root, tmp_path / "responses")
    first = store.participant_payload("r01")
    second = store.participant_payload("r01")

    variants = [item["presentation_variant"] for item in first["presentations"]]
    assert variants == ["isolated", "isolated", "contextual"]
    assert first["presentation_order"] == "isolated_then_contextual_rater_hash_v1"
    assert [item["presentation_id"] for item in first["presentations"]] == [
        item["presentation_id"] for item in second["presentations"]
    ]
    assert first["presentations"][0]["presentation_id"] != "ctx_first"
