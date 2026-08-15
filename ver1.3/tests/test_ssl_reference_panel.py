from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jp_speech_eval.shadow_assessment import run_assessment_shadows
from jp_speech_eval.ssl_reference_panel import (
    SSL_REFERENCE_PANEL_SCHEMA,
    load_ssl_reference_panel,
)


def _write_panel(tmp_path: Path, *, include_tts: bool = True) -> Path:
    refs = []
    for name, speaker, kind in [
        ("r1.wav", "n1", "human_native"),
        ("r2.wav", "n2", "human_native"),
    ]:
        (tmp_path / name).write_bytes(b"placeholder")
        refs.append(
            {
                "reference_id": name[:-4],
                "target_text": "テストです",
                "audio_path": name,
                "speaker_id": speaker,
                "reference_kind": kind,
                "provenance": "consented_reference_recording",
            }
        )
    if include_tts:
        (tmp_path / "tts.wav").write_bytes(b"placeholder")
        refs.append(
            {
                "reference_id": "tts",
                "target_text": "テストです",
                "audio_path": "tts.wav",
                "speaker_id": "",
                "reference_kind": "tts_fallback",
                "provenance": "generated",
            }
        )
    path = tmp_path / "panel.json"
    path.write_text(
        json.dumps({"schema": SSL_REFERENCE_PANEL_SCHEMA, "references": refs}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_panel_selects_human_references_and_excludes_tts_when_humans_exist(tmp_path: Path):
    panel = load_ssl_reference_panel(_write_panel(tmp_path))
    selected = panel.references_for_target("テストです")
    assert [item.reference_id for item in selected] == ["r1", "r2"]
    assert all(item.reference_kind == "human_native" for item in selected)
    assert panel.panel_id.startswith("sslref_")


def test_explicit_panel_does_not_fall_back_to_wrong_target(tmp_path: Path):
    panel = load_ssl_reference_panel(_write_panel(tmp_path))
    assert panel.references_for_target("別の文") == []


def test_duplicate_reference_ids_are_rejected(tmp_path: Path):
    path = _write_panel(tmp_path, include_tts=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["references"][1]["reference_id"] = payload["references"][0]["reference_id"]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate SSL reference_id"):
        load_ssl_reference_panel(path)


class _FakeExtractor:
    def extract_layer(self, audio, layer_idx: int, sr: int):
        marker = int(round(float(np.asarray(audio).reshape(-1)[0])))
        frames = 24
        phase = {0: 0.0, 1: 0.0, 2: 0.08}.get(marker, 0.0)
        t = np.linspace(0.0, 2.2, frames) + phase
        return np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float32)


def test_shadow_aggregates_two_human_references_without_loading_real_model(tmp_path: Path, monkeypatch):
    panel_path = _write_panel(tmp_path)
    user_path = tmp_path / "user.wav"
    user_path.write_bytes(b"placeholder")

    def fake_audio(path: str, sample_rate: int):
        name = Path(path).name
        marker = {"user.wav": 0.0, "r1.wav": 1.0, "r2.wav": 2.0, "tts.wav": 3.0}[name]
        return np.asarray([marker, 0.0, 0.0], dtype=np.float32), sample_rate

    monkeypatch.setattr("jp_speech_eval.shadow_assessment._audio", fake_audio)
    result = {"target_text": "テストです", "details": {}}
    shadows = run_assessment_shadows(
        result,
        user_audio_path=str(user_path),
        enable_ssl_shadow=True,
        ssl_reference_panel_path=str(panel_path),
        ssl_reference_aggregation="median",
        ssl_extractor=_FakeExtractor(),
    )
    pronunciation = shadows["ssl_pronunciation"]
    rhythm = shadows["rhythm_dtw_v1"]
    assert pronunciation["available"] is True
    assert pronunciation["reference_count"] == 2
    assert len(pronunciation["reference_distances"]) == 2
    assert {item["reference_id"] for item in pronunciation["reference_distances"]} == {"r1", "r2"}
    assert pronunciation["reference_panel"]["human_only"] is True
    assert pronunciation["score_mapped"] is False
    assert rhythm["available"] is True
    assert rhythm["reference_count"] == 2
    assert len(rhythm["reference_rhythm"]) == 2
    assert rhythm["score_mapped"] is False


def test_shadow_fails_closed_when_explicit_panel_has_no_matching_target(tmp_path: Path, monkeypatch):
    panel_path = _write_panel(tmp_path)
    user_path = tmp_path / "user.wav"
    user_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "jp_speech_eval.shadow_assessment._audio",
        lambda path, sample_rate: (np.asarray([0.0, 0.0], dtype=np.float32), sample_rate),
    )
    result = {"target_text": "別の文", "details": {}}
    shadows = run_assessment_shadows(
        result,
        user_audio_path=str(user_path),
        enable_ssl_shadow=True,
        ssl_reference_panel_path=str(panel_path),
        ssl_extractor=_FakeExtractor(),
    )
    assert shadows["ssl_pronunciation"]["available"] is False
    assert "no human reference" in shadows["ssl_pronunciation"]["error"]
