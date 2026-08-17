from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_free_speech_confirmatory_v11 as runner


def _freeze_json(tmp_path: Path) -> Path:
    base = tmp_path / "free_speech_v5_promotion_protocol.json"
    v10 = tmp_path / "free_speech_v10_consumer_promotion_protocol.json"
    base.write_text("{}", encoding="utf-8")
    v10.write_text("{}", encoding="utf-8")
    frozen = {
        "schema": "free_speech_confirmatory_freeze_v11",
        "freeze_fingerprint_sha256": "a" * 64,
        "manifest": {"path": str(tmp_path / "manifest.csv")},
        "audio_root": str(tmp_path / "audio"),
        "protocols": [{"path": str(base)}, {"path": str(v10)}],
    }
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")
    return path


def test_runner_stops_before_v10_when_freeze_verification_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    freeze = _freeze_json(tmp_path)
    called = {"v10": False}

    monkeypatch.setattr(runner, "verify_freeze", lambda _: {"verification_ok": False, "drift_detected": True})

    def fake_v10(*args, **kwargs):
        called["v10"] = True
        return {}

    monkeypatch.setattr(runner, "run_v10", fake_v10)
    with pytest.raises(RuntimeError, match="freeze verification failed"):
        runner.run(freeze, tmp_path / "out")
    assert called["v10"] is False


def test_runner_uses_only_frozen_manifest_audio_and_protocols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    freeze = _freeze_json(tmp_path)
    captured = {}
    monkeypatch.setattr(runner, "verify_freeze", lambda _: {"verification_ok": True, "drift_detected": False})

    def fake_v10(manifest, out_dir, **kwargs):
        captured["manifest"] = manifest
        captured["out_dir"] = str(out_dir)
        captured.update(kwargs)
        return {"decision": "none", "next_action": "collect ratings"}

    monkeypatch.setattr(runner, "run_v10", fake_v10)
    report = runner.run(freeze, tmp_path / "out")
    frozen = json.loads(freeze.read_text(encoding="utf-8"))

    assert captured["manifest"] == frozen["manifest"]["path"]
    assert captured["audio_root"] == frozen["audio_root"]
    assert Path(captured["base_protocol_json"]).name == "free_speech_v5_promotion_protocol.json"
    assert Path(captured["v10_protocol_json"]).name == "free_speech_v10_consumer_promotion_protocol.json"
    assert report["held_set_used_for_threshold_tuning"] is False
    assert report["protocol_retuning_allowed"] is False
    assert report["product_score_changed"] is False
