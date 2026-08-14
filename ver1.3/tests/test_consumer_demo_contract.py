from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_consumer_launcher():
    path = ROOT / "scripts" / "consumer_demo.py"
    spec = importlib.util.spec_from_file_location("consumer_demo_contract", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_consumer_demo_uses_bundled_reference_assets() -> None:
    launcher = _load_consumer_launcher()
    assert launcher.DEFAULT_CACHE.with_suffix(".json").exists()
    assert launcher.DEFAULT_CACHE.with_suffix(".npz").exists()
    assert launcher.DEFAULT_WAV.exists()
    args = launcher._inject_defaults([])
    assert "--public-demo" in args
    assert str(launcher.DEFAULT_CACHE) in args
    assert str(launcher.DEFAULT_WAV) in args


def test_consumer_demo_allows_explicit_cache_override() -> None:
    launcher = _load_consumer_launcher()
    args = launcher._inject_defaults(["--cache", "/tmp/custom", "--wav=/tmp/custom.wav"])
    assert str(launcher.DEFAULT_CACHE) not in args
    assert str(launcher.DEFAULT_WAV) not in args


def test_consumer_ui_matches_personal_site_design_tokens() -> None:
    ui = (ROOT / "debug_ui" / "consumer_v2.html").read_text(encoding="utf-8")
    for token in ("--ivory:#f5f0e8", "--navy:#1b2944", "--rose:#9a5a69", "--gold:#b2905e"):
        assert token in ui
    assert "JAPANESE SPEAKING / PUBLIC BETA" in ui
    assert "PRACTICE REFERENCE" in ui
    assert "研究指标不会抢占你的注意力" in ui
    assert 'fetch("/api/config")' in ui
    assert 'fetch("/api/evaluate"' in ui
    assert 'fetch("/api/evaluate-confirmed-asr"' in ui
