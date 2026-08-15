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


def test_consumer_entry_paths_never_use_legacy_score_renderer() -> None:
    launcher = _load_consumer_launcher()
    for path in ("", "/", "/?lang=ja", "/index.html", "/index.html?lang=ja"):
        assert launcher._is_consumer_entry_path(path)
    assert not launcher._is_consumer_entry_path("/consumer_v2.html")
    assert not launcher._is_consumer_entry_path("/api/config")


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


def test_consumer_ui_renders_unavailable_dimension_as_dash_not_zero() -> None:
    ui = (ROOT / "debug_ui" / "consumer_v2.html").read_text(encoding="utf-8")
    assert 'd.available!==false&&Number.isFinite(val)' in ui
    assert 'ok?Math.round(val):"--"' in ui


def test_consumer_launcher_overrides_legacy_dimension_labels_with_semantic_four() -> None:
    launcher = _load_consumer_launcher()
    html = launcher._consumer_html_bytes().decode("utf-8")
    for key in ("delivery_fluency", "clarity", "mora_timing", "intonation"):
        assert key in html
    assert 'clarity:{"zh-CN":"清晰度"' in html
    assert 'intonation:{"zh-CN":"抑扬"' in html
    assert 'legacy_pronunciation_timing_proxy_is_not_clarity' not in html


def test_consumer_launcher_surfaces_low_confidence_without_hiding_numeric_dimension() -> None:
    launcher = _load_consumer_launcher()
    html = launcher._consumer_html_bytes().decode("utf-8")
    assert "dim-confidence" in html
    assert '"zh-CN":{low:"参考"}' in html
    assert 'ja:{low:"参考"}' in html
    assert "consumerConfidenceMarker" in html
    assert "consumerDecorateResult" in html
    assert "d.available!==false&&Number.isFinite(val)" in html


def test_consumer_target_mismatch_copy_is_nonpunitive_and_score_preserving() -> None:
    launcher = _load_consumer_launcher()
    html = launcher._consumer_html_bytes().decode("utf-8")
    assert "句子不同，但这段日语仍然可以评分" in html
    assert "没有把你和固定例句硬比较" in html
    assert "お題とは違いますが、日本語として評価できます" in html
    assert "Different sentence, still scoreable as Japanese" in html
    assert "content_mismatch_general_score" in html
    assert "reference_mismatch_general_japanese" in html
