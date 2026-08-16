from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "debug_ui" / "consumer_v3.html"


class _IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "id" and value:
                self.ids.append(value)


def test_consumer_v3_uses_exact_four_dimension_language():
    text = HTML.read_text(encoding="utf-8")
    for label in ("流暢さ", "明瞭さ", "リズム", "抑揚"):
        assert label in text
    assert "fallbackLabels=['流暢さ','明瞭さ','リズム','抑揚']" in text
    assert "ピッチ</" not in text
    assert "発音</h4>" not in text


def test_consumer_v3_free_mode_is_direct_broad_japanese_not_pseudo_reference():
    text = HTML.read_text(encoding="utf-8")
    assert "transcript_assisted_light" in text
    assert "自由に話す" in text
    assert "asr_pseudo_reference" not in text
    assert "ASR-generated pseudo-reference" not in text


def test_karaoke_sync_requires_real_timestamp_payload():
    text = HTML.read_text(encoding="utf-8")
    assert "dataset.start=w.start_sec" in text
    assert "dataset.end=w.end_sec" in text
    assert "t.sync_mode==='word_timestamps'" in text
    assert "sentence_progress_only" in text
    # The UI must not fabricate token timing from text length.
    assert "transcript.length" not in text
    assert "text.length" not in text
    assert "/ words.length" not in text


def test_karaoke_visualization_keeps_construct_guardrail_copy_visible():
    text = HTML.read_text(encoding="utf-8")
    assert "単語ごとの発音正誤やアクセント核の判定ではありません" in text
    assert "発音の正誤判定ではありません" in text
    assert "低い抑揚スコアを意味しません" in text
    assert "採点項目ではありません" in text


def test_dimension_evidence_states_are_visually_distinct():
    text = HTML.read_text(encoding="utf-8")
    assert "measured_proxy" in text
    assert "broad_proxy" in text
    assert "neutral_prior" in text
    assert "音声から確認" in text
    assert "大まかな目安" in text
    assert "参考値" in text
    assert "直接測定ではない参考値" in text


def test_playback_visual_uses_audio_clock_and_exposes_layer_controls():
    text = HTML.read_text(encoding="utf-8")
    assert "userAudio" in text
    assert "currentTime" in text
    assert "requestAnimationFrame" in text
    assert "timelineCanvas" in text
    assert "pitchLayer" in text
    assert "pauseLayer" in text
    assert "声の動き" in text
    assert ">間<" in text


def test_consumer_v3_has_no_duplicate_dom_ids():
    parser = _IdCollector()
    parser.feed(HTML.read_text(encoding="utf-8"))
    assert len(parser.ids) == len(set(parser.ids))


def test_consumer_v3_inline_javascript_parses_when_node_is_available():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed in this environment")
    text = HTML.read_text(encoding="utf-8")
    scripts = text.split("<script>")
    assert len(scripts) == 2
    script = scripts[1].split("</script>", 1)[0]
    completed = subprocess.run(
        [node, "--check", "-"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
