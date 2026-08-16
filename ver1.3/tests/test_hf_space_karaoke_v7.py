from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "debug_ui" / "index.html"
DEPLOY = ROOT.parent / "deploy" / "start_full_demo.sh"


class _IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "id" and value:
                self.ids.append(value)


def test_existing_space_index_is_the_karaoke_surface_not_a_parallel_replacement():
    text = HTML.read_text(encoding="utf-8")
    assert 'id="languageSwitch"' in text
    assert 'id="karaokeSection"' in text
    assert 'id="karaokeLyrics"' in text
    assert 'id="karaokeCanvas"' in text
    assert 'id="karaokeScrub"' in text
    assert 'id="consumer' not in text


def test_public_demo_exposes_fixed_and_direct_free_speech_only():
    text = HTML.read_text(encoding="utf-8")
    deploy = DEPLOY.read_text(encoding="utf-8")
    assert 'const publicUiModes = ["reference", "transcript_assisted_light"];' in text
    assert '--available-modes reference,transcript_assisted_light' in deploy
    assert '--available-modes reference,asr_pseudo_reference' not in deploy


def test_public_consumer_dimensions_use_frozen_four_dimension_contract():
    text = HTML.read_text(encoding="utf-8")
    assert 'dimensions:{delivery_fluency:"流暢さ", clarity:"明瞭さ", mora_timing:"リズム", intonation:"抑揚"}' in text
    assert 'dimensions:{delivery_fluency:"Fluency", clarity:"Clarity", mora_timing:"Rhythm", intonation:"Intonation"}' in text
    assert 'renderScores(payload.result, payload.user_facing, payload.karaoke_timeline)' in text
    assert 'const timelineDimensions = Array.isArray(timeline?.dimensions)' in text


def test_evidence_states_are_visible_without_turning_neutral_prior_into_measurement():
    text = HTML.read_text(encoding="utf-8")
    assert 'stateName === "measured_proxy"' in text
    assert 'stateName === "broad_proxy"' in text
    assert 'stateName === "neutral_prior"' in text
    assert 'evidenceMeasured:"音声から確認"' in text
    assert 'evidenceBroad:"大まかな目安"' in text
    assert 'evidenceNeutral:"参考値"' in text


def test_karaoke_uses_backend_timestamps_and_never_fabricates_token_timing():
    text = HTML.read_text(encoding="utf-8")
    assert 'button.dataset.start = row.start_sec' in text
    assert 'button.dataset.end = row.end_sec' in text
    assert 'const words = Array.isArray(timeline.words)' in text
    assert 'const moras = Array.isArray(timeline.moras)' in text
    assert 'timeline.transcript || cc("karaokePlaceholder")' in text
    assert '/ rows.length' not in text
    assert 'transcript.length' not in text


def test_karaoke_replay_follows_recording_clock_and_has_pause_pitch_layers():
    text = HTML.read_text(encoding="utf-8")
    assert 'const now = Number(audio.currentTime)' in text
    assert 'requestAnimationFrame(tick)' in text
    assert 'karaokePitchLayer' in text
    assert 'karaokePauseLayer' in text
    assert '(timeline.pauses || []).forEach' in text
    assert 'timeline.pitch?.available' in text
    assert 'timeline.pitch.reference_points' in text


def test_uploaded_audio_and_sample_audio_can_drive_the_same_replay_surface():
    text = HTML.read_text(encoding="utf-8")
    assert 'state.lastRecordingUrl = URL.createObjectURL(file)' in text
    assert '$("lastRecordingAudio").src = state.lastRecordingUrl' in text
    assert '$("lastRecordingAudio").src = $("sampleAudio").src' in text
    assert 'if ($("lastRecordingAudio").src) $("lastRecordingAudio").play()' in text


def test_microphone_capture_requests_rawish_audio_for_measurement_consistency():
    text = HTML.read_text(encoding="utf-8")
    assert 'channelCount: 1' in text
    assert 'echoCancellation: false' in text
    assert 'noiseSuppression: false' in text
    assert 'autoGainControl: false' in text


def test_public_pitch_view_does_not_present_hl_match_as_red_green_correctness():
    text = HTML.read_text(encoding="utf-8")
    assert 'const showLexicalDebug = !isPublicDemo();' in text
    assert 'if (showLexicalDebug)' in text
    gate = text.index('if (showLexicalDebug)')
    legacy_color = text.index('obs === targetPitch[i] ? "#047857" : "#b42318"')
    assert legacy_color > gate


def test_public_demo_hides_engineering_replay_and_legacy_diagnostic_panels():
    text = HTML.read_text(encoding="utf-8")
    assert '.public-mode .debug-only-panel { display:none!important; }' in text
    assert '<details class="advanced-panel debug-only-panel">' in text
    assert '<section class="debug-only-panel">\n        <div class="panel-head">\n          <h2 data-i18n="realtimeReplay">' in text
    assert '<section class="debug-only-panel">\n        <div class="panel-head">\n          <h2 data-i18n="speechRegion">' in text
    assert '<section class="debug-only-panel">\n        <div class="panel-head">\n          <h2 data-i18n="pitchContour">' in text
    assert 'document.body.classList.toggle("public-mode", config.server_label === "Public demo")' in text


def test_existing_space_index_has_no_duplicate_dom_ids():
    parser = _IdCollector()
    parser.feed(HTML.read_text(encoding="utf-8"))
    assert len(parser.ids) == len(set(parser.ids))


def test_existing_space_inline_javascript_parses_when_node_is_available():
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
