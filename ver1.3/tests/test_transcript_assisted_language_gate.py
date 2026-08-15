from __future__ import annotations

import numpy as np
import soundfile as sf

from jp_speech_eval.asr import AsrTranscript
from jp_speech_eval.transcript_assisted import evaluate_transcript_assisted_light
from jp_speech_eval.user_score_policy import apply_user_score_policy


def _write_voice_like_wav(path, *, sr: int = 16000) -> None:
    t = np.arange(sr * 2, dtype=float) / sr
    y = 0.12 * np.sin(2.0 * np.pi * 190.0 * t)
    sf.write(path, y, sr, subtype="FLOAT")


def test_confident_non_japanese_is_no_score_even_if_asr_text_looks_japanese(tmp_path, monkeypatch) -> None:
    wav = tmp_path / "english.wav"
    _write_voice_like_wav(wav)

    def fake_language_aware(*args, **kwargs):
        return AsrTranscript(
            available=True,
            provider="test",
            model="small",
            text="これはテストです",
            language="en",
            note="ok",
            language_probability=0.93,
        )

    monkeypatch.setattr(
        "jp_speech_eval.transcript_assisted.transcribe_language_aware",
        fake_language_aware,
    )
    raw = evaluate_transcript_assisted_light(wav)
    assert raw["details"]["language_gate"]["eligible"] is False
    assert raw["details"]["language_gate"]["reason"] == "detected_non_japanese"
    assert raw["details"]["transcript_sanity"]["ok"] is False
    assert raw["details"]["score_eligible"] is False
    assert raw["total_score"] is None
    assert raw["fluency_score"] is None

    product = apply_user_score_policy(raw, mode="transcript_assisted_light")
    assert product["score_available"] is False
    assert product["display_score"] is None
    assert product["main_message_key"] == "invalid_or_non_japanese"


def test_external_non_japanese_transcript_cannot_drive_mora_rate_score(tmp_path) -> None:
    wav = tmp_path / "voice.wav"
    _write_voice_like_wav(wav)
    raw = evaluate_transcript_assisted_light(wav, transcript="I like ramen very much")
    assert raw["details"]["score_eligible"] is False
    assert raw["details"]["transcript_sanity"]["ok"] is False
    assert raw["total_score"] is None
    assert raw["moras"] == []


def test_external_japanese_transcript_keeps_existing_score_path_and_adds_fluency_shadow(tmp_path, monkeypatch) -> None:
    wav = tmp_path / "japanese.wav"
    _write_voice_like_wav(wav)

    def fake_f0(y, sr):
        times = np.linspace(0.0, len(y) / sr, 20, endpoint=False)
        f0 = np.linspace(180.0, 230.0, 20)
        return times, f0, "test_f0"

    monkeypatch.setattr("jp_speech_eval.transcript_assisted.extract_f0", fake_f0)
    raw = evaluate_transcript_assisted_light(wav, transcript="今日はラーメンを食べます")
    assert raw["details"]["score_eligible"] is True
    assert raw["details"]["language_gate"]["eligible"] is True
    assert raw["details"]["transcript_sanity"]["ok"] is True
    assert isinstance(raw["total_score"], int)
    shadow = raw["details"]["spontaneous_fluency_v2"]
    assert shadow["score_mapped"] is False
    assert shadow["breakdown"]["pause_location_available"] is False
