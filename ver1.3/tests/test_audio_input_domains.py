from __future__ import annotations

import numpy as np
import soundfile as sf

from jp_speech_eval.audio_features import load_audio, load_audio_views
from jp_speech_eval.recording_quality import assess_recording_quality


def _write_tone(path, *, amplitude: float, sr: int = 44100, duration: float = 0.8) -> None:
    t = np.arange(int(sr * duration), dtype=float) / sr
    y = amplitude * np.sin(2.0 * np.pi * 220.0 * t)
    sf.write(path, y, sr, subtype="FLOAT")


def test_load_audio_views_preserves_recording_level_but_normalizes_analysis(tmp_path) -> None:
    quiet_path = tmp_path / "quiet.wav"
    normal_path = tmp_path / "normal.wav"
    _write_tone(quiet_path, amplitude=0.005)
    _write_tone(normal_path, amplitude=0.20)

    quiet = load_audio_views(str(quiet_path), sr=16000)
    normal = load_audio_views(str(normal_path), sr=16000)

    assert quiet.raw_sr == 44100
    assert normal.raw_sr == 44100
    assert np.max(np.abs(quiet.raw_y)) < 0.006
    assert 0.19 < np.max(np.abs(normal.raw_y)) < 0.21
    assert np.max(np.abs(quiet.analysis.y)) > 0.99
    assert np.max(np.abs(normal.analysis.y)) > 0.99
    assert quiet.analysis_normalization_gain > normal.analysis_normalization_gain * 20.0


def test_legacy_load_audio_keeps_raw_provenance_for_recording_quality(tmp_path) -> None:
    quiet_path = tmp_path / "quiet.wav"
    _write_tone(quiet_path, amplitude=0.005)

    audio = load_audio(str(quiet_path), sr=16000)
    quality = assess_recording_quality(audio.y, audio.sr)

    assert quality["input_domain"] == "amplitude_preserved_decode_from_analysis_provenance"
    assert quality["recording_sample_rate"] == 44100
    assert quality["peak"] < 0.006
    assert quality["analysis_normalization_gain"] > 100.0
    assert any("Low speech level" in warning for warning in quality["warnings"])


def test_clipping_is_measured_before_analysis_normalization(tmp_path) -> None:
    path = tmp_path / "clipped.wav"
    sr = 48000
    y = np.zeros(sr, dtype=float)
    y[8000:32000] = 1.0
    sf.write(path, y, sr, subtype="FLOAT")

    audio = load_audio(str(path), sr=16000)
    quality = assess_recording_quality(audio.y, audio.sr)

    assert quality["recording_sample_rate"] == sr
    assert quality["clipping_ratio"] > 0.01
    assert any("Possible clipping" in warning for warning in quality["warnings"])
