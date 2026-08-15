from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from jp_speech_eval.sentence_cache import build_sentence_cache, load_sentence_cache


def test_external_lab_alignment_becomes_verified_reference_boundary(tmp_path) -> None:
    sr = 16000
    wav_path = tmp_path / "reference.wav"
    t = np.arange(int(0.60 * sr), dtype=float) / sr
    y = np.zeros_like(t)
    active = (t >= 0.10) & (t < 0.50)
    y[active] = 0.20 * np.sin(2.0 * np.pi * 220.0 * t[active])
    sf.write(wav_path, y, sr, subtype="FLOAT")

    lab_path = tmp_path / "reference.lab"
    lab_path.write_text("0.10 0.30 a\n0.30 0.50 i\n", encoding="utf-8")

    prefix = tmp_path / "aligned_cache"
    cache = build_sentence_cache(
        "あい",
        prefix,
        sr=sr,
        reference_wav_path=wav_path,
        reference_alignment_path=lab_path,
        reference_alignment_method="lab",
        reference_source="test_human_reference",
    )

    assert cache.meta.ref_boundary_method == "external_lab"
    assert cache.meta.ref_boundary_tier == "verified_phone_alignment"
    assert cache.meta.ref_boundary_confidence >= 0.80
    assert cache.meta.ref_boundary_source == str(lab_path)
    assert cache.meta.ref_duration_sec == pytest.approx(0.40, abs=0.01)
    assert len(cache.meta.ref_mora_boundaries) == 2
    assert cache.meta.ref_mora_boundaries[0][0] == pytest.approx(0.0, abs=1e-4)
    assert cache.meta.ref_mora_boundaries[-1][1] == pytest.approx(0.40, abs=0.01)


def test_reference_alignment_requires_exact_reference_wav_timebase(tmp_path) -> None:
    lab_path = tmp_path / "reference.lab"
    lab_path.write_text("0.00 0.10 a\n0.10 0.20 i\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires reference_wav_path"):
        build_sentence_cache(
            "あい",
            tmp_path / "invalid_cache",
            reference_alignment_path=lab_path,
        )


def test_legacy_equal_cache_loads_with_explicit_low_precision_tier(tmp_path) -> None:
    sr = 16000
    prefix = tmp_path / "legacy"
    payload = {
        "text": "あい",
        "kana": "アイ",
        "moras": ["ア", "イ"],
        "target_pitch": ["L", "H"],
        "pitch_target_source": "heuristic",
        "is_question": False,
        "sr": sr,
        "ref_duration_sec": 0.4,
        "ref_mora_boundaries": [[0.0, 0.2], [0.2, 0.4]],
        "frontend_raw": [],
        "accent_phrases": [],
        "reference_text": "あい",
        "reference_source": "legacy_reference",
        "ref_boundary_method": "equal_mora",
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        ref_y=np.zeros(int(sr * 0.4), dtype=np.float32),
        ref_mfcc=np.zeros((13, 10), dtype=np.float32),
        ref_f0_times=np.zeros(10, dtype=np.float32),
        ref_f0=np.zeros(10, dtype=np.float32),
    )

    cache = load_sentence_cache(prefix)
    assert cache.meta.ref_boundary_tier == "equal_fallback"
    assert cache.meta.ref_boundary_confidence < 0.30
