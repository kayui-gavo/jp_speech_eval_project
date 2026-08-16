from __future__ import annotations

import inspect

import numpy as np
import soundfile as sf

from jp_speech_eval.app_core.karaoke_timeline import build_consumer_karaoke_timeline
from jp_speech_eval.consumer_dimension_policy import build_consumer_score_components
from jp_speech_eval.feedback_renderer import render_user_facing_result
from jp_speech_eval.reliability_gate import evaluate_reliability_gate
from jp_speech_eval.scoring_policy import policy_from_result
from jp_speech_eval.transcript_assisted import evaluate_transcript_assisted_light
from jp_speech_eval.transcript_sanity import check_free_speech_transcript_sanity


def _write_voice_like_wav(path, *, sr: int = 16000, duration: float = 0.8) -> None:
    t = np.arange(int(sr * duration), dtype=float) / sr
    y = 0.12 * np.sin(2.0 * np.pi * 190.0 * t)
    sf.write(path, y, sr, subtype="FLOAT")


def _fake_f0(y, sr):
    times = np.linspace(0.0, len(y) / sr, 24, endpoint=False)
    f0 = np.linspace(180.0, 220.0, 24)
    return times, f0, "test_f0"


def test_direct_free_speech_sanity_accepts_short_and_long_japanese() -> None:
    assert check_free_speech_transcript_sanity("はい").ok is True
    assert check_free_speech_transcript_sanity("え？").ok is True
    long_turn = "今日は大学で研究の打ち合わせをして、そのあと友達と昼ご飯を食べました。午後は図書館で資料を探してから、研究室に戻って実験の結果を整理しました。帰る前に先生へメールも送り、明日の発表で説明する内容をもう一度確認しました。"
    assert len(long_turn) > 80
    assert check_free_speech_transcript_sanity(long_turn).ok is True
    assert check_free_speech_transcript_sanity("I like ramen very much").ok is False
    assert check_free_speech_transcript_sanity("あ" * 12).ok is False


def test_short_valid_japanese_turn_receives_score_in_direct_free_speech(tmp_path, monkeypatch) -> None:
    wav = tmp_path / "short_ja.wav"
    _write_voice_like_wav(wav)
    monkeypatch.setattr("jp_speech_eval.transcript_assisted.extract_f0", _fake_f0)
    raw = evaluate_transcript_assisted_light(wav, transcript="はい")
    assert raw["details"]["score_eligible"] is True
    assert raw["details"]["transcript_sanity"]["ok"] is True
    assert raw["total_score"] is not None
    assert raw["moras"]


def _low_reference_provenance_result():
    return {
        "alignment_mode": "cached_dtw",
        "pronunciation_score": 92,
        "prosody_score": 91,
        "fluency_score": 86,
        "tone_score": 88,
        "moras": ["ラ", "ー", "メ", "ン"],
        "mora_table": [
            {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2, "f0_hz": 180.0},
            {"mora": "ー", "start_sec": 0.2, "end_sec": 0.4, "f0_hz": 195.0},
            {"mora": "メ", "start_sec": 0.4, "end_sec": 0.6, "f0_hz": 188.0},
            {"mora": "ン", "start_sec": 0.6, "end_sec": 0.8, "f0_hz": 175.0},
        ],
        "endpointing": {"detected": True, "raw_duration": 0.9, "speech_start": 0.05, "speech_end": 0.85},
        "details": {
            "mode": "reference",
            "verified_level": "human_checked",
            "content_match": {"status": "pass", "score": 0.9, "kana_similarity": 0.95, "duration_ratio": 1.0},
            "alignment": {
                "available": True,
                "confidence": 0.92,
                "used_equal_fallback": False,
                "mode": "cached_dtw",
                "reference_boundary_method": "equal_mora",
                "reference_boundary_confidence": 0.15,
                "reference_boundary_tier": "equal_fallback",
            },
            "reliability": {
                "overall": 0.95,
                "endpointing": 1.0,
                "alignment": 0.92,
                "mora_evidence": 0.9,
                "f0_coverage": 0.9,
                "duration_ratio_to_reference": 1.0,
            },
            "recording_quality": {"score": 0.95},
            "fluency": {"rate_score": 88.0, "pause_score": 85.0, "speech_rate_mora_per_sec": 5.0},
            "prosody": {"contour_corr": 0.8, "contour_valid_mora_count": 4, "note": "ok"},
            "tone": {"pitch_range_log": 0.4, "pitch_score": 88.0},
            "reference_f0_by_mora": [175.0, 190.0, 185.0, 172.0],
        },
    }


def test_low_reference_boundary_provenance_cannot_become_local_public_evidence() -> None:
    raw = _low_reference_provenance_result()
    components = {item["key"]: item for item in build_consumer_score_components(raw, mode="reference")}
    assert components["mora_timing"]["evidence_tier"] == "alignment_fallback_broad_timing"
    assert components["mora_timing"]["confidence"] == "low"
    assert components["intonation"]["source_field"] != "prosody_score"
    assert components["intonation"]["confidence"] == "low"


def test_low_reference_boundary_provenance_blocks_local_feedback_without_marking_learner_down() -> None:
    raw = _low_reference_provenance_result()
    policy = policy_from_result(raw, mode="reference")
    assert policy.allow_pitch_feedback is True
    gate = evaluate_reliability_gate(raw, policy)
    assert gate.practice_check_result == "ok"
    assert gate.allow_pitch_feedback is False
    assert gate.allow_special_mora_feedback is False
    assert gate.allow_pronunciation_detail is False
    assert "reference_boundary_precision_low_broad_only" in gate.reasons
    assert gate.messages


def test_low_reference_boundary_provenance_marks_karaoke_mora_sync_approximate() -> None:
    raw = _low_reference_provenance_result()
    timeline = build_consumer_karaoke_timeline(raw, {"score_dimensions": []})
    assert timeline["sync_mode"] == "mora_alignment_approximate"
    assert timeline["moras"]
    assert all(item["approximate"] for item in timeline["moras"])
    assert all(item["alignment_confidence"] == 0.15 for item in timeline["moras"])


def test_special_mora_user_feedback_is_opt_in_by_default() -> None:
    defaults = inspect.signature(render_user_facing_result).parameters
    assert defaults["enable_user_facing_calibrated_special_mora"].default is False
