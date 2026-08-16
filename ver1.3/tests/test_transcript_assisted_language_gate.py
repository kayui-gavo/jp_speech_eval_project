from __future__ import annotations

import numpy as np
import soundfile as sf

from jp_speech_eval.asr import AsrTranscript
from jp_speech_eval.consumer_dimension_policy import build_consumer_score_dimensions
from jp_speech_eval.transcript_assisted import evaluate_transcript_assisted_light
from jp_speech_eval.user_score_policy import apply_user_score_policy


def _write_voice_like_wav(path, *, sr: int = 16000) -> None:
    t = np.arange(sr * 2, dtype=float) / sr
    y = 0.12 * np.sin(2.0 * np.pi * 190.0 * t)
    sf.write(path, y, sr, subtype="FLOAT")


def _fake_f0(y, sr):
    times = np.linspace(0.0, len(y) / sr, 20, endpoint=False)
    f0 = np.linspace(180.0, 230.0, 20)
    return times, f0, "test_f0"


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
    monkeypatch.setattr("jp_speech_eval.transcript_assisted.extract_f0", _fake_f0)
    raw = evaluate_transcript_assisted_light(wav, transcript="今日はラーメンを食べます")
    assert raw["details"]["score_eligible"] is True
    assert raw["details"]["language_gate"]["eligible"] is True
    assert raw["details"]["transcript_sanity"]["ok"] is True
    assert isinstance(raw["total_score"], int)
    shadow = raw["details"]["spontaneous_fluency_v2"]
    assert shadow["score_mapped"] is False
    assert shadow["breakdown"]["pause_location_available"] is False


def test_language_aware_asr_word_timestamps_feed_only_weak_pause_location_candidates(tmp_path, monkeypatch) -> None:
    wav = tmp_path / "japanese_asr.wav"
    _write_voice_like_wav(wav)

    def fake_language_aware(*args, **kwargs):
        return AsrTranscript(
            available=True,
            provider="faster-whisper",
            model="small",
            text="今日は、映画を見ます",
            language="ja",
            note="ok",
            language_probability=0.98,
            words=[
                {"start_sec": 0.05, "end_sec": 0.55, "text": "今日は、", "probability": 0.95},
                {"start_sec": 0.95, "end_sec": 1.35, "text": "映画を", "probability": 0.94},
                {"start_sec": 1.55, "end_sec": 1.90, "text": "見ます", "probability": 0.93},
            ],
        )

    monkeypatch.setattr("jp_speech_eval.transcript_assisted.transcribe_language_aware", fake_language_aware)
    monkeypatch.setattr("jp_speech_eval.transcript_assisted.extract_f0", _fake_f0)
    monkeypatch.setattr(
        "jp_speech_eval.transcript_assisted.detect_pauses",
        lambda y, sr: {
            "pause_count": 1,
            "pause_total": 0.35,
            "pause_ratio": 0.175,
            "pause_segments": [(0.57, 0.92)],
            "analysis_duration_sec": 2.0,
        },
    )

    raw = evaluate_transcript_assisted_light(wav)
    breakdown = raw["details"]["spontaneous_fluency_v2"]["breakdown"]
    assert raw["details"]["score_eligible"] is True
    assert breakdown["pause_location_available"] is True
    assert breakdown["pause_location_confidence"] == "low"
    assert breakdown["pause_location_counts"]["after_asr_punctuation_candidate"] == 1
    assert breakdown["pause_location_source"] == "faster_whisper_word_timestamps_plus_asr_punctuation"


def test_v4_free_speech_evidence_is_end_to_end_shadow_only(tmp_path, monkeypatch) -> None:
    wav = tmp_path / "japanese_v4.wav"
    _write_voice_like_wav(wav)

    def fake_language_aware(*args, **kwargs):
        return AsrTranscript(
            available=True,
            provider="faster-whisper",
            model="small",
            text="今日は友達と駅でラーメンを食べます",
            language="ja",
            note="ok",
            language_probability=0.99,
            words=[
                {"start_sec": 0.05, "end_sec": 0.32, "text": "今日は", "probability": 0.96},
                {"start_sec": 0.34, "end_sec": 0.68, "text": "友達と", "probability": 0.92},
                {"start_sec": 0.72, "end_sec": 0.92, "text": "駅で", "probability": 0.89},
                {"start_sec": 0.98, "end_sec": 1.37, "text": "ラーメンを", "probability": 0.94},
                {"start_sec": 1.42, "end_sec": 1.72, "text": "食べます", "probability": 0.91},
            ],
            segments=[
                {
                    "start_sec": 0.0,
                    "end_sec": 1.8,
                    "avg_logprob": -0.15,
                    "no_speech_prob": 0.01,
                    "compression_ratio": 1.12,
                }
            ],
        )

    monkeypatch.setattr("jp_speech_eval.transcript_assisted.transcribe_language_aware", fake_language_aware)
    monkeypatch.setattr("jp_speech_eval.transcript_assisted.extract_f0", _fake_f0)

    raw = evaluate_transcript_assisted_light(wav)
    evidence = raw["details"]["shadow"]["free_speech_dimension_evidence"]
    candidate = raw["details"]["shadow"]["free_speech_candidate_surface"]
    assert evidence["clarity"]["available"] is True
    assert evidence["rhythm"]["available"] is True
    assert evidence["intonation"]["available"] is True
    assert candidate["product_score_changed"] is False
    assert candidate["user_facing"] is False
    assert candidate["component_candidates"]["clarity"] != 70.0

    product = apply_user_score_policy(raw, mode="transcript_assisted_light")
    assert product["score_contract_version"] == "consumer_four_score_v2"
    assert product["evidence_schema_version"] == "consumer_evidence_v5"
    assert product["component_scores"]["clarity"]["value"] == 70
    assert product["component_scores"]["mora_timing"]["value"] == 70
    assert product["component_scores"]["intonation"]["value"] == 70

    dimensions = {
        item["key"]: item
        for item in build_consumer_score_dimensions(raw, product, mode="transcript_assisted_light")
    }
    assert dimensions["clarity"]["evidence_state"] == "neutral_prior"
    assert dimensions["mora_timing"]["evidence_state"] == "neutral_prior"
    assert dimensions["intonation"]["evidence_state"] == "neutral_prior"
    assert dimensions["clarity"]["numeric_semantics"] == "neutral_anchor_not_direct_measurement"
