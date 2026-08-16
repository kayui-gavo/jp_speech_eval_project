from __future__ import annotations

import math

from jp_speech_eval.app_core.karaoke_timeline import (
    CANONICAL_DIMENSION_ORDER,
    build_consumer_karaoke_timeline,
    build_relative_f0_visualization,
)


def _user_facing():
    return {
        "score_dimensions": [
            {
                "key": "clarity",
                "label": "明瞭さ",
                "value": 70,
                "available": True,
                "confidence": "low",
                "evidence_state": "neutral_prior",
                "precision_hint": "neutral_placeholder",
                "construct": "broad_clarity_prior_without_target_independent_measurement",
                "numeric_semantics": "neutral_anchor_not_direct_measurement",
            },
            {
                "key": "delivery_fluency",
                "label": "流暢さ",
                "value": 82,
                "available": True,
                "confidence": "medium",
                "evidence_state": "measured_proxy",
                "precision_hint": "normal",
                "construct": "speed_and_breakdown_fluency",
            },
            {
                "key": "mora_timing",
                "label": "リズム",
                "value": 74,
                "available": True,
                "confidence": "low",
                "evidence_state": "broad_proxy",
                "precision_hint": "rough",
                "construct": "reference_independent_broad_japanese_timing_proxy",
            },
            {
                "key": "intonation",
                "label": "抑揚",
                "value": 79,
                "available": True,
                "confidence": "low",
                "evidence_state": "measured_proxy",
                "precision_hint": "rough",
                "construct": "reference_independent_pitch_movement_naturalness_proxy",
            },
        ]
    }


def _result(with_words: bool = True):
    words = (
        [
            {"text": "今日は", "start_sec": 0.10, "end_sec": 0.52, "probability": 0.94},
            {"text": "いい", "start_sec": 0.60, "end_sec": 0.88, "probability": 0.76},
            {"text": "天気", "start_sec": 0.95, "end_sec": 1.32, "probability": 0.55},
            {"text": "ですね", "start_sec": 1.40, "end_sec": 1.78, "probability": None},
        ]
        if with_words
        else None
    )
    f0_points = [
        {"t_sec": 0.10, "relative_semitone": -1.0},
        {"t_sec": 0.50, "relative_semitone": 0.0},
        {"t_sec": 1.00, "relative_semitone": None},
        {"t_sec": 1.50, "relative_semitone": 2.0},
    ]
    return {
        "target_text": "今日はいい天気ですね",
        "duration_sec": 2.0,
        "endpointing": {
            "raw_duration": 2.8,
            "speech_start": 0.40,
            "speech_end": 2.40,
            "speech_duration": 2.0,
            "detected": True,
        },
        "pause_info": {
            "pause_segments": [(0.80, 1.15)],
            "pause_count": 1,
            "pause_total": 0.35,
        },
        "details": {
            "asr": {
                "available": True,
                "text": "今日はいい天気ですね",
                "provider": "faster-whisper",
                "model": "small",
                "words": words,
            },
            "visualization_source": {
                "score_role": "visualization_only",
                "timebase": "speech_trim_relative",
                "relative_f0": {
                    "available": True,
                    "points": f0_points,
                },
            },
            "recording_quality": {"score": 0.91},
        },
    }


def test_relative_f0_visualization_is_speaker_relative_and_keeps_gaps():
    times = [i * 0.01 for i in range(30)]
    f0 = [200.0 + 10.0 * math.sin(i / 5.0) for i in range(30)]
    f0[8] = 0.0
    f0[9] = 0.0
    payload = build_relative_f0_visualization(times, f0, max_points=15)
    assert payload["available"] is True
    assert payload["vertical_unit"] == "semitone_relative_to_speaker_median"
    assert payload["interpretation"] == "visualization_only_not_pitch_accent_correctness"
    assert payload["speaker_median_f0_hz"] > 0
    assert len(payload["points"]) <= 15
    assert all(
        point["relative_semitone"] is None or -12 <= point["relative_semitone"] <= 12
        for point in payload["points"]
    )


def test_karaoke_timeline_offsets_trim_relative_evidence_to_original_playback():
    timeline = build_consumer_karaoke_timeline(_result(), _user_facing())
    assert timeline["timebase"] == "original_recording_playback_seconds"
    assert timeline["duration_sec"] == 2.8
    assert timeline["sync_mode"] == "word_timestamps"

    first_word = timeline["words"][0]
    assert first_word["start_sec"] == 0.5
    assert first_word["end_sec"] == 0.92
    assert first_word["asr_confidence"] == "high"
    assert first_word["interpretation"] == "playback_alignment_not_phone_correctness"

    pause = timeline["pauses"][0]
    assert pause["start_sec"] == 1.2
    assert pause["end_sec"] == 1.55
    assert pause["interpretation"] == "acoustic_pause_not_automatically_a_fluency_error"

    pitch = timeline["pitch"]
    assert pitch["available"] is True
    assert pitch["points"][0]["t_sec"] == 0.5
    assert pitch["contextual_intonation_claim"] is False


def test_missing_word_timestamps_never_fabricates_character_timing():
    timeline = build_consumer_karaoke_timeline(_result(with_words=False), _user_facing())
    assert timeline["transcript"] == "今日はいい天気ですね"
    assert timeline["sync_mode"] == "sentence_progress_only"
    assert timeline["words"] == []
    assert "実時間スタンプがない" in timeline["sync_note"]


def test_dimension_contract_uses_exact_four_product_semantics_and_evidence_states():
    timeline = build_consumer_karaoke_timeline(_result(), _user_facing())
    dimensions = timeline["dimensions"]
    assert [item["key"] for item in dimensions] == list(CANONICAL_DIMENSION_ORDER)
    assert [item["label"] for item in dimensions] == ["流暢さ", "明瞭さ", "リズム", "抑揚"]
    clarity = next(item for item in dimensions if item["key"] == "clarity")
    assert clarity["score"] == 70
    assert clarity["evidence_state"] == "neutral_prior"
    assert clarity["numeric_semantics"] == "neutral_anchor_not_direct_measurement"
    assert timeline["recording"]["is_product_dimension"] is False


def test_guardrails_explicitly_reject_fake_karaoke_correctness_claims():
    timeline = build_consumer_karaoke_timeline(_result(), _user_facing())
    assert timeline["score_role"] == "visualization_only"
    assert timeline["product_score_changed"] is False
    assert timeline["guardrails"] == {
        "word_timestamps_are_phone_correctness": False,
        "pitch_curve_is_lexical_pitch_accent_correctness": False,
        "pause_is_automatically_an_error": False,
        "recording_quality_is_a_speaking_dimension": False,
        "missing_evidence_should_be_rendered_as_bad_performance": False,
    }
