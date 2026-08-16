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


def _fixed_result(*, approximate: bool = False):
    return {
        "target_text": "おはようございます",
        "duration_sec": 1.4,
        "alignment_mode": "cached_dtw_fallback_equal" if approximate else "cached_dtw",
        "endpointing": {
            "raw_duration": 2.0,
            "speech_start": 0.30,
            "speech_end": 1.70,
            "speech_duration": 1.4,
            "detected": True,
        },
        "pause_info": {"pause_segments": []},
        "mora_table": [
            {"mora": "オ", "start_sec": 0.00, "end_sec": 0.25, "f0_hz": 185.0},
            {"mora": "ハ", "start_sec": 0.25, "end_sec": 0.50, "f0_hz": 205.0},
            {"mora": "ヨ", "start_sec": 0.50, "end_sec": 0.75, "f0_hz": 220.0},
            {"mora": "ー", "start_sec": 0.75, "end_sec": 1.00, "f0_hz": 210.0},
            {"mora": "ゴ", "start_sec": 1.00, "end_sec": 1.20, "f0_hz": 195.0},
            {"mora": "ザ", "start_sec": 1.20, "end_sec": 1.40, "f0_hz": 180.0},
        ],
        "details": {
            "alignment": {
                "confidence": 0.35 if approximate else 0.88,
                "used_equal_fallback": approximate,
                "mode": "cached_dtw_fallback_equal" if approximate else "cached_dtw",
            },
            "reference_f0_by_mora": [160.0, 180.0, 195.0, 190.0, 175.0, 155.0],
            "recording_quality": {"score": 0.95},
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
    assert pitch["reference_points"] == []
    assert pitch["contextual_intonation_claim"] is False


def test_missing_word_timestamps_never_fabricates_character_timing():
    timeline = build_consumer_karaoke_timeline(_result(with_words=False), _user_facing())
    assert timeline["transcript"] == "今日はいい天気ですね"
    assert timeline["sync_mode"] == "sentence_progress_only"
    assert timeline["words"] == []
    assert timeline["moras"] == []
    assert "単語・モーラ境界がない" in timeline["sync_note"]


def test_fixed_practice_uses_user_mora_boundaries_for_karaoke_sync():
    timeline = build_consumer_karaoke_timeline(_fixed_result(), _user_facing())
    assert timeline["sync_mode"] == "mora_alignment"
    assert timeline["words"] == []
    assert [item["text"] for item in timeline["moras"]] == ["オ", "ハ", "ヨ", "ー", "ゴ", "ザ"]
    assert timeline["moras"][0]["start_sec"] == 0.3
    assert timeline["moras"][0]["end_sec"] == 0.55
    assert timeline["moras"][0]["approximate"] is False
    assert timeline["moras"][0]["interpretation"] == "mora_playback_alignment_not_phone_correctness"


def test_fixed_practice_marks_equal_fallback_mora_sync_as_approximate():
    timeline = build_consumer_karaoke_timeline(_fixed_result(approximate=True), _user_facing())
    assert timeline["sync_mode"] == "mora_alignment_approximate"
    assert all(item["approximate"] for item in timeline["moras"])
    assert "概算" in timeline["sync_note"]


def test_fixed_pitch_overlay_normalizes_user_and_reference_independently():
    timeline = build_consumer_karaoke_timeline(_fixed_result(), _user_facing())
    pitch = timeline["pitch"]
    assert pitch["available"] is True
    assert pitch["reference_available"] is True
    assert pitch["vertical_unit"] == "semitone_relative_to_each_speaker_median"
    assert len(pitch["points"]) == len(pitch["reference_points"]) == 6
    assert pitch["user_median_f0_hz"] != pitch["reference_median_f0_hz"]
    assert pitch["reference_time_mapping"] == "reference_mora_shape_mapped_to_user_mora_midpoints"
    assert "not_strict_pitch_accent_correctness" in pitch["interpretation"]


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
        "mora_alignment_is_phone_correctness": False,
        "pitch_curve_is_lexical_pitch_accent_correctness": False,
        "pause_is_automatically_an_error": False,
        "recording_quality_is_a_speaking_dimension": False,
        "missing_evidence_should_be_rendered_as_bad_performance": False,
    }
