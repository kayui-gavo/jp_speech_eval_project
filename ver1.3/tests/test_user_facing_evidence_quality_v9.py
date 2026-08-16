from __future__ import annotations

from jp_speech_eval.feedback_renderer import render_user_facing_result


def test_user_facing_separates_clean_recording_from_weak_four_score_evidence() -> None:
    result = {
        "total_score": 72,
        "pronunciation_score": 70,
        "prosody_score": 70,
        "fluency_score": 78,
        "tone_score": 70,
        "alignment_mode": "none",
        "target_text": "今日はいい天気です",
        "moras": ["キョ", "ウ", "ワ", "イ", "イ", "テ", "ン", "キ", "デ", "ス"],
        "mora_table": [],
        "feedback": [],
        "details": {
            "mode": "transcript_assisted_light",
            "score_eligible": True,
            "content_match": {"status": "general_japanese"},
            "recording_quality": {"score": 0.96},
            "reliability": {
                "level": "high",
                "overall": 0.95,
                "recording_quality": 0.96,
                "endpointing": 0.94,
                "alignment": 1.0,
                "f0_coverage": 0.0,
            },
            "fluency": {},
            "transcript_sanity": {"ok": True},
        },
    }

    user = render_user_facing_result(result, mode="transcript_assisted_light")

    assert user["display_score"] is not None
    assert user["recording_analyzability"]["level"] == "high"
    assert user["score_evidence"]["level"] == "low"
    assert user["score_evidence"]["neutral_prior_dimension_count"] == 3
    assert user["score_evidence"]["interpretation"] == "evidence_coverage_not_probability_score_is_correct"
