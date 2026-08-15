from __future__ import annotations

from jp_speech_eval.user_score_policy import apply_user_score_policy


def _broad_result() -> dict:
    return {
        "pronunciation_score": 12,
        "prosody_score": 18,
        "fluency_score": 84,
        "moras": ["キョ", "ウ", "ハ", "イ", "イ", "テ", "ン", "キ", "デ", "ス"],
        "alignment_mode": "transcript_assisted_light",
        "details": {
            "mode": "transcript_assisted_light",
            "fallback_reason": "target_mismatch_but_plausible_japanese",
            "content_match": {
                "status": "general_japanese",
                "content_verified": False,
                "japanese_content_plausible": True,
                "original_fixed_reference_status": "fail",
            },
            "reliability": {
                "level": "medium",
                "overall": 0.75,
                "alignment": 0.7,
                "f0_coverage": 0.7,
                "endpointing": 1.0,
            },
            "recording_quality": {"score": 0.9},
            "alignment": {"available": False, "mode": "transcript_assisted_light"},
            "fluency": {"rate_score": 82, "pause_score": 88, "delivery_fluency_score": 85},
            "tone": {"pitch_range_log": 0.25, "pitch_score": 84},
            "prosody": {},
            "shadow": {},
        },
    }


def test_effective_broad_mode_keeps_nonpunitive_fixed_prompt_mismatch_notice() -> None:
    policy = apply_user_score_policy(
        _broad_result(),
        mode="reference_mismatch_general_japanese",
    )
    assert policy["score_available"] is True
    assert policy["display_score"] is not None
    assert policy["main_message_key"] == "content_mismatch_general_score"
    assert policy["scoring_gate"]["target_match_ok"] is False
    assert policy["scoring_gate"]["user_message_type"] == "content_mismatch_general_score"
    assert "target_content_mismatch_general_score" in policy["score_policy_warnings"]
    assert policy["detail_feedback_allowed"] is False
    assert policy["inputs"]["broad_target_mismatch"] is True
    # The old fixed-target timing/prosody values are deliberately awful. Broad
    # Japanese mode must not inherit them as target-relative punishment.
    assert policy["component_scores"]["mora_timing"]["value"] != 12
    assert policy["component_scores"]["intonation"]["value"] != 18


def test_general_japanese_provenance_also_preserves_notice_if_mode_alias_is_lost() -> None:
    policy = apply_user_score_policy(_broad_result(), mode="transcript_assisted_light")
    assert policy["display_score"] is not None
    assert policy["main_message_key"] == "content_mismatch_general_score"
    assert policy["inputs"]["broad_target_mismatch"] is True
