from __future__ import annotations

import pytest

from jp_speech_eval.conversation_mode_policy import (
    DEEP_REVIEW,
    FIXED_PRACTICE,
    INSTANT_CONVERSATION,
    plan_conversation_evaluation,
)


def test_instant_conversation_never_generates_hidden_scoring_reference():
    plan = plan_conversation_evaluation(INSTANT_CONVERSATION)
    assert plan.evaluator_mode == "transcript_assisted_light"
    assert plan.action == "evaluate_now"
    assert plan.requires_user_confirmation is False
    assert plan.target_relative_local_feedback_allowed is False
    assert plan.reference_strength == "none"


def test_deep_review_requires_confirmation_before_weak_reference():
    pending = plan_conversation_evaluation(DEEP_REVIEW, has_user_confirmed_text=False)
    assert pending.evaluator_mode is None
    assert pending.action == "request_transcript_confirmation"
    assert pending.requires_user_confirmation is True

    confirmed = plan_conversation_evaluation(DEEP_REVIEW, has_user_confirmed_text=True)
    assert confirmed.evaluator_mode == "asr_confirmed_weak_reference"
    assert confirmed.reference_strength == "confirmed_text_tts_weak_reference"
    assert confirmed.target_relative_local_feedback_allowed is False
    assert confirmed.strict_pitch_accent_allowed is False


def test_fixed_practice_local_detail_requires_verified_reference():
    weak = plan_conversation_evaluation(
        FIXED_PRACTICE,
        has_fixed_target=True,
        has_verified_fixed_reference=False,
    )
    strong = plan_conversation_evaluation(
        FIXED_PRACTICE,
        has_fixed_target=True,
        has_verified_fixed_reference=True,
    )
    assert weak.evaluator_mode == "reference"
    assert weak.target_relative_local_feedback_allowed is False
    assert strong.target_relative_local_feedback_allowed is True
    # Strict lexical pitch accent remains a separate downstream reliability
    # decision; a verified waveform alone is not enough.
    assert strong.strict_pitch_accent_allowed is False


def test_fixed_practice_requires_target():
    with pytest.raises(ValueError, match="requires a fixed target"):
        plan_conversation_evaluation(FIXED_PRACTICE)
