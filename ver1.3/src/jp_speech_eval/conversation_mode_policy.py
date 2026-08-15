"""Product routing contract for conversation-vs-deep-review speech evaluation.

The user-facing product should not expose internal evaluator names such as
``transcript_assisted_light`` or ``asr_confirmed_weak_reference`` as if they
were interchangeable scoring systems. This module defines the intended product
workflow without changing any acoustic score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


CONVERSATION_MODE_POLICY_VERSION = "conversation_mode_policy_v1"

INSTANT_CONVERSATION = "instant_conversation"
DEEP_REVIEW = "deep_review"
FIXED_PRACTICE = "fixed_practice"


@dataclass(frozen=True)
class ConversationEvaluationPlan:
    product_mode: str
    evaluator_mode: Optional[str]
    action: str
    requires_user_confirmation: bool
    target_relative_local_feedback_allowed: bool
    strict_pitch_accent_allowed: bool
    reference_strength: str
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": CONVERSATION_MODE_POLICY_VERSION,
            **asdict(self),
        }


def plan_conversation_evaluation(
    product_mode: str,
    *,
    has_fixed_target: bool = False,
    has_verified_fixed_reference: bool = False,
    has_user_confirmed_text: bool = False,
) -> ConversationEvaluationPlan:
    """Resolve a product intent to an existing safe evaluator workflow.

    Instant conversation never generates a scoring TTS reference behind the
    user's back. Deep review requires confirmed text before using the existing
    weak-reference path. Fixed practice can expose target-local detail only
    when the product actually has a fixed target/reference workflow.
    """

    mode = str(product_mode or "").strip().lower()
    if mode == INSTANT_CONVERSATION:
        return ConversationEvaluationPlan(
            product_mode=mode,
            evaluator_mode="transcript_assisted_light",
            action="evaluate_now",
            requires_user_confirmation=False,
            target_relative_local_feedback_allowed=False,
            strict_pitch_accent_allowed=False,
            reference_strength="none",
            note="broad_general_japanese_four_score;no_local_kana_error_claims",
        )

    if mode == DEEP_REVIEW:
        if not has_user_confirmed_text:
            return ConversationEvaluationPlan(
                product_mode=mode,
                evaluator_mode=None,
                action="request_transcript_confirmation",
                requires_user_confirmation=True,
                target_relative_local_feedback_allowed=False,
                strict_pitch_accent_allowed=False,
                reference_strength="none_pending_confirmation",
                note="do_not_generate_scoring_reference_from_unconfirmed_asr",
            )
        return ConversationEvaluationPlan(
            product_mode=mode,
            evaluator_mode="asr_confirmed_weak_reference",
            action="evaluate_now",
            requires_user_confirmation=True,
            target_relative_local_feedback_allowed=False,
            strict_pitch_accent_allowed=False,
            reference_strength="confirmed_text_tts_weak_reference",
            note="deep_review_can_add_alignment_evidence_but_weak_tts_is_not_ground_truth",
        )

    if mode == FIXED_PRACTICE:
        if not has_fixed_target:
            raise ValueError("fixed_practice requires a fixed target")
        return ConversationEvaluationPlan(
            product_mode=mode,
            evaluator_mode="reference",
            action="evaluate_now",
            requires_user_confirmation=False,
            target_relative_local_feedback_allowed=bool(has_verified_fixed_reference),
            strict_pitch_accent_allowed=False,
            reference_strength=(
                "verified_fixed_reference" if has_verified_fixed_reference else "fixed_reference_unverified"
            ),
            note=(
                "local_detail_still_depends_on_alignment_and_construct_specific_reliability_gates"
            ),
        )

    raise ValueError(
        f"unknown product conversation mode {product_mode!r}; "
        f"expected {INSTANT_CONVERSATION}|{DEEP_REVIEW}|{FIXED_PRACTICE}"
    )
