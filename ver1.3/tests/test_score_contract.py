from __future__ import annotations

from jp_speech_eval.score_contract import (
    SCORE_CONTRACT_VERSION,
    apply_display_transform,
    comparison_context,
    history_comparability,
    mode_family,
    score_contract_payload,
)


def test_score_contract_payload_is_single_product_truth():
    payload = score_contract_payload()
    assert payload["version"] == SCORE_CONTRACT_VERSION
    assert payload["weights"] == {
        "clarity": 0.30,
        "mora_timing": 0.25,
        "delivery_fluency": 0.25,
        "intonation": 0.20,
    }
    assert payload["display_transform"]["formula"] == "70 + 1.08 * (raw - 70)"
    assert payload["product_calibrated"] is False


def test_display_transform_preserves_existing_contract_values():
    assert apply_display_transform(70.0) == 70.0
    assert round(apply_display_transform(80.0), 6) == 80.8
    assert apply_display_transform(120.0) == 100.0


def test_mode_family_separates_fixed_and_general_japanese():
    assert mode_family("reference_based") == "fixed_reference"
    assert mode_family("reference_mismatch_general_japanese") == "general_japanese"
    assert mode_family("transcript_assisted_light") == "general_japanese"


def test_legacy_history_is_not_compared_as_progress():
    current = {
        "score_contract_version": SCORE_CONTRACT_VERSION,
        "mode_family": "general_japanese",
    }
    no_previous = history_comparability(current, None)
    assert no_previous == {"comparable": False, "reason": "no_previous_record"}

    legacy_empty = history_comparability(current, {})
    assert legacy_empty == {
        "comparable": False,
        "reason": "legacy_record_without_score_contract",
    }

    legacy = history_comparability(current, {"mode_family": "general_japanese"})
    assert legacy["comparable"] is False
    assert legacy["reason"] == "legacy_record_without_score_contract"


def test_fixed_reference_progress_requires_same_target_and_reference_identity():
    raw = {
        "target_text": "ラーメンをください",
        "cache_prefix": "cache/ref_v1",
        "details": {"mode": "reference_based"},
    }
    current = comparison_context(raw)
    same = dict(current)
    assert history_comparability(current, same)["comparable"] is True

    changed_reference = dict(same)
    changed_reference["reference_id"] = "cache/ref_v2"
    decision = history_comparability(current, changed_reference)
    assert decision["comparable"] is False
    assert decision["reason"] == "fixed_reference_changed"

    changed_target = dict(same)
    changed_target["target_text"] = "おはようございます"
    decision = history_comparability(current, changed_target)
    assert decision["comparable"] is False
    assert decision["reason"] == "fixed_target_mismatch"
