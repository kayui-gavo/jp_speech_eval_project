from __future__ import annotations

import pytest

from jp_speech_eval.free_assessment_stack import (
    discover_free_assessment_stack,
    free_only_policy_ok,
    recommended_free_routes,
)


def test_free_stack_never_default_enables_noncommercial_product_tool():
    stack = discover_free_assessment_stack()
    assert stack["opensmile"].default_enabled is False
    assert stack["opensmile"].product_policy == "excluded_from_commercial_product_default"
    assert free_only_policy_ok() is True


def test_recommended_routes_contain_no_paid_provider():
    routes = recommended_free_routes()
    flattened = " ".join(item for values in routes.values() for item in values).lower()
    assert "azure" not in flattened
    assert "speechsuper" not in flattened
    assert "dolphin" not in flattened
    assert "paid_cloud_pronunciation_apis" in routes["never_auto_enable"]


def test_optional_heavy_aligners_are_never_default_enabled():
    stack = discover_free_assessment_stack()
    assert stack["whisperx"].default_enabled is False
    assert stack["mfa"].default_enabled is False
    assert stack["marine"].default_enabled is False


def test_manual_reading_override_does_not_invent_accent_target():
    pytest.importorskip("pyopenjtalk")
    from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence

    evidence = build_japanese_target_evidence("東京", reading_override="トーキョー")
    assert evidence.reading_source == "manual_override"
    assert evidence.reading_kana == "トーキョー"
    assert evidence.moras
    assert evidence.phones
    assert evidence.accent_source == "unavailable_for_manual_reading_without_verified_accent"
    assert evidence.accent_phrases == []
    assert "manual_reading_override_disables_automatic_accent_target" in evidence.warnings


def test_normal_target_evidence_has_source_provenance():
    pytest.importorskip("pyopenjtalk")
    from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence

    evidence = build_japanese_target_evidence("ラーメン")
    assert evidence.reading_kana
    assert evidence.phones
    assert evidence.moras
    assert evidence.fullcontext_labels
    assert evidence.reading_source in {"verified_target", "pyopenjtalk_g2p"}
    assert evidence.marine_used is False
