from __future__ import annotations

import json

from scripts.audit_reference_boundary_provenance import audit_reference_boundaries, inspect_cache_json


def _cache_payload(**overrides):
    payload = {
        "text": "あい",
        "kana": "アイ",
        "moras": ["ア", "イ"],
        "sr": 16000,
        "ref_mora_boundaries": [[0.0, 0.2], [0.2, 0.4]],
        "ref_boundary_method": "equal_mora",
        "reference_source": "test",
    }
    payload.update(overrides)
    return payload


def test_inventory_classifies_legacy_equal_cache_as_needing_migration(tmp_path) -> None:
    path = tmp_path / "equal.json"
    path.write_text(json.dumps(_cache_payload(), ensure_ascii=False), encoding="utf-8")

    row = inspect_cache_json(path)
    assert row is not None
    assert row["boundary_tier"] == "equal_fallback"
    assert row["boundary_confidence"] < 0.30
    assert row["migration_state"] == "needs_alignment_migration"


def test_inventory_counts_verified_cache_and_ignores_non_cache_json(tmp_path) -> None:
    verified = tmp_path / "verified.json"
    verified.write_text(
        json.dumps(
            _cache_payload(
                ref_boundary_method="external_lab",
                ref_boundary_tier="verified_phone_alignment",
                ref_boundary_confidence=0.85,
                ref_boundary_source="reference.lab",
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

    rows, summary = audit_reference_boundaries([tmp_path])
    assert len(rows) == 1
    assert rows[0]["migration_state"] == "verified"
    assert summary["cache_count"] == 1
    assert summary["verified_count"] == 1
    assert summary["verified_rate"] == 1.0
    assert summary["ready_for_precise_local_feedback_count"] == 1


def test_inventory_rejects_false_verified_state_when_boundary_count_is_wrong(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            _cache_payload(
                ref_mora_boundaries=[[0.0, 0.4]],
                ref_boundary_method="external_lab",
                ref_boundary_tier="verified_phone_alignment",
                ref_boundary_confidence=0.9,
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    row = inspect_cache_json(path)
    assert row is not None
    assert row["one_to_one"] is False
    assert row["migration_state"] == "review_required"
