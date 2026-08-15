from __future__ import annotations

from scripts.validate_consumer_ratings import REQUIRED_COLUMNS, validate_rating_rows


def _row(**overrides):
    row = {
        "rater_id": "r1",
        "sample_id": "s1",
        "presentation_id": "p1",
        "task_mode": "spontaneous",
        "assigned_constructs": "clarity_comprehensibility|fluency",
        "analyzable_yes_no": "yes",
        "clarity_comprehensibility_1to7": "6",
        "fluency_1to7": "5.0",
        "rhythm_naturalness_1to7": "",
        "intonation_naturalness_1to7": "",
        "intonation_context_available": "false",
        "timestamp": "2026-08-16T02:00:00+09:00",
    }
    row.update(overrides)
    return row


def _fields():
    return list(_row().keys())


def test_valid_balanced_construct_assignment_accepts_integer_like_csv_values():
    report = validate_rating_rows([_row()], _fields())
    assert report["ok"] is True
    assert report["construct_rating_counts"] == {
        "clarity_comprehensibility": 1,
        "fluency": 1,
    }


def test_unassigned_construct_must_not_receive_hidden_rating():
    report = validate_rating_rows(
        [_row(rhythm_naturalness_1to7="4")],
        _fields(),
    )
    assert report["ok"] is False
    assert any("unassigned_construct_must_be_null:rhythm_naturalness" in item for item in report["errors"])


def test_unanalyzable_clip_cannot_be_converted_into_low_construct_scores():
    report = validate_rating_rows(
        [
            _row(
                analyzable_yes_no="no",
                clarity_comprehensibility_1to7="1",
                fluency_1to7="1",
            )
        ],
        _fields(),
    )
    assert report["ok"] is False
    assert any("unanalyzable_must_have_null" in item for item in report["errors"])


def test_assigned_construct_requires_rating_when_analyzable():
    report = validate_rating_rows(
        [_row(clarity_comprehensibility_1to7="")],
        _fields(),
    )
    assert report["ok"] is False
    assert any("assigned_construct_missing_rating:clarity_comprehensibility" in item for item in report["errors"])


def test_intonation_without_context_is_allowed_but_flagged_for_separate_analysis():
    report = validate_rating_rows(
        [
            _row(
                assigned_constructs="intonation_naturalness",
                clarity_comprehensibility_1to7="",
                fluency_1to7="",
                intonation_naturalness_1to7="5",
                intonation_context_available="false",
            )
        ],
        _fields(),
    )
    assert report["ok"] is True
    assert any("intonation_rating_without_context" in item for item in report["warnings"])


def test_rating_schema_requires_construct_assignment_and_analyzability():
    assert "assigned_constructs" in REQUIRED_COLUMNS
    assert "analyzable_yes_no" in REQUIRED_COLUMNS
