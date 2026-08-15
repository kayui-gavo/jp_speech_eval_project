from __future__ import annotations

import pytest

from scripts.build_research_splits import (
    assign_group_folds,
    split_rows,
    verify_group_disjointness,
)


def _rows():
    rows = []
    for speaker_index in range(10):
        for target_index in range(7):
            rows.append(
                {
                    "sample_id": f"s{speaker_index}_t{target_index}",
                    "speaker_id": f"spk{speaker_index}",
                    "target_id": f"t{target_index}",
                    "target_text": f"target {target_index}",
                    "audio_path": f"audio/{speaker_index}_{target_index}.wav",
                    "task": "fixed_reading",
                    "criterion": "pronunciation_accuracy",
                    "human_rating": "5",
                }
            )
    return rows


def test_group_assignments_are_deterministic_and_group_disjoint():
    rows = _rows()
    first = assign_group_folds(rows, group_key="speaker_id", fold_count=5, seed=33017)
    second = assign_group_folds(rows, group_key="speaker_id", fold_count=5, seed=33017)
    assert first == second
    assert len(first) == 10
    assert set(first.values()) == {0, 1, 2, 3, 4}


def test_split_rows_produces_separate_speaker_and_target_views_without_leakage():
    split, report = split_rows(_rows(), fold_count=5, seed=33017)
    speaker = verify_group_disjointness(split, group_key="speaker_id", fold_key="speaker_fold")
    target = verify_group_disjointness(split, group_key="target_id", fold_key="target_fold")
    assert speaker["ok"] is True
    assert target["ok"] is True
    assert report["speaker_group_count"] == 10
    assert report["target_group_count"] == 7
    assert "two separate evaluation views" in report["note"]


def test_group_split_rejects_more_folds_than_groups():
    with pytest.raises(ValueError, match="fewer than fold_count"):
        assign_group_folds(_rows(), group_key="target_id", fold_count=8)


def test_balancer_does_not_split_large_groups_across_folds():
    rows = _rows()
    rows.extend(
        {
            "sample_id": f"extra_{index}",
            "speaker_id": "spk0",
            "target_id": f"t{index % 7}",
            "target_text": "x",
            "audio_path": "x.wav",
            "task": "fixed_reading",
            "criterion": "pronunciation_accuracy",
            "human_rating": "5",
        }
        for index in range(30)
    )
    split, _ = split_rows(rows, fold_count=5)
    spk0_folds = {row["speaker_fold"] for row in split if row["speaker_id"] == "spk0"}
    assert len(spk0_folds) == 1
