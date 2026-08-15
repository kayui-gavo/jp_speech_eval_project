from __future__ import annotations

import math

from scripts.analyze_research_evidence import (
    aggregate_human_ratings,
    analyze_joined_rows,
    join_human_and_evidence,
)


def _human_rows():
    rows = []
    ratings = {
        "s1": [2, 2],
        "s2": [4, 4],
        "s3": [6, 6],
        "s4": [7, 7],
    }
    for sample_index, (sample_id, values) in enumerate(ratings.items(), start=1):
        for rater_index, value in enumerate(values, start=1):
            rows.append(
                {
                    "sample_id": sample_id,
                    "criterion": "clarity_comprehensibility",
                    "human_rating": str(value),
                    "rater_id": f"r{rater_index}",
                    "speaker_id": f"spk{sample_index}",
                    "target_id": "t1" if sample_index <= 2 else "t2",
                    "task": "fixed_reading",
                    "speaker_fold": str((sample_index - 1) % 2),
                    "target_fold": "0" if sample_index <= 2 else "1",
                }
            )
    return rows


def test_human_ratings_are_aggregated_before_machine_correlation():
    aggregated = aggregate_human_ratings(_human_rows())
    assert len(aggregated) == 4
    first = next(row for row in aggregated if row["sample_id"] == "s1")
    assert first["human_rating_mean"] == 2.0
    assert first["human_rating_count"] == 2


def test_positive_and_distance_like_candidates_keep_raw_direction_visible():
    human = aggregate_human_ratings(_human_rows())
    evidence = []
    for sample_id, positive, distance in [
        ("s1", 0.2, 0.8),
        ("s2", 0.4, 0.6),
        ("s3", 0.6, 0.4),
        ("s4", 0.8, 0.2),
    ]:
        evidence.extend(
            [
                {
                    "sample_id": sample_id,
                    "candidate": "positive_support",
                    "evidence_value": str(positive),
                    "available": "true",
                    "evidence_direction": "higher_is_better",
                },
                {
                    "sample_id": sample_id,
                    "candidate": "distance",
                    "evidence_value": str(distance),
                    "available": "true",
                    "evidence_direction": "lower_is_better",
                },
            ]
        )
    report = analyze_joined_rows(join_human_and_evidence(human, evidence))
    positive = report["results"]["clarity_comprehensibility"]["positive_support"]["overall_spearman"]
    distance = report["results"]["clarity_comprehensibility"]["distance"]["overall_spearman"]
    assert math.isclose(positive["rho"], 1.0)
    assert math.isclose(distance["rho"], -1.0)
    # Analyzer reports raw association. It does not silently flip distances or
    # choose the nicer-looking sign after observing results.
    assert report["interpretation"].startswith("raw construct-matched")


def test_missing_evidence_reduces_availability_instead_of_becoming_bad_score():
    human = aggregate_human_ratings(_human_rows())
    evidence = [
        {
            "sample_id": "s1",
            "candidate": "candidate_a",
            "evidence_value": "0.2",
            "available": "true",
        },
        {
            "sample_id": "s2",
            "candidate": "candidate_a",
            "evidence_value": "",
            "available": "false",
            "failure_reason": "f0_unavailable",
        },
        {
            "sample_id": "s3",
            "candidate": "candidate_a",
            "evidence_value": "0.7",
            "available": "true",
        },
    ]
    report = analyze_joined_rows(join_human_and_evidence(human, evidence))
    result = report["results"]["clarity_comprehensibility"]["candidate_a"]
    assert result["row_count"] == 4
    assert result["available_count"] == 2
    assert result["availability_rate"] == 0.5
    assert result["failure_reasons"]["f0_unavailable"] == 1
    assert result["failure_reasons"]["missing_evidence_row"] == 1
    assert result["overall_spearman"]["rho"] is None
    assert result["overall_spearman"]["reason"] == "fewer_than_3_usable_pairs"


def test_conflicting_metadata_for_same_sample_criterion_is_rejected():
    rows = _human_rows()
    rows[1] = {**rows[1], "speaker_id": "different"}
    try:
        aggregate_human_ratings(rows)
    except ValueError as exc:
        assert "conflicting speaker_id" in str(exc)
    else:
        raise AssertionError("conflicting metadata must fail closed")
