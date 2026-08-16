from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_research_evidence import analyze_files  # noqa: E402
from export_free_speech_v4_evidence import export_file  # noqa: E402
from normalize_consumer_ratings import normalize_file  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _raw_result(sample_offset: float) -> dict:
    clarity_index = 0.72 + sample_offset
    return {
        "fluency_score": 68 + int(round(sample_offset * 100)),
        "details": {
            "asr": {"provider": "faster-whisper", "model": "small"},
            "fluency": {"speech_rate_mora_per_sec": 4.2 + sample_offset},
            "acoustic_features": {"f0_method": "test_f0"},
            "shadow": {
                "free_speech_dimension_evidence": {
                    "schema_version": "free_speech_dimension_evidence_v1",
                    "clarity": {
                        "asr_recoverability_index_0to1": clarity_index,
                        "word_probability": {"median": clarity_index + 0.02},
                    },
                    "rhythm": {
                        "local_tempo_irregularity_mad_log_sec_per_mora": 0.18 - sample_offset,
                        "local_tempo_spread_p90_p10_log_sec_per_mora": 0.42 - sample_offset,
                    },
                    "intonation": {
                        "robust_range_semitones_p90_p10": 4.0 + 10.0 * sample_offset,
                    },
                },
                "free_speech_candidate_surface": {
                    "policy_id": "free_speech_candidate_surface_v1_shadow",
                    "component_candidates": {
                        "clarity": 70 + 20 * sample_offset,
                        "mora_timing": 70 + 18 * sample_offset,
                        "delivery_fluency": 70 + 15 * sample_offset,
                        "intonation": 70 + 16 * sample_offset,
                    },
                },
            },
        },
    }


def test_exporter_flattens_raw_and_acceptance_wrapped_results(tmp_path: Path) -> None:
    input_path = tmp_path / "results.jsonl"
    output_path = tmp_path / "evidence.csv"
    _write_jsonl(
        input_path,
        [
            {"sample_id": "s1", "raw_result": _raw_result(0.01)},
            {"sample_id": "s2", "response": {"raw_result": _raw_result(0.05)}},
        ],
    )
    report = export_file(input_path, output_path)
    assert report["sample_count"] == 2
    assert report["candidate_count"] == 10
    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8")))
    assert len(rows) == 20
    assert {row["sample_id"] for row in rows} == {"s1", "s2"}
    assert any(row["candidate"] == "clarity.asr_recoverability_index" for row in rows)
    assert any(row["candidate"] == "rhythm.negative_local_tempo_mad" for row in rows)
    assert any(row["candidate"] == "intonation.shadow_candidate_score" for row in rows)


def _rating_row(rater: str, sample: str, presentation: str, value: int) -> dict:
    return {
        "rater_id": rater,
        "sample_id": sample,
        "presentation_id": presentation,
        "task_mode": "spontaneous",
        "assigned_constructs": "clarity_comprehensibility|fluency|rhythm_naturalness|intonation_naturalness",
        "analyzable_yes_no": "yes",
        "clarity_comprehensibility_1to7": value,
        "fluency_1to7": value,
        "rhythm_naturalness_1to7": value,
        "intonation_naturalness_1to7": value,
        "intonation_context_available": "true",
        "timestamp": "2026-08-16T12:00:00+09:00",
    }


def test_wide_listener_ratings_normalize_without_averaging_constructs(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    normalized_path = tmp_path / "ratings_long.csv"
    fields = list(_rating_row("r1", "s1", "p1", 4))
    with ratings_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(_rating_row("r1", "s1", "p1", 4))
        writer.writerow(_rating_row("r2", "s1", "p2", 5))
    report = normalize_file(ratings_path, normalized_path)
    assert report["normalized_rating_count"] == 8
    rows = list(csv.DictReader(normalized_path.open("r", encoding="utf-8")))
    assert len(rows) == 8
    assert set(row["criterion"] for row in rows) == {
        "clarity_comprehensibility",
        "fluency",
        "rhythm_naturalness",
        "intonation_naturalness",
    }
    assert all("human_rating" in row for row in rows)


def test_normalized_ratings_and_exported_evidence_feed_existing_analyzer(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    evidence_path = tmp_path / "evidence.csv"
    ratings_path = tmp_path / "ratings.csv"
    normalized_path = tmp_path / "ratings_long.csv"

    offsets = {"s1": 0.00, "s2": 0.04, "s3": 0.08}
    _write_jsonl(results_path, [{"sample_id": sid, "raw_result": _raw_result(offset)} for sid, offset in offsets.items()])
    export_file(results_path, evidence_path)

    fields = list(_rating_row("r1", "s1", "p1", 3))
    with ratings_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        presentation = 0
        for sample_index, sample_id in enumerate(offsets, start=3):
            for rater_index in range(1, 4):
                presentation += 1
                writer.writerow(
                    _rating_row(
                        f"r{rater_index}",
                        sample_id,
                        f"p{presentation}",
                        min(7, sample_index),
                    )
                )
    normalize_file(ratings_path, normalized_path)
    report = analyze_files(normalized_path, evidence_path)
    assert "clarity_comprehensibility" in report["criteria"]
    clarity = report["results"]["clarity_comprehensibility"]["clarity.asr_recoverability_index"]
    assert clarity["availability_rate"] == 1.0
    assert clarity["overall_spearman"]["n"] == 3
    assert clarity["overall_spearman"]["rho"] is not None


def test_promotion_protocol_blocks_shadow_shortcuts() -> None:
    protocol_path = ROOT / "data" / "research_eval" / "free_speech_v4_promotion_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_before_criterion_results"
    assert protocol["minimum_promotion_gates"]["human_criterion_required"] is True
    assert protocol["product_ux_gate"]["promotion_requires_new_score_contract_version"] is True
    assert "do_not_tune_thresholds_or_shrinkage_on_the_held_set" in protocol["anti_leakage_rules"]
    assert "do_not_use_language_probability_as_a_clarity_score" in protocol["anti_leakage_rules"]
    assert protocol["minimum_promotion_gates"]["negative_controls_must_route_to_no_score_or_retry_when_confidently_non_japanese_or_unusable"] is True
