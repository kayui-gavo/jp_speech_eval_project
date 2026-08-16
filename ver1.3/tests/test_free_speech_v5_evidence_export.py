from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.export_free_speech_v5_evidence import CANDIDATES, export_file


def _batch_row(*, status: str, clarity: float = 0.9, intonation_available: bool = True):
    row = {
        "run_schema": "free_speech_validation_batch_v1",
        "sample_id": "s1",
        "metadata": {
            "speaker_id": "spk1",
            "speaker_group": "learner",
            "l1": "zh",
            "task_mode": "spontaneous",
            "prompt_id": "p1",
            "split": "held",
            "expected_language": "ja",
            "channel_condition": "clean",
            "channel_pair_id": "pair1",
            "source_recording_id": "source1",
            "context_type": "prompt",
            "context_id": "ctx1",
            "source_note": "test",
        },
        "audio_path": "s1.wav",
        "scoring_used_gold_transcript": False,
        "status": status,
    }
    if status == "ok":
        row["raw_result"] = {
            "fluency_score": 74,
            "details": {
                "asr": {"provider": "faster-whisper", "model": "small"},
                "language_gate": {"eligible": True, "reason": "ja"},
                "recording_quality": {"score": 0.9},
                "fluency": {"speech_rate_mora_per_sec": 4.8},
                "shadow": {
                    "free_speech_dimension_evidence": {
                        "schema_version": "free_speech_dimension_evidence_v1",
                        "clarity": {
                            "asr_recoverability_index_0to1": clarity,
                            "word_probability": {"median": 0.91},
                        },
                        "rhythm": {
                            "local_tempo_irregularity_mad_log_sec_per_mora": 0.1,
                            "local_tempo_spread_p90_p10_log_sec_per_mora": 0.2,
                        },
                        "intonation": {
                            "robust_range_semitones_p90_p10": 6.0 if intonation_available else None
                        },
                    },
                    "free_speech_candidate_surface": {
                        "policy_id": "free_speech_candidate_surface_v1_shadow",
                        "available_evidence": {
                            "clarity": True,
                            "mora_timing": True,
                            "delivery_fluency": True,
                            "intonation": intonation_available,
                        },
                        "component_candidates": {
                            "clarity": 77.0,
                            "mora_timing": 72.0,
                            "delivery_fluency": 74.0,
                            "intonation": 77.0 if intonation_available else 70.0,
                        },
                    },
                },
            },
        }
        row["user_score"] = {
            "score_available": True,
            "score_contract_version": "consumer_four_score_v2",
            "evidence_schema_version": "consumer_evidence_v3",
        }
    else:
        row["error_type"] = "RuntimeError"
        row["error"] = "temporary failure"
    return row


def test_export_uses_latest_attempt_and_keeps_shadow_policy_provenance(tmp_path):
    source = tmp_path / "batch.jsonl"
    output = tmp_path / "evidence.csv"
    attempts = [_batch_row(status="error"), _batch_row(status="ok", clarity=0.93)]
    source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in attempts) + "\n", encoding="utf-8")

    report = export_file(source, output)
    assert report["attempt_count"] == 2
    assert report["sample_count"] == 1
    assert report["superseded_attempt_count"] == 1
    assert report["error_sample_count"] == 0
    assert report["evidence_row_count"] == len(CANDIDATES)

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["sample_id"] for row in rows} == {"s1"}
    assert {row["source_recording_id"] for row in rows} == {"source1"}
    assert {row["score_contract_version"] for row in rows} == {"consumer_four_score_v2"}
    assert {row["evidence_schema_version"] for row in rows} == {"consumer_evidence_v3"}

    clarity = next(row for row in rows if row["candidate"] == "clarity.asr_recoverability_index")
    assert float(clarity["evidence_value"]) == 0.93
    assert clarity["model_id"] == "faster-whisper"
    assert clarity["model_version"] == "small"

    shadow = next(row for row in rows if row["candidate"] == "clarity.shadow_candidate_score")
    assert shadow["available"] == "true"
    assert float(shadow["evidence_value"]) == 77.0
    assert float(shadow["fallback_numeric_value"]) == 77.0
    assert shadow["model_id"] == "free_speech_candidate_surface"
    assert shadow["model_version"] == "free_speech_candidate_surface_v1_shadow"
    assert shadow["candidate_surface_policy_id"] == "free_speech_candidate_surface_v1_shadow"


def test_neutral_intonation_placeholder_does_not_count_as_available_evidence(tmp_path):
    source = tmp_path / "batch.jsonl"
    output = tmp_path / "evidence.csv"
    source.write_text(
        json.dumps(_batch_row(status="ok", intonation_available=False), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    export_file(source, output)
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    raw_f0 = next(row for row in rows if row["candidate"] == "intonation.robust_range_semitones")
    shadow = next(row for row in rows if row["candidate"] == "intonation.shadow_candidate_score")
    assert raw_f0["available"] == "false"
    assert raw_f0["evidence_value"] == ""
    assert shadow["available"] == "false"
    assert shadow["evidence_value"] == ""
    assert float(shadow["fallback_numeric_value"]) == 70.0
    assert shadow["failure_reason"] == "neutral_placeholder_without_source_evidence"
