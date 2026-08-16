#!/usr/bin/env python3
"""Export v5 free-speech validation evidence with gate-critical metadata.

Unlike the v4 flattener, this exporter preserves speaker/task/split/channel
metadata from the real-product batch runner and explicit eligibility outcomes.
Batch JSONL may contain multiple attempts for a failed sample; analysis uses the
latest attempt per sample while preserving the raw attempt history on disk.
This script performs no score promotion or fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


EXPORT_SCHEMA = "free_speech_v5_evidence_export_v1"
CANDIDATES = (
    ("clarity.asr_recoverability_index", "clarity_comprehensibility", "higher_better_hypothesis"),
    ("clarity.word_probability_median", "clarity_comprehensibility", "higher_better_hypothesis"),
    ("clarity.shadow_candidate_score", "clarity_comprehensibility", "higher_better_hypothesis"),
    ("rhythm.negative_local_tempo_mad", "rhythm_naturalness", "higher_better_hypothesis"),
    ("rhythm.negative_local_tempo_spread", "rhythm_naturalness", "higher_better_hypothesis"),
    ("rhythm.shadow_candidate_score", "rhythm_naturalness", "higher_better_hypothesis"),
    ("fluency.current_product_proxy", "fluency", "higher_better_hypothesis"),
    ("fluency.speech_rate_mora_per_sec", "fluency", "nonmonotonic_diagnostic_only"),
    ("intonation.robust_range_semitones", "intonation_utterance_naturalness", "nonmonotonic_diagnostic_only"),
    ("intonation.shadow_candidate_score", "intonation_utterance_naturalness", "higher_better_hypothesis"),
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool_text(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def _read_jsonl(path: str | Path) -> Iterable[Mapping[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, Mapping):
                yield value


def _latest_attempts(path: str | Path) -> tuple[list[Mapping[str, Any]], int, int]:
    """Return the last JSONL row for each sample id, preserving final order.

    A retry appends rather than mutates the raw run log.  The analysis view is
    deterministic: latest attempt wins.  Rows without sample ids are ignored.
    """
    latest: OrderedDict[str, Mapping[str, Any]] = OrderedDict()
    attempt_count = 0
    for row in _read_jsonl(path):
        sample_id = _text(row.get("sample_id"))
        if not sample_id:
            continue
        attempt_count += 1
        if sample_id in latest:
            del latest[sample_id]
        latest[sample_id] = row
    unique_count = len(latest)
    return list(latest.values()), attempt_count, attempt_count - unique_count


def _candidate_values(batch_row: Mapping[str, Any]) -> Dict[str, Any]:
    raw = _mapping(batch_row.get("raw_result"))
    details = _mapping(raw.get("details"))
    shadow = _mapping(details.get("shadow"))
    evidence = _mapping(shadow.get("free_speech_dimension_evidence"))
    surface = _mapping(shadow.get("free_speech_candidate_surface"))
    clarity = _mapping(evidence.get("clarity"))
    rhythm = _mapping(evidence.get("rhythm"))
    intonation = _mapping(evidence.get("intonation"))
    fluency = _mapping(details.get("fluency"))
    components = _mapping(surface.get("component_candidates"))
    word_probability = _mapping(clarity.get("word_probability"))
    mad = _finite(rhythm.get("local_tempo_irregularity_mad_log_sec_per_mora"))
    spread = _finite(rhythm.get("local_tempo_spread_p90_p10_log_sec_per_mora"))
    return {
        "clarity.asr_recoverability_index": clarity.get("asr_recoverability_index_0to1"),
        "clarity.word_probability_median": word_probability.get("median"),
        "clarity.shadow_candidate_score": components.get("clarity"),
        "rhythm.negative_local_tempo_mad": None if mad is None else -mad,
        "rhythm.negative_local_tempo_spread": None if spread is None else -spread,
        "rhythm.shadow_candidate_score": components.get("mora_timing"),
        "fluency.current_product_proxy": raw.get("fluency_score"),
        "fluency.speech_rate_mora_per_sec": fluency.get("speech_rate_mora_per_sec"),
        "intonation.robust_range_semitones": intonation.get("robust_range_semitones_p90_p10"),
        "intonation.shadow_candidate_score": components.get("intonation"),
    }


def evidence_rows_for_batch_row(batch_row: Mapping[str, Any]) -> list[Dict[str, Any]]:
    sample_id = _text(batch_row.get("sample_id"))
    meta = _mapping(batch_row.get("metadata"))
    raw = _mapping(batch_row.get("raw_result"))
    details = _mapping(raw.get("details"))
    shadow = _mapping(details.get("shadow"))
    surface = _mapping(shadow.get("free_speech_candidate_surface"))
    evidence = _mapping(shadow.get("free_speech_dimension_evidence"))
    asr = _mapping(details.get("asr"))
    language_gate = _mapping(details.get("language_gate"))
    recording = _mapping(details.get("recording_quality"))
    product = _mapping(batch_row.get("user_score"))
    batch_ok = _text(batch_row.get("status")) == "ok"
    values = _candidate_values(batch_row) if batch_ok else {}
    asr_model_id = _text(asr.get("provider")) or "free_speech_v5"
    asr_model_version = _text(asr.get("model")) or "unknown"
    evidence_schema = _text(product.get("evidence_schema_version")) or _text(evidence.get("schema_version"))
    score_contract = _text(product.get("score_contract_version"))
    surface_policy_id = _text(surface.get("policy_id")) or "unknown_shadow_policy"

    common = {
        "sample_id": sample_id,
        "speaker_id": _text(meta.get("speaker_id")),
        "speaker_group": _text(meta.get("speaker_group")),
        "l1": _text(meta.get("l1")),
        "task": _text(meta.get("task_mode")),
        "prompt_id": _text(meta.get("prompt_id")),
        "subset": _text(meta.get("split")),
        "expected_language": _text(meta.get("expected_language")),
        "condition": _text(meta.get("channel_condition")),
        "channel_pair_id": _text(meta.get("channel_pair_id")),
        "source_recording_id": _text(meta.get("source_recording_id")),
        "context_type": _text(meta.get("context_type")),
        "context_id": _text(meta.get("context_id")),
        "batch_status": _text(batch_row.get("status")),
        "product_score_available": _bool_text(product.get("score_available")),
        "language_gate_eligible": _bool_text(language_gate.get("eligible")),
        "language_gate_reason": _text(language_gate.get("reason")),
        "recording_quality_score": "" if _finite(recording.get("score")) is None else _finite(recording.get("score")),
        "scoring_used_gold_transcript": _bool_text(batch_row.get("scoring_used_gold_transcript")),
        "score_contract_version": score_contract,
        "evidence_schema_version": evidence_schema,
        "candidate_surface_policy_id": surface_policy_id,
        "export_schema": EXPORT_SCHEMA,
    }

    rows: list[Dict[str, Any]] = []
    for candidate, construct, direction in CANDIDATES:
        value = _finite(values.get(candidate)) if batch_ok else None
        if not batch_ok:
            failure_reason = "batch_error:" + (_text(batch_row.get("error_type")) or "unknown")
        elif value is None:
            failure_reason = "evidence_unavailable"
        else:
            failure_reason = ""
        is_shadow_surface = "shadow_candidate_score" in candidate
        rows.append(
            {
                **common,
                "candidate": candidate,
                "candidate_construct": construct,
                "evidence_value": "" if value is None else value,
                "available": "true" if value is not None else "false",
                "failure_reason": failure_reason,
                "evidence_direction": direction,
                "model_id": "free_speech_candidate_surface" if is_shadow_surface else asr_model_id,
                "model_version": surface_policy_id if is_shadow_surface else asr_model_version,
            }
        )
    return rows


def export_file(input_jsonl: str | Path, output_csv: str | Path) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    latest, attempt_count, superseded_attempt_count = _latest_attempts(input_jsonl)
    error_sample_count = 0
    for batch_row in latest:
        if _text(batch_row.get("status")) != "ok":
            error_sample_count += 1
        rows.extend(evidence_rows_for_batch_row(batch_row))

    fields = [
        "sample_id", "speaker_id", "speaker_group", "l1", "task", "prompt_id", "subset",
        "expected_language", "condition", "channel_pair_id", "source_recording_id", "context_type", "context_id",
        "candidate", "candidate_construct", "evidence_value", "available", "failure_reason",
        "evidence_direction", "model_id", "model_version", "batch_status", "product_score_available",
        "language_gate_eligible", "language_gate_reason", "recording_quality_score",
        "scoring_used_gold_transcript", "score_contract_version", "evidence_schema_version",
        "candidate_surface_policy_id", "export_schema",
    ]
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "schema": EXPORT_SCHEMA,
        "attempt_count": attempt_count,
        "sample_count": len(latest),
        "superseded_attempt_count": superseded_attempt_count,
        "error_sample_count": error_sample_count,
        "evidence_row_count": len(rows),
        "candidate_count": len(CANDIDATES),
        "contextual_intonation_candidate_count": 0,
        "note": "latest attempt per sample is exported; isolated acoustic/F0 evidence is intentionally not exported as contextual intonation evidence",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = export_file(args.input_jsonl, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
