#!/usr/bin/env python3
"""Export free-speech v4 shadow evidence to the generic criterion analyzer.

The input may be raw evaluation JSONL or C-end acceptance JSONL containing
``response.raw_result``.  This script performs no inference and no score
promotion: it only flattens already-stored shadow evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


EXPORT_SCHEMA = "free_speech_v4_evidence_export_v1"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _raw_result(row: Mapping[str, Any]) -> Mapping[str, Any]:
    response = _mapping(row.get("response"))
    raw = response.get("raw_result")
    if isinstance(raw, Mapping):
        return raw
    raw = row.get("raw_result")
    if isinstance(raw, Mapping):
        return raw
    return row


def _sample_id(row: Mapping[str, Any], raw: Mapping[str, Any], index: int) -> str:
    for value in (
        row.get("sample_id"),
        row.get("id"),
        _mapping(row.get("case")).get("sample_id"),
        _mapping(row.get("case")).get("id"),
        _mapping(row.get("metadata")).get("sample_id"),
        _mapping(raw.get("details")).get("sample_id"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return f"row_{index:05d}"


def _candidate_row(
    *,
    sample_id: str,
    candidate: str,
    construct: str,
    value: Any,
    direction: str,
    model_id: str,
    model_version: str,
    failure_reason: str = "",
) -> Dict[str, Any]:
    number = _finite(value)
    return {
        "sample_id": sample_id,
        "candidate": candidate,
        "candidate_construct": construct,
        "evidence_value": "" if number is None else number,
        "available": "true" if number is not None else "false",
        "failure_reason": "" if number is not None else (failure_reason or "evidence_unavailable"),
        "evidence_direction": direction,
        "model_id": model_id,
        "model_version": model_version,
        "export_schema": EXPORT_SCHEMA,
    }


def evidence_rows_for_result(sample_id: str, raw: Mapping[str, Any]) -> list[Dict[str, Any]]:
    details = _mapping(raw.get("details"))
    shadow = _mapping(details.get("shadow"))
    evidence = _mapping(shadow.get("free_speech_dimension_evidence"))
    surface = _mapping(shadow.get("free_speech_candidate_surface"))
    clarity = _mapping(evidence.get("clarity"))
    rhythm = _mapping(evidence.get("rhythm"))
    intonation = _mapping(evidence.get("intonation"))
    fluency = _mapping(details.get("fluency"))
    components = _mapping(surface.get("component_candidates"))
    asr = _mapping(details.get("asr"))
    model_id = str(asr.get("provider") or "free_speech_v4")
    model_version = str(asr.get("model") or evidence.get("schema_version") or "unknown")

    word_probability = _mapping(clarity.get("word_probability"))
    rhythm_mad = _finite(rhythm.get("local_tempo_irregularity_mad_log_sec_per_mora"))
    rhythm_spread = _finite(rhythm.get("local_tempo_spread_p90_p10_log_sec_per_mora"))

    rows = [
        _candidate_row(
            sample_id=sample_id,
            candidate="clarity.asr_recoverability_index",
            construct="clarity_comprehensibility",
            value=clarity.get("asr_recoverability_index_0to1"),
            direction="higher_better_hypothesis",
            model_id=model_id,
            model_version=model_version,
        ),
        _candidate_row(
            sample_id=sample_id,
            candidate="clarity.word_probability_median",
            construct="clarity_comprehensibility",
            value=word_probability.get("median"),
            direction="higher_better_hypothesis",
            model_id=model_id,
            model_version=model_version,
        ),
        _candidate_row(
            sample_id=sample_id,
            candidate="clarity.shadow_candidate_score",
            construct="clarity_comprehensibility",
            value=components.get("clarity"),
            direction="higher_better_hypothesis",
            model_id="free_speech_candidate_surface",
            model_version=str(surface.get("policy_id") or "unknown"),
        ),
        _candidate_row(
            sample_id=sample_id,
            candidate="rhythm.negative_local_tempo_mad",
            construct="rhythm_naturalness",
            value=None if rhythm_mad is None else -rhythm_mad,
            direction="higher_better_hypothesis",
            model_id=model_id,
            model_version=model_version,
        ),
        _candidate_row(
            sample_id=sample_id,
            candidate="rhythm.negative_local_tempo_spread",
            construct="rhythm_naturalness",
            value=None if rhythm_spread is None else -rhythm_spread,
            direction="higher_better_hypothesis",
            model_id=model_id,
            model_version=model_version,
        ),
        _candidate_row(
            sample_id=sample_id,
            candidate="rhythm.shadow_candidate_score",
            construct="rhythm_naturalness",
            value=components.get("mora_timing"),
            direction="higher_better_hypothesis",
            model_id="free_speech_candidate_surface",
            model_version=str(surface.get("policy_id") or "unknown"),
        ),
        _candidate_row(
            sample_id=sample_id,
            candidate="fluency.current_product_proxy",
            construct="fluency",
            value=raw.get("fluency_score"),
            direction="higher_better_hypothesis",
            model_id="transcript_assisted_light",
            model_version="legacy_rate_pause_mapping",
        ),
        _candidate_row(
            sample_id=sample_id,
            candidate="fluency.speech_rate_mora_per_sec",
            construct="fluency",
            value=fluency.get("speech_rate_mora_per_sec"),
            direction="nonmonotonic_diagnostic_only",
            model_id="text_frontend_plus_endpointing",
            model_version="mora_rate_v1",
        ),
        _candidate_row(
            sample_id=sample_id,
            candidate="intonation.robust_range_semitones",
            construct="intonation_naturalness",
            value=intonation.get("robust_range_semitones_p90_p10"),
            direction="nonmonotonic_diagnostic_only",
            model_id=str(_mapping(details.get("acoustic_features")).get("f0_method") or "f0"),
            model_version=str(evidence.get("schema_version") or "unknown"),
        ),
        _candidate_row(
            sample_id=sample_id,
            candidate="intonation.shadow_candidate_score",
            construct="intonation_naturalness",
            value=components.get("intonation"),
            direction="higher_better_hypothesis",
            model_id="free_speech_candidate_surface",
            model_version=str(surface.get("policy_id") or "unknown"),
        ),
    ]
    return rows


def read_jsonl(path: str | Path) -> Iterable[Mapping[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, Mapping):
                yield value


def export_file(input_path: str | Path, output_path: str | Path) -> Dict[str, Any]:
    output_rows: list[Dict[str, Any]] = []
    sample_count = 0
    for index, row in enumerate(read_jsonl(input_path), start=1):
        raw = _raw_result(row)
        sample_id = _sample_id(row, raw, index)
        output_rows.extend(evidence_rows_for_result(sample_id, raw))
        sample_count += 1

    fields = [
        "sample_id",
        "candidate",
        "candidate_construct",
        "evidence_value",
        "available",
        "failure_reason",
        "evidence_direction",
        "model_id",
        "model_version",
        "export_schema",
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    return {
        "schema": EXPORT_SCHEMA,
        "input": str(input_path),
        "output": str(output),
        "sample_count": sample_count,
        "evidence_row_count": len(output_rows),
        "candidate_count": len({row["candidate"] for row in output_rows}),
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
