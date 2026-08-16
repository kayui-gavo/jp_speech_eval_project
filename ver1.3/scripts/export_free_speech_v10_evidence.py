#!/usr/bin/env python3
"""Export v5 free-speech candidates plus speech-duration metadata for v10 gates.

This intentionally reuses the frozen v5 candidate definitions and availability
semantics.  The only additive field is ``speech_duration_sec`` from the actual
product-condition result, allowing C-end short/long robustness checks without
using a gold transcript.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional

from export_free_speech_v5_evidence import _latest_attempts, evidence_rows_for_batch_row


EXPORT_SCHEMA = "free_speech_v10_evidence_export_v1"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _speech_duration_sec(batch_row: Mapping[str, Any]) -> Optional[float]:
    raw = _mapping(batch_row.get("raw_result"))
    details = _mapping(raw.get("details"))
    endpointing = _mapping(raw.get("endpointing"))
    if not endpointing:
        endpointing = _mapping(details.get("endpointing"))
    for key in ("speech_duration", "speech_duration_sec", "voiced_duration_sec"):
        value = _finite(endpointing.get(key))
        if value is not None and value >= 0:
            return value
    return None


def export_file(input_jsonl: str | Path, output_csv: str | Path) -> dict[str, Any]:
    latest, attempt_count, superseded_attempt_count = _latest_attempts(input_jsonl)
    rows: list[dict[str, Any]] = []
    duration_available_samples = 0
    error_sample_count = 0

    for batch_row in latest:
        if str(batch_row.get("status") or "").strip() != "ok":
            error_sample_count += 1
        duration = _speech_duration_sec(batch_row)
        if duration is not None:
            duration_available_samples += 1
        for row in evidence_rows_for_batch_row(batch_row):
            enriched = dict(row)
            enriched["speech_duration_sec"] = "" if duration is None else duration
            enriched["export_schema"] = EXPORT_SCHEMA
            rows.append(enriched)

    fields = [
        "sample_id", "speaker_id", "speaker_group", "l1", "task", "prompt_id", "subset",
        "expected_language", "condition", "channel_pair_id", "source_recording_id", "context_type", "context_id",
        "speech_duration_sec", "candidate", "candidate_construct", "evidence_value", "fallback_numeric_value",
        "available", "failure_reason", "evidence_direction", "model_id", "model_version", "batch_status",
        "product_score_available", "language_gate_eligible", "language_gate_reason", "recording_quality_score",
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
        "speech_duration_available_sample_count": duration_available_samples,
        "speech_duration_availability_rate": (
            round(duration_available_samples / len(latest), 6) if latest else None
        ),
        "gold_transcript_used_for_duration_bucket": False,
        "note": "candidate values and availability are unchanged from v5; v10 adds only product-condition speech duration",
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
