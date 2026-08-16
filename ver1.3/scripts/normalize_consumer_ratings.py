#!/usr/bin/env python3
"""Convert validated consumer listener ratings from wide to long form.

The generic evidence analyzer consumes ``sample_id, criterion, human_rating``.
The listener collection schema intentionally stores four constructs separately.
This adapter preserves that construct separation and never averages them.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from validate_consumer_ratings import CONSTRUCT_TO_FIELD, validate_rating_file


SCHEMA_VERSION = "consumer_rating_long_v1"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _assigned(value: Any) -> set[str]:
    text = _text(value)
    return {item.strip() for item in text.replace(",", "|").split("|") if item.strip()}


def normalize_file(input_path: str | Path, output_path: str | Path) -> Dict[str, Any]:
    validation = validate_rating_file(input_path)
    if not validation.get("ok"):
        raise ValueError("consumer rating file failed validation: " + ";".join(validation.get("errors") or []))

    output_rows: list[Dict[str, Any]] = []
    with Path(input_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if _text(row.get("analyzable_yes_no")).lower() != "yes":
                continue
            assigned = _assigned(row.get("assigned_constructs"))
            for construct, field in CONSTRUCT_TO_FIELD.items():
                if construct not in assigned:
                    continue
                rating_text = _text(row.get(field))
                if not rating_text:
                    continue
                output_rows.append(
                    {
                        "sample_id": _text(row.get("sample_id")),
                        "criterion": construct,
                        "human_rating": int(float(rating_text)),
                        "rater_id": _text(row.get("rater_id")),
                        "presentation_id": _text(row.get("presentation_id")),
                        "task": _text(row.get("task_mode")),
                        "intonation_context_available": _text(row.get("intonation_context_available")).lower(),
                        "normalization_schema": SCHEMA_VERSION,
                    }
                )

    fields = [
        "sample_id",
        "criterion",
        "human_rating",
        "rater_id",
        "presentation_id",
        "task",
        "intonation_context_available",
        "normalization_schema",
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    return {
        "schema": SCHEMA_VERSION,
        "input": str(input_path),
        "output": str(output),
        "input_row_count": validation.get("row_count"),
        "normalized_rating_count": len(output_rows),
        "criteria": sorted({row["criterion"] for row in output_rows}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ratings_csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = normalize_file(args.ratings_csv, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
