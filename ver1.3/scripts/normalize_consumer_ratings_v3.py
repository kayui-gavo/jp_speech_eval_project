#!/usr/bin/env python3
"""Normalize consumer criterion v3 ratings and reattach analysis-only metadata.

Listener collection remains blinded.  Speaker/channel/split metadata is joined
only after rating collection from the private sample manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

from validate_consumer_ratings_v3 import CONSTRUCT_TO_FIELD, validate_rating_file
from validate_free_speech_sample_manifest import validate_manifest_file


SCHEMA_VERSION = "consumer_rating_long_v2"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _assigned(value: Any) -> set[str]:
    text = _text(value)
    return {item.strip() for item in text.replace(",", "|").split("|") if item.strip()}


def _read_csv(path: str | Path) -> list[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def normalize_file(
    ratings_csv: str | Path,
    manifest_csv: str | Path,
    output_path: str | Path,
) -> Dict[str, Any]:
    rating_validation = validate_rating_file(ratings_csv)
    if not rating_validation.get("ok"):
        raise ValueError("consumer rating file failed validation: " + ";".join(rating_validation.get("errors") or []))
    manifest_validation = validate_manifest_file(manifest_csv)
    if not manifest_validation.get("ok"):
        raise ValueError("sample manifest failed validation: " + ";".join(manifest_validation.get("errors") or []))

    manifest_rows = _read_csv(manifest_csv)
    manifest = {_text(row.get("sample_id")): row for row in manifest_rows}
    output_rows: list[Dict[str, Any]] = []

    for row in _read_csv(ratings_csv):
        if _text(row.get("analyzable_yes_no")).lower() != "yes":
            continue
        sample_id = _text(row.get("sample_id"))
        meta = manifest.get(sample_id)
        if meta is None:
            raise ValueError(f"rating references sample missing from manifest: {sample_id}")
        assigned = _assigned(row.get("assigned_constructs"))
        for construct, field in CONSTRUCT_TO_FIELD.items():
            if construct not in assigned:
                continue
            rating_text = _text(row.get(field))
            if not rating_text:
                continue
            output_rows.append(
                {
                    "sample_id": sample_id,
                    "criterion": construct,
                    "human_rating": int(float(rating_text)),
                    "rater_id": _text(row.get("rater_id")),
                    "presentation_id": _text(row.get("presentation_id")),
                    "presentation_variant": _text(row.get("presentation_variant")),
                    "task": _text(meta.get("task_mode")),
                    "speaker_id": _text(meta.get("speaker_id")),
                    "speaker_group": _text(meta.get("speaker_group")),
                    "l1": _text(meta.get("l1")),
                    "prompt_id": _text(meta.get("prompt_id")),
                    "subset": _text(meta.get("split")),
                    "expected_language": _text(meta.get("expected_language")),
                    "condition": _text(meta.get("channel_condition")),
                    "channel_pair_id": _text(meta.get("channel_pair_id")),
                    "source_recording_id": _text(meta.get("source_recording_id")),
                    "context_type": _text(meta.get("context_type")),
                    "context_id": _text(meta.get("context_id")),
                    "normalization_schema": SCHEMA_VERSION,
                }
            )

    fields = [
        "sample_id", "criterion", "human_rating", "rater_id", "presentation_id",
        "presentation_variant", "task", "speaker_id", "speaker_group", "l1",
        "prompt_id", "subset", "expected_language", "condition", "channel_pair_id",
        "source_recording_id", "context_type", "context_id", "normalization_schema",
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    return {
        "schema": SCHEMA_VERSION,
        "input_rating_rows": rating_validation.get("row_count"),
        "normalized_rating_count": len(output_rows),
        "criteria": sorted({row["criterion"] for row in output_rows}),
        "speaker_count": len({row["speaker_id"] for row in output_rows if row["speaker_id"]}),
        "channel_pair_count": len({row["channel_pair_id"] for row in output_rows if row["channel_pair_id"]}),
        "source_recording_count": len({row["source_recording_id"] for row in output_rows if row["source_recording_id"]}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ratings_csv")
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = normalize_file(args.ratings_csv, args.sample_manifest, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
