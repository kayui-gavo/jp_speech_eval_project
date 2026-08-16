#!/usr/bin/env python3
"""Validate consumer criterion v3 listener ratings.

The validator keeps the public four dimensions intact while treating isolated
intonation naturalness and contextual intonation appropriateness as separate
human validation constructs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


SCHEMA_VERSION = "consumer_four_dimension_validation_v3"
CONSTRUCT_TO_FIELD = {
    "clarity_comprehensibility": "clarity_comprehensibility_1to7",
    "fluency": "fluency_1to7",
    "rhythm_naturalness": "rhythm_naturalness_1to7",
    "intonation_utterance_naturalness": "intonation_utterance_naturalness_1to7",
    "intonation_contextual_appropriateness": "intonation_contextual_appropriateness_1to7",
}
ISOLATED = {
    "clarity_comprehensibility",
    "fluency",
    "rhythm_naturalness",
    "intonation_utterance_naturalness",
}
CONTEXTUAL = {"intonation_contextual_appropriateness"}
REQUIRED_COLUMNS = (
    "rater_id",
    "sample_id",
    "presentation_id",
    "presentation_variant",
    "task_mode",
    "audio_asset_id",
    "context_type",
    "context_id",
    "context_text",
    "context_audio_asset_id",
    "assigned_constructs",
    "analyzable_yes_no",
    *CONSTRUCT_TO_FIELD.values(),
    "context_presented_yes_no",
    "timestamp",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _assigned(value: Any) -> set[str]:
    text = _text(value)
    return {item.strip() for item in text.replace(",", "|").split("|") if item.strip()}


def _rating(value: Any) -> int | None:
    text = _text(value)
    if not text:
        return None
    number = float(text)
    if not number.is_integer():
        raise ValueError("rating must be integer-like")
    integer = int(number)
    if not 1 <= integer <= 7:
        raise ValueError("rating must be 1..7")
    return integer


def validate_rating_rows(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Dict[str, Any]:
    missing = [name for name in REQUIRED_COLUMNS if name not in set(fieldnames)]
    if missing:
        return {
            "schema": SCHEMA_VERSION,
            "ok": False,
            "row_count": len(rows),
            "errors": ["missing_required_columns:" + ",".join(missing)],
            "warnings": [],
        }

    errors: list[str] = []
    warnings: list[str] = []
    construct_counts = Counter()
    variant_counts = Counter()
    seen: set[tuple[str, str]] = set()

    for index, row in enumerate(rows, start=2):
        prefix = f"row_{index}"
        rater_id = _text(row.get("rater_id"))
        sample_id = _text(row.get("sample_id"))
        presentation_id = _text(row.get("presentation_id"))
        variant = _text(row.get("presentation_variant"))
        assigned = _assigned(row.get("assigned_constructs"))
        analyzable = _text(row.get("analyzable_yes_no")).lower()
        context_presented = _text(row.get("context_presented_yes_no")).lower()
        context_type = _text(row.get("context_type"))
        has_context_payload = bool(_text(row.get("context_text")) or _text(row.get("context_audio_asset_id")))

        for name, value in (("rater_id", rater_id), ("sample_id", sample_id), ("presentation_id", presentation_id)):
            if not value:
                errors.append(f"{prefix}:empty_{name}")
        key = (rater_id, presentation_id)
        if all(key):
            if key in seen:
                errors.append(f"{prefix}:duplicate_rater_presentation")
            seen.add(key)

        if variant not in {"isolated", "contextual"}:
            errors.append(f"{prefix}:invalid_presentation_variant:{variant}")
        else:
            variant_counts[variant] += 1
        if analyzable not in {"yes", "no"}:
            errors.append(f"{prefix}:invalid_analyzable:{analyzable}")
        if context_presented not in {"yes", "no"}:
            errors.append(f"{prefix}:invalid_context_presented:{context_presented}")

        unknown = assigned - set(CONSTRUCT_TO_FIELD)
        if unknown:
            errors.append(f"{prefix}:unknown_constructs:{'|'.join(sorted(unknown))}")
        if not assigned:
            errors.append(f"{prefix}:no_assigned_constructs")

        if variant == "isolated":
            if assigned - ISOLATED:
                errors.append(f"{prefix}:contextual_construct_on_isolated_presentation")
            if context_presented == "yes" or context_type != "none" or has_context_payload:
                errors.append(f"{prefix}:isolated_presentation_must_not_show_context")
        elif variant == "contextual":
            if assigned - CONTEXTUAL:
                errors.append(f"{prefix}:non_contextual_construct_on_contextual_presentation")
            if context_presented != "yes":
                errors.append(f"{prefix}:contextual_presentation_requires_context_presented")
            if context_type == "none" or not has_context_payload:
                errors.append(f"{prefix}:contextual_presentation_requires_actual_context_payload")

        ratings: Dict[str, int | None] = {}
        for construct, field in CONSTRUCT_TO_FIELD.items():
            try:
                ratings[construct] = _rating(row.get(field))
            except (TypeError, ValueError) as exc:
                errors.append(f"{prefix}:invalid_{field}:{exc}")
                ratings[construct] = None

        if analyzable == "no":
            for construct, rating in ratings.items():
                if rating is not None:
                    errors.append(f"{prefix}:unanalyzable_must_have_null:{construct}")
        elif analyzable == "yes":
            for construct, rating in ratings.items():
                if construct in assigned:
                    construct_counts[construct] += 1
                    if rating is None:
                        errors.append(f"{prefix}:assigned_construct_missing_rating:{construct}")
                elif rating is not None:
                    errors.append(f"{prefix}:unassigned_construct_must_be_null:{construct}")

        if not _text(row.get("timestamp")):
            warnings.append(f"{prefix}:missing_timestamp")

    return {
        "schema": SCHEMA_VERSION,
        "ok": not errors,
        "row_count": len(rows),
        "construct_rating_counts": dict(sorted(construct_counts.items())),
        "presentation_variant_counts": dict(sorted(variant_counts.items())),
        "errors": errors,
        "warnings": warnings,
    }


def validate_rating_file(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return validate_rating_rows(rows, list(reader.fieldnames or []))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ratings_csv")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    report = validate_rating_file(args.ratings_csv)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
