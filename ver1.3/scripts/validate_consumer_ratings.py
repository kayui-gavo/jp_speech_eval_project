#!/usr/bin/env python3
"""Validate consumer four-construct listener ratings before analysis."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


CONSTRUCT_TO_FIELD = {
    "clarity_comprehensibility": "clarity_comprehensibility_1to7",
    "fluency": "fluency_1to7",
    "rhythm_naturalness": "rhythm_naturalness_1to7",
    "intonation_naturalness": "intonation_naturalness_1to7",
}
REQUIRED_COLUMNS = (
    "rater_id",
    "sample_id",
    "presentation_id",
    "task_mode",
    "assigned_constructs",
    "analyzable_yes_no",
    *CONSTRUCT_TO_FIELD.values(),
    "intonation_context_available",
    "timestamp",
)
ALLOWED_TASK_MODES = {"fixed_reading", "spontaneous", "controlled_dialogue"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _assigned(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        items = [_text(item) for item in value]
    else:
        text = _text(value)
        items = [item.strip() for item in text.replace(",", "|").split("|") if item.strip()]
    return set(items)


def _rating(value: Any) -> int | None:
    text = _text(value)
    if not text:
        return None
    number = int(text)
    if str(number) != text and text not in {f"{number}.0"}:
        # Reject 4.5 etc.; tolerate CSVs that serialized an integer as 4.0.
        float_number = float(text)
        if float_number != number:
            raise ValueError("rating must be an integer")
    if number < 1 or number > 7:
        raise ValueError("rating must be between 1 and 7")
    return number


def _bool_text(value: Any) -> bool | None:
    text = _text(value).lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def validate_rating_rows(
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> Dict[str, Any]:
    fields = set(fieldnames)
    missing = [column for column in REQUIRED_COLUMNS if column not in fields]
    if missing:
        return {
            "ok": False,
            "row_count": len(rows),
            "missing_required_columns": missing,
            "errors": ["missing_required_columns:" + ",".join(missing)],
            "warnings": [],
        }

    errors: List[str] = []
    warnings: List[str] = []
    seen_presentations: set[tuple[str, str]] = set()
    construct_counts = Counter()
    analyzable_counts = Counter()
    task_counts = Counter()

    for index, row in enumerate(rows, start=2):
        prefix = f"row_{index}"
        rater_id = _text(row.get("rater_id"))
        sample_id = _text(row.get("sample_id"))
        presentation_id = _text(row.get("presentation_id"))
        task_mode = _text(row.get("task_mode"))
        analyzable = _text(row.get("analyzable_yes_no")).lower()
        assigned = _assigned(row.get("assigned_constructs"))

        for name, value in (
            ("rater_id", rater_id),
            ("sample_id", sample_id),
            ("presentation_id", presentation_id),
        ):
            if not value:
                errors.append(f"{prefix}:empty_{name}")
        if task_mode not in ALLOWED_TASK_MODES:
            errors.append(f"{prefix}:invalid_task_mode:{task_mode}")
        else:
            task_counts[task_mode] += 1
        if analyzable not in {"yes", "no"}:
            errors.append(f"{prefix}:invalid_analyzable:{analyzable}")
        else:
            analyzable_counts[analyzable] += 1

        unknown = assigned - set(CONSTRUCT_TO_FIELD)
        if unknown:
            errors.append(f"{prefix}:unknown_assigned_constructs:{','.join(sorted(unknown))}")
        if not assigned:
            errors.append(f"{prefix}:no_assigned_constructs")

        key = (rater_id, presentation_id)
        if all(key):
            if key in seen_presentations:
                errors.append(f"{prefix}:duplicate_rater_presentation:{rater_id}|{presentation_id}")
            seen_presentations.add(key)

        context_value = _bool_text(row.get("intonation_context_available"))
        if context_value is None:
            errors.append(f"{prefix}:invalid_intonation_context_available")

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
                    errors.append(f"{prefix}:unanalyzable_must_have_null_{construct}")
        elif analyzable == "yes":
            for construct in CONSTRUCT_TO_FIELD:
                rating = ratings[construct]
                if construct in assigned:
                    construct_counts[construct] += 1
                    if rating is None:
                        errors.append(f"{prefix}:assigned_construct_missing_rating:{construct}")
                elif rating is not None:
                    errors.append(f"{prefix}:unassigned_construct_must_be_null:{construct}")

        if "intonation_naturalness" in assigned and context_value is False:
            warnings.append(f"{prefix}:intonation_rating_without_context;analyze_separately")

    return {
        "ok": not errors,
        "row_count": len(rows),
        "construct_rating_counts": dict(sorted(construct_counts.items())),
        "analyzable_counts": dict(sorted(analyzable_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "errors": errors,
        "warnings": warnings,
    }


def validate_rating_file(path: str | Path) -> Dict[str, Any]:
    rating_path = Path(path)
    with rating_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return validate_rating_rows(rows, fields)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ratings")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    report = validate_rating_file(args.ratings)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
