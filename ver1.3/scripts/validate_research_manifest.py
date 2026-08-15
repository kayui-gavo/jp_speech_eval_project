#!/usr/bin/env python3
"""Validate normalized external-corpus manifests without bundling corpus data.

The validator deliberately knows nothing about a proprietary corpus directory
layout. A local mapping step converts licensed/downloaded corpus metadata into
this normalized schema; all repository benchmarks can then consume the same
contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REQUIRED_COLUMNS = (
    "sample_id",
    "audio_path",
    "speaker_id",
    "target_id",
    "target_text",
    "task",
    "criterion",
    "human_rating",
)
OPTIONAL_COLUMNS = (
    "l1",
    "proficiency",
    "subset",
    "rater_id",
    "condition",
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _finite_rating(value: Any) -> float:
    text = _clean(value)
    if not text:
        raise ValueError("human_rating is empty")
    number = float(text)
    if not math.isfinite(number):
        raise ValueError("human_rating must be finite")
    return number


def load_manifest(path: str | Path) -> tuple[List[Dict[str, str]], List[str]]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(item) for item in (reader.fieldnames or [])]
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def validate_manifest_rows(
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
    *,
    manifest_dir: str | Path = ".",
    check_audio_exists: bool = False,
) -> Dict[str, Any]:
    fields = set(fieldnames)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fields]
    errors: List[str] = []
    warnings: List[str] = []
    if missing_columns:
        errors.append("missing_required_columns:" + ",".join(missing_columns))
        return {
            "ok": False,
            "row_count": len(rows),
            "errors": errors,
            "warnings": warnings,
            "missing_required_columns": missing_columns,
        }

    base = Path(manifest_dir)
    seen_keys: set[tuple[str, str, str]] = set()
    sample_audio: Dict[str, str] = {}
    sample_speaker: Dict[str, str] = {}
    criteria = Counter()
    tasks = Counter()
    subsets = Counter()
    missing_audio_count = 0
    absolute_audio_path_count = 0

    for index, row in enumerate(rows, start=2):
        prefix = f"row_{index}"
        for column in REQUIRED_COLUMNS:
            if not _clean(row.get(column)):
                errors.append(f"{prefix}:empty_{column}")

        sample_id = _clean(row.get("sample_id"))
        audio_path = _clean(row.get("audio_path"))
        speaker_id = _clean(row.get("speaker_id"))
        criterion = _clean(row.get("criterion"))
        rater_id = _clean(row.get("rater_id")) or "__aggregate__"
        task = _clean(row.get("task"))
        subset = _clean(row.get("subset")) or "__unspecified__"

        try:
            _finite_rating(row.get("human_rating"))
        except (TypeError, ValueError) as exc:
            errors.append(f"{prefix}:invalid_human_rating:{exc}")

        key = (sample_id, criterion, rater_id)
        if all(key):
            if key in seen_keys:
                errors.append(
                    f"{prefix}:duplicate_sample_criterion_rater:{sample_id}|{criterion}|{rater_id}"
                )
            seen_keys.add(key)

        if sample_id and audio_path:
            previous = sample_audio.setdefault(sample_id, audio_path)
            if previous != audio_path:
                errors.append(f"{prefix}:sample_audio_path_conflict:{sample_id}")
        if sample_id and speaker_id:
            previous_speaker = sample_speaker.setdefault(sample_id, speaker_id)
            if previous_speaker != speaker_id:
                errors.append(f"{prefix}:sample_speaker_conflict:{sample_id}")

        if criterion:
            criteria[criterion] += 1
        if task:
            tasks[task] += 1
        subsets[subset] += 1

        if audio_path:
            audio = Path(audio_path)
            if audio.is_absolute():
                absolute_audio_path_count += 1
            resolved = audio if audio.is_absolute() else base / audio
            if check_audio_exists and not resolved.is_file():
                missing_audio_count += 1
                errors.append(f"{prefix}:audio_not_found:{audio_path}")

    if absolute_audio_path_count:
        warnings.append(
            f"absolute_audio_paths:{absolute_audio_path_count};relative_paths_are_preferred_for_reproducibility"
        )
    if not rows:
        warnings.append("manifest_has_no_data_rows")

    return {
        "ok": not errors,
        "row_count": len(rows),
        "unique_sample_count": len(sample_audio),
        "unique_speaker_count": len(set(sample_speaker.values())),
        "criteria": dict(sorted(criteria.items())),
        "tasks": dict(sorted(tasks.items())),
        "subsets": dict(sorted(subsets.items())),
        "absolute_audio_path_count": absolute_audio_path_count,
        "missing_audio_count": missing_audio_count,
        "errors": errors,
        "warnings": warnings,
        "required_columns": list(REQUIRED_COLUMNS),
        "optional_columns": list(OPTIONAL_COLUMNS),
    }


def validate_manifest(
    path: str | Path,
    *,
    check_audio_exists: bool = False,
) -> Dict[str, Any]:
    manifest_path = Path(path)
    rows, fieldnames = load_manifest(manifest_path)
    return validate_manifest_rows(
        rows,
        fieldnames,
        manifest_dir=manifest_path.parent,
        check_audio_exists=check_audio_exists,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Normalized research CSV manifest")
    parser.add_argument(
        "--check-audio-exists",
        action="store_true",
        help="Resolve relative audio paths from the manifest directory and require files to exist",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    report = validate_manifest(args.manifest, check_audio_exists=args.check_audio_exists)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
