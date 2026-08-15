#!/usr/bin/env python3
"""Build deterministic speaker/target-disjoint folds for research manifests.

This script freezes split mechanics before external benchmark results are seen.
It does not fit a model and does not choose the number of folds from outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:  # package-style import in tests / `python -m`
    from scripts.validate_research_manifest import load_manifest, validate_manifest_rows
except ImportError:  # direct `python scripts/build_research_splits.py`
    from validate_research_manifest import load_manifest, validate_manifest_rows


SPLIT_SCHEMA = "research_group_split_v1"


def _stable_tiebreak(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def assign_group_folds(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    fold_count: int = 5,
    seed: int = 33017,
) -> Dict[str, int]:
    """Assign whole groups to folds with deterministic greedy row balancing."""

    folds = int(fold_count)
    if folds < 2:
        raise ValueError("fold_count must be at least 2")
    counts = Counter(str(row.get(group_key) or "").strip() for row in rows)
    counts.pop("", None)
    if len(counts) < folds:
        raise ValueError(
            f"{group_key} has only {len(counts)} unique groups, fewer than fold_count={folds}"
        )
    ordered = sorted(
        counts,
        key=lambda group: (-counts[group], _stable_tiebreak(group, seed)),
    )
    loads = [0 for _ in range(folds)]
    group_counts = [0 for _ in range(folds)]
    assignment: Dict[str, int] = {}
    for group in ordered:
        fold = min(
            range(folds),
            key=lambda index: (loads[index], group_counts[index], index),
        )
        assignment[group] = fold
        loads[fold] += counts[group]
        group_counts[fold] += 1
    return assignment


def split_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold_count: int = 5,
    seed: int = 33017,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    speaker_assignment = assign_group_folds(
        rows,
        group_key="speaker_id",
        fold_count=fold_count,
        seed=seed,
    )
    target_assignment = assign_group_folds(
        rows,
        group_key="target_id",
        fold_count=fold_count,
        seed=seed + 1,
    )
    output: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        speaker_id = str(item.get("speaker_id") or "").strip()
        target_id = str(item.get("target_id") or "").strip()
        item["speaker_fold"] = speaker_assignment[speaker_id]
        item["target_fold"] = target_assignment[target_id]
        output.append(item)

    speaker_row_counts = Counter(int(row["speaker_fold"]) for row in output)
    target_row_counts = Counter(int(row["target_fold"]) for row in output)
    report = {
        "schema": SPLIT_SCHEMA,
        "fold_count": int(fold_count),
        "seed": int(seed),
        "row_count": len(output),
        "speaker_group_count": len(speaker_assignment),
        "target_group_count": len(target_assignment),
        "speaker_fold_row_counts": {str(k): speaker_row_counts[k] for k in range(fold_count)},
        "target_fold_row_counts": {str(k): target_row_counts[k] for k in range(fold_count)},
        "speaker_leakage": False,
        "target_leakage": False,
        "note": (
            "speaker_fold and target_fold define two separate evaluation views. "
            "They must not be misread as a single jointly speaker-and-target-disjoint split."
        ),
    }
    return output, report


def verify_group_disjointness(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    fold_key: str,
) -> Dict[str, Any]:
    seen: Dict[str, set[int]] = defaultdict(set)
    for row in rows:
        group = str(row.get(group_key) or "").strip()
        if not group:
            continue
        seen[group].add(int(row[fold_key]))
    leaking = sorted(group for group, folds in seen.items() if len(folds) > 1)
    return {
        "ok": not leaking,
        "group_key": group_key,
        "fold_key": fold_key,
        "leaking_groups": leaking,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty split manifest")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=33017)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows, fields = load_manifest(manifest_path)
    validation = validate_manifest_rows(rows, fields, manifest_dir=manifest_path.parent)
    if not validation["ok"]:
        print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    split, report = split_rows(rows, fold_count=args.folds, seed=args.seed)
    speaker_check = verify_group_disjointness(
        split,
        group_key="speaker_id",
        fold_key="speaker_fold",
    )
    target_check = verify_group_disjointness(
        split,
        group_key="target_id",
        fold_key="target_fold",
    )
    report["speaker_check"] = speaker_check
    report["target_check"] = target_check
    if not speaker_check["ok"] or not target_check["ok"]:
        raise RuntimeError("split leakage detected")

    output = Path(args.out)
    _write_csv(output, split)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
