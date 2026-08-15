#!/usr/bin/env python3
"""Analyze construct-matched machine evidence against human ratings.

This script is intentionally model-agnostic. Heavy/local workers export one
machine-evidence row per sample/candidate, while listener manifests may contain
several rater rows per sample/criterion. The analyzer aggregates human ratings
first, then reports correlations and availability without fitting a ProductScore
mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr


ANALYSIS_SCHEMA = "research_evidence_analysis_v1"
REQUIRED_HUMAN = {"sample_id", "criterion", "human_rating"}
REQUIRED_EVIDENCE = {"sample_id", "candidate", "evidence_value"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_csv(path: str | Path) -> tuple[List[Dict[str, str]], List[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def aggregate_human_ratings(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[float]] = defaultdict(list)
    metadata: Dict[tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        sample_id = _text(row.get("sample_id"))
        criterion = _text(row.get("criterion"))
        rating = _finite(row.get("human_rating"))
        if not sample_id or not criterion or rating is None:
            continue
        key = (sample_id, criterion)
        grouped[key].append(rating)
        meta = metadata.setdefault(key, {})
        for field in ("speaker_id", "target_id", "task", "subset", "l1", "condition", "speaker_fold", "target_fold"):
            value = _text(row.get(field))
            if not value:
                continue
            previous = meta.get(field)
            if previous is not None and previous != value:
                raise ValueError(f"conflicting {field} for sample={sample_id}, criterion={criterion}")
            meta[field] = value
    output: List[Dict[str, Any]] = []
    for (sample_id, criterion), values in sorted(grouped.items()):
        output.append({
            "sample_id": sample_id,
            "criterion": criterion,
            "human_rating_mean": float(np.mean(values)),
            "human_rating_median": float(np.median(values)),
            "human_rating_count": len(values),
            **metadata[(sample_id, criterion)],
        })
    return output


def _evidence_index(rows: Sequence[Mapping[str, Any]]) -> Dict[tuple[str, str], Dict[str, Any]]:
    index: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        sample_id = _text(row.get("sample_id"))
        candidate = _text(row.get("candidate"))
        if not sample_id or not candidate:
            continue
        key = (sample_id, candidate)
        if key in index:
            raise ValueError(f"duplicate evidence row for sample={sample_id}, candidate={candidate}")
        index[key] = dict(row)
    return index


def join_human_and_evidence(
    human_aggregated: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    evidence = _evidence_index(evidence_rows)
    candidates = sorted({key[1] for key in evidence})
    joined: List[Dict[str, Any]] = []
    for human in human_aggregated:
        sample_id = _text(human.get("sample_id"))
        for candidate in candidates:
            row = evidence.get((sample_id, candidate))
            if row is None:
                joined.append({
                    **dict(human),
                    "candidate": candidate,
                    "evidence_available": False,
                    "evidence_value": None,
                    "failure_reason": "missing_evidence_row",
                })
                continue
            value = _finite(row.get("evidence_value"))
            available_text = _text(row.get("available")).lower()
            explicitly_unavailable = available_text in {"false", "0", "no"}
            available = value is not None and not explicitly_unavailable
            joined.append({
                **dict(human),
                "candidate": candidate,
                "evidence_available": available,
                "evidence_value": value if available else None,
                "failure_reason": _text(row.get("failure_reason")) if not available else "",
                "model_id": _text(row.get("model_id")),
                "model_version": _text(row.get("model_version")),
                "evidence_direction": _text(row.get("evidence_direction")) or "unspecified",
            })
    return joined


def _spearman(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    usable = [
        row for row in rows
        if bool(row.get("evidence_available"))
        and _finite(row.get("evidence_value")) is not None
        and _finite(row.get("human_rating_mean")) is not None
    ]
    n = len(usable)
    if n < 3:
        return {"n": n, "rho": None, "pvalue": None, "reason": "fewer_than_3_usable_pairs"}
    x = np.asarray([float(row["evidence_value"]) for row in usable], dtype=float)
    y = np.asarray([float(row["human_rating_mean"]) for row in usable], dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"n": n, "rho": None, "pvalue": None, "reason": "constant_input"}
    result = spearmanr(x, y)
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "n": n,
        "rho": rho if math.isfinite(rho) else None,
        "pvalue": pvalue if math.isfinite(pvalue) else None,
        "reason": "ok",
    }


def _group_correlations(rows: Sequence[Mapping[str, Any]], group_key: str) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        group = _text(row.get(group_key))
        if group:
            grouped[group].append(row)
    output: Dict[str, Any] = {}
    for group, group_rows in sorted(grouped.items()):
        output[group] = _spearman(group_rows)
    return output


def analyze_joined_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    candidates = sorted({_text(row.get("candidate")) for row in rows if _text(row.get("candidate"))})
    criteria = sorted({_text(row.get("criterion")) for row in rows if _text(row.get("criterion"))})
    results: Dict[str, Any] = {}
    for criterion in criteria:
        criterion_rows = [row for row in rows if _text(row.get("criterion")) == criterion]
        results[criterion] = {}
        for candidate in candidates:
            subset = [row for row in criterion_rows if _text(row.get("candidate")) == candidate]
            total = len(subset)
            available = sum(bool(row.get("evidence_available")) for row in subset)
            failure_counts: Dict[str, int] = defaultdict(int)
            for row in subset:
                if not row.get("evidence_available"):
                    failure_counts[_text(row.get("failure_reason")) or "unspecified"] += 1
            results[criterion][candidate] = {
                "row_count": total,
                "availability_rate": (available / total) if total else None,
                "available_count": available,
                "failure_reasons": dict(sorted(failure_counts.items())),
                "overall_spearman": _spearman(subset),
                "within_target": _group_correlations(subset, "target_id"),
                "within_speaker": _group_correlations(subset, "speaker_id"),
                "speaker_fold": _group_correlations(subset, "speaker_fold"),
                "target_fold": _group_correlations(subset, "target_fold"),
            }
    return {
        "schema": ANALYSIS_SCHEMA,
        "criteria": criteria,
        "candidates": candidates,
        "results": results,
        "interpretation": (
            "raw construct-matched association/availability report; no model selection or /100 mapping"
        ),
    }


def analyze_files(human_path: str | Path, evidence_path: str | Path) -> Dict[str, Any]:
    human_rows, human_fields = _read_csv(human_path)
    evidence_rows, evidence_fields = _read_csv(evidence_path)
    missing_human = sorted(REQUIRED_HUMAN - set(human_fields))
    missing_evidence = sorted(REQUIRED_EVIDENCE - set(evidence_fields))
    if missing_human:
        raise ValueError("human file missing columns: " + ",".join(missing_human))
    if missing_evidence:
        raise ValueError("evidence file missing columns: " + ",".join(missing_evidence))
    aggregated = aggregate_human_ratings(human_rows)
    joined = join_human_and_evidence(aggregated, evidence_rows)
    report = analyze_joined_rows(joined)
    report["human_rating_rows"] = len(human_rows)
    report["human_sample_criterion_count"] = len(aggregated)
    report["evidence_rows"] = len(evidence_rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", required=True, help="normalized human-rating CSV")
    parser.add_argument("--evidence", required=True, help="machine evidence CSV")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    report = analyze_files(args.human, args.evidence)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
