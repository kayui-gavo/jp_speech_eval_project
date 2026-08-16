#!/usr/bin/env python3
"""Compare current ProductScore with the shadow partial-evidence aggregate.

This script is descriptive only. It never tunes thresholds, changes ProductScore,
or promotes the candidate. It can recompute the candidate from stored
``user_score.component_scores`` so old validation JSONL remains useful even if
it predates the ``score_candidates`` field.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from jp_speech_eval.partial_evidence_aggregate import build_partial_evidence_aggregate_candidate


ANALYSIS_SCHEMA = "partial_evidence_aggregate_analysis_v1"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _summary(values: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if not arr.size:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "p10": round(float(np.quantile(arr, 0.10)), 4),
        "p25": round(float(np.quantile(arr, 0.25)), 4),
        "median": round(float(np.median(arr)), 4),
        "p75": round(float(np.quantile(arr, 0.75)), 4),
        "p90": round(float(np.quantile(arr, 0.90)), 4),
        "max": round(float(np.max(arr)), 4),
        "iqr": round(float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)), 4),
    }


def _latest_attempts(path: str | Path) -> Dict[str, Mapping[str, Any]]:
    latest: Dict[str, Mapping[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "").strip()
            if sample_id:
                latest[sample_id] = row
    return latest


def analyze_batch(path: str | Path) -> Dict[str, Any]:
    latest = _latest_attempts(path)
    rows = []
    superseded = 0
    # Count raw attempts separately without retaining them in analysis.
    raw_attempts = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw_attempts += 1
    superseded = max(0, raw_attempts - len(latest))

    for sample_id, row in latest.items():
        if str(row.get("status") or "") != "ok":
            continue
        user_score = row.get("user_score") if isinstance(row.get("user_score"), Mapping) else {}
        components = (
            user_score.get("component_scores")
            if isinstance(user_score.get("component_scores"), Mapping)
            else {}
        )
        candidate = build_partial_evidence_aggregate_candidate(components)
        current = _finite(user_score.get("display_score"))
        candidate_score = _finite(candidate.get("candidate_display_score"))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        rows.append(
            {
                "sample_id": sample_id,
                "task_mode": str(metadata.get("task_mode") or "unknown"),
                "speaker_group": str(metadata.get("speaker_group") or "unknown"),
                "channel_condition": str(metadata.get("channel_condition") or "unknown"),
                "current_display_score": current,
                "candidate_display_score": candidate_score,
                "candidate_available": bool(candidate.get("available")),
                "effective_coverage": _finite(candidate.get("effective_coverage")),
                "neutral_prior_count": int(candidate.get("neutral_prior_count") or 0),
                "available_component_count": int(candidate.get("available_component_count") or 0),
            }
        )

    paired = [
        row for row in rows
        if row["current_display_score"] is not None and row["candidate_display_score"] is not None
    ]
    current = [row["current_display_score"] for row in paired]
    candidate = [row["candidate_display_score"] for row in paired]
    deltas = [b - a for a, b in zip(current, candidate)]
    coverage = [
        row["effective_coverage"] for row in rows
        if row["effective_coverage"] is not None
    ]

    def by_field(field: str) -> Dict[str, Any]:
        groups: Dict[str, list[Dict[str, Any]]] = {}
        for row in paired:
            groups.setdefault(str(row[field]), []).append(row)
        return {
            key: {
                "current": _summary(item["current_display_score"] for item in group),
                "candidate": _summary(item["candidate_display_score"] for item in group),
                "delta": _summary(
                    item["candidate_display_score"] - item["current_display_score"]
                    for item in group
                ),
            }
            for key, group in sorted(groups.items())
        }

    return {
        "schema": ANALYSIS_SCHEMA,
        "input": str(path),
        "raw_attempt_count": raw_attempts,
        "latest_sample_count": len(latest),
        "superseded_attempt_count": superseded,
        "ok_latest_count": len(rows),
        "paired_score_count": len(paired),
        "current_display": _summary(current),
        "partial_evidence_candidate": _summary(candidate),
        "candidate_minus_current": _summary(deltas),
        "effective_coverage": _summary(coverage),
        "neutral_prior_count": _summary(float(row["neutral_prior_count"]) for row in rows),
        "available_component_count": _summary(float(row["available_component_count"]) for row in rows),
        "candidate_available_rate": (
            round(sum(bool(row["candidate_available"]) for row in rows) / len(rows), 6)
            if rows else None
        ),
        "by_task_mode": by_field("task_mode"),
        "by_speaker_group": by_field("speaker_group"),
        "by_channel_condition": by_field("channel_condition"),
        "decision": "telemetry_only_no_product_promotion",
        "user_facing": False,
        "product_score_changed": False,
        "interpretation": (
            "Use this report to inspect dispersion and missing-evidence behavior. "
            "It does not establish criterion validity, learner ordering, or channel safety."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = analyze_batch(args.input_jsonl)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
