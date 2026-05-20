#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


FEATURE_DIRECTIONS = {
    "total_score": "higher",
    "pronunciation_score": "higher",
    "prosody_score": "higher",
    "fluency_score": "higher",
    "tone_score": "higher",
    "reliability_overall": "higher",
    "contour_corr": "higher",
    "transition_agreement": "higher",
    "accent_drop_agreement": "higher",
    "mora_duration_cv": "lower",
    "boundary_cv": "lower",
    "special_mora_penalty": "lower",
}


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _as_float(row: Dict[str, str], key: str) -> Optional[float]:
    try:
        value = float(row.get(key, ""))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _cohens_d(native: List[float], learner: List[float]) -> Optional[float]:
    if len(native) < 2 or len(learner) < 2:
        return None
    var_native = statistics.variance(native)
    var_learner = statistics.variance(learner)
    pooled = ((len(native) - 1) * var_native + (len(learner) - 1) * var_learner) / (len(native) + len(learner) - 2)
    if pooled <= 0:
        return None
    return (statistics.mean(native) - statistics.mean(learner)) / math.sqrt(pooled)


def _auc(native: List[float], learner: List[float], direction: str) -> Optional[float]:
    if not native or not learner:
        return None
    wins = 0.0
    total = 0
    for n in native:
        for l in learner:
            if direction == "higher":
                wins += 1.0 if n > l else 0.5 if n == l else 0.0
            else:
                wins += 1.0 if n < l else 0.5 if n == l else 0.0
            total += 1
    return wins / total if total else None


def _paired_stimulus_gap(rows: Iterable[Dict[str, str]], feature: str, direction: str) -> Optional[float]:
    by_stimulus: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"native": [], "learner": []})
    for row in rows:
        value = _as_float(row, feature)
        if value is None:
            continue
        group = "native" if row.get("native_language") == "Japanese" else "learner"
        by_stimulus[row.get("stimulus") or ""]["native" if group == "native" else "learner"].append(value)
    gaps: List[float] = []
    for values in by_stimulus.values():
        if not values["native"] or not values["learner"]:
            continue
        native_mean = statistics.mean(values["native"])
        learner_mean = statistics.mean(values["learner"])
        gap = native_mean - learner_mean if direction == "higher" else learner_mean - native_mean
        gaps.append(gap)
    return statistics.mean(gaps) if gaps else None


def build_report(rows: List[Dict[str, str]]) -> Dict[str, object]:
    native_rows = [row for row in rows if row.get("native_language") == "Japanese"]
    learner_rows = [row for row in rows if row.get("native_language") and row.get("native_language") != "Japanese"]
    features: List[Dict[str, object]] = []
    for feature, direction in FEATURE_DIRECTIONS.items():
        native = [value for row in native_rows if (value := _as_float(row, feature)) is not None]
        learner = [value for row in learner_rows if (value := _as_float(row, feature)) is not None]
        if not native or not learner:
            continue
        features.append({
            "feature": feature,
            "direction": direction,
            "n_native": len(native),
            "n_learner": len(learner),
            "native_mean": round(statistics.mean(native), 6),
            "learner_mean": round(statistics.mean(learner), 6),
            "cohens_d_native_minus_learner": None if (d := _cohens_d(native, learner)) is None else round(d, 6),
            "native_better_auc": None if (auc := _auc(native, learner, direction)) is None else round(auc, 6),
            "mean_paired_stimulus_gap": None if (gap := _paired_stimulus_gap(rows, feature, direction)) is None else round(gap, 6),
        })
    features.sort(
        key=lambda row: (
            -(row["native_better_auc"] or 0.0),
            -abs(row["cohens_d_native_minus_learner"] or 0.0),
        )
    )
    return {
        "summary": {
            "n_rows": len(rows),
            "n_native": len(native_rows),
            "n_learner": len(learner_rows),
            "interpretation": "diagnostic report only; use teacher labels before turning these into scoring weights",
        },
        "features": features,
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize native-vs-learner separability from JANON analysis CSV.")
    parser.add_argument("--csv", required=True, help="CSV produced by scripts/analyze_janon.py")
    parser.add_argument("--out-json", default="outputs/janon_calibration_report.json")
    parser.add_argument("--out-csv", default="outputs/janon_calibration_report.csv")
    args = parser.parse_args()

    rows = _read_rows(Path(args.csv))
    report = build_report(rows)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    if not out_json.is_absolute():
        out_json = Path(__file__).resolve().parents[1] / out_json
    if not out_csv.is_absolute():
        out_csv = Path(__file__).resolve().parents[1] / out_csv
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_csv, list(report["features"]))
    print(json.dumps({
        "out_json": str(out_json),
        "out_csv": str(out_csv),
        "n_features": len(report["features"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
