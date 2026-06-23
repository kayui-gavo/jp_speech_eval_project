#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.pitch_naturalness_v2 import FEATURES, predict_continuous_naturalness  # noqa: E402


SOURCE_FEATURES = {
    "f0_coverage": "f0_coverage",
    "utterance_f0_range_log": "f0_range_log",
    "local_pitch_movement": "local_pitch_movement",
    "transition_smoothness": "smoothness",
    "flatness_penalty": "flatness_penalty",
    "instability_penalty": "instability_penalty",
}
ANCHORS = {"native_normal": 95.0, "flat": 20.0, "shuffled": 45.0}
RIDGE_ALPHA = 50.0


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _usable(row: Mapping[str, Any]) -> bool:
    try:
        return all(row.get(source) not in (None, "") and np.isfinite(float(row[source])) for source in SOURCE_FEATURES.values())
    except (TypeError, ValueError):
        return False


def _matrix(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(row[SOURCE_FEATURES[name]]) for name in FEATURES] for row in rows], dtype=float)


def _details(row: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(row[source]) for name, source in SOURCE_FEATURES.items()}


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None, "floor_rate": None, "ceiling_rate": None}
    data = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": round(float(np.mean(data)), 4),
        "p10": round(float(np.percentile(data, 10)), 4),
        "p50": round(float(np.percentile(data, 50)), 4),
        "p90": round(float(np.percentile(data, 90)), 4),
        "floor_rate": round(float(np.mean(data <= 10.0)), 4),
        "ceiling_rate": round(float(np.mean(data >= 95.0)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a continuous, non-saturating weak pitch-naturalness v2 candidate.")
    parser.add_argument("--source-rows", type=Path, default=ROOT / "results/calibration/pitch_naturalness_calibration_rows.csv")
    parser.add_argument("--out-config", type=Path, default=ROOT / "configs/pitch_naturalness_v2.json")
    parser.add_argument("--out-summary", type=Path, default=ROOT / "data/calibration_candidates/pitch_naturalness_v2_summary.csv")
    args = parser.parse_args()
    if not args.source_rows.exists():
        raise FileNotFoundError(
            f"Missing {args.source_rows}. Run scripts/train_pitch_naturalness_calibrator.py first to build the local feature rows."
        )
    rows = _read(args.source_rows)
    train = [row for row in rows if row["split"] == "train" and row["condition"] in ANCHORS and _usable(row)]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])
    model.fit(_matrix(train), np.asarray([ANCHORS[row["condition"]] for row in train], dtype=float))
    scaler: StandardScaler = model.named_steps["scale"]
    ridge: Ridge = model.named_steps["ridge"]
    config = {
        "schema_version": 2,
        "active": False,
        "status": "validated_candidate_not_yet_runtime_default",
        "model_type": "standard_scaler_plus_ridge_regression",
        "score_type": "weak_reference_pitch_naturalness_v2",
        "training_target": "broad_native_likeness_pseudo_anchors",
        "not_a_target": "teacher_grade_pitch_accent_correctness",
        "features": list(FEATURES),
        "pseudo_score_anchors": ANCHORS,
        "ridge_alpha": RIDGE_ALPHA,
        "scaler_mean": [round(float(value), 10) for value in scaler.mean_],
        "scaler_scale": [round(float(value), 10) for value in scaler.scale_],
        "coefficients": [round(float(value), 10) for value in ridge.coef_],
        "intercept": round(float(ridge.intercept_), 10),
        "output_floor": 10.0,
        "output_ceiling": 98.0,
        "automatic_accent_hint_weight": 0.08,
        "verified_accent_hint_weight": 0.20,
        "training_split": "JVS jvs001-jvs030 nonpara30; actual 28 speakers with complete local assets",
        "development_split": "JVS jvs031-jvs040 nonpara30",
        "locked_evaluation_policy": "speaker-disjoint JVS plus external JANON; never fit on JANON",
    }
    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    args.out_config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if not _usable(row):
            continue
        score = predict_continuous_naturalness(_details(row), config)
        if score is not None:
            grouped[(row["split"], row["condition"])].append(score)
    summary = [
        {"split": split, "condition": condition, **_distribution(values)}
        for (split, condition), values in sorted(grouped.items())
    ]
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.out_summary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"train_rows": len(train), "config": str(args.out_config), "summary": str(args.out_summary)}, indent=2))


if __name__ == "__main__":
    main()
