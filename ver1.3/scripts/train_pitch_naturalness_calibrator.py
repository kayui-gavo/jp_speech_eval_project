#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_cross_speaker_prosody_reference import (  # noqa: E402
    _lab_f0,
    flat_f0,
    shuffled_f0,
    wrong_drop_f0,
)
from audit_pitch_heldout_validation import _read_csv, _transcripts  # noqa: E402
from jp_speech_eval.scoring import score_weak_reference_native_likeness  # noqa: E402
from jp_speech_eval.text_frontend import build_text_info  # noqa: E402


FEATURES = (
    "f0_coverage",
    "f0_range_log",
    "local_pitch_movement",
    "smoothness",
    "flatness_penalty",
    "instability_penalty",
)
TRAIN_SPEAKERS = tuple(f"jvs{index:03d}" for index in range(1, 31))
DEV_SPEAKERS = tuple(f"jvs{index:03d}" for index in range(31, 41))
TEST_SPEAKERS = tuple(f"jvs{index:03d}" for index in range(41, 101))
UTTERANCES_PER_SPEAKER = 10


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _speaker_manifest(jvs_root: Path, speakers: Sequence[str], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for speaker_id in speakers:
        directory = jvs_root / speaker_id / "nonpara30"
        transcript_path = directory / "transcripts_utf8.txt"
        if not transcript_path.exists():
            continue
        transcripts = _transcripts(transcript_path)
        wav_ids = {path.stem for path in (directory / "wav24kHz16bit").glob("*.wav")}
        lab_ids = {path.stem for path in (directory / "lab" / "mon").glob("*.lab")}
        utterance_ids = sorted(set(transcripts) & wav_ids & lab_ids)
        if not utterance_ids:
            continue
        indexes = np.linspace(0, len(utterance_ids) - 1, UTTERANCES_PER_SPEAKER, dtype=int)
        for index in indexes:
            utterance_id = utterance_ids[int(index)]
            rows.append({
                "sample_id": f"{speaker_id}:{utterance_id}",
                "dataset": "JVS",
                "split": split,
                "speaker_id": speaker_id,
                "reference_text": transcripts[utterance_id],
                "audio_path": str((directory / "wav24kHz16bit" / f"{utterance_id}.wav").resolve()),
                "lab_path": str((directory / "lab" / "mon" / f"{utterance_id}.lab").resolve()),
            })
    return rows


def _feature_row(item: Mapping[str, Any], condition: str, f0_values: Sequence[float], *, sample_rate: int) -> dict[str, Any]:
    score, _feedback, details = score_weak_reference_native_likeness(list(f0_values))
    return {
        **dict(item),
        "condition": condition,
        "label_native_like": 1 if condition == "native_normal" else 0 if condition in {"flat", "shuffled"} else "",
        "manual_formula_score": score,
        "available": bool(details.get("available")),
        "f0_coverage": details.get("f0_coverage"),
        "f0_range_log": details.get("utterance_f0_range_log"),
        "local_pitch_movement": details.get("local_pitch_movement"),
        "smoothness": details.get("transition_smoothness"),
        "flatness_penalty": details.get("flatness_penalty"),
        "instability_penalty": details.get("instability_penalty"),
        "sample_rate": sample_rate,
    }


def extract_development_rows(
    jvs_root: Path,
    checkpoint: Path,
    *,
    sample_rate: int,
) -> list[dict[str, Any]]:
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed[(row["sample_id"], row["condition"])] = row
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    manifest = _speaker_manifest(jvs_root, TRAIN_SPEAKERS, "train") + _speaker_manifest(jvs_root, DEV_SPEAKERS, "dev")
    conditions = ("native_normal", "flat", "shuffled", "wrong_drop")
    for index, item in enumerate(manifest, start=1):
        if all((item["sample_id"], condition) in completed for condition in conditions):
            continue
        text_info = build_text_info(str(item["reference_text"]))
        f0_values, _meta = _lab_f0(item, text_info.moras, sample_rate=sample_rate)
        if f0_values is None:
            continue
        variants = {
            "native_normal": f0_values,
            "flat": flat_f0(f0_values),
            "shuffled": shuffled_f0(f0_values, seed=sum(ord(char) for char in str(item["sample_id"]))),
            "wrong_drop": wrong_drop_f0(f0_values, f0_values, text_info.accent_phrases),
        }
        for condition, values in variants.items():
            key = (str(item["sample_id"]), condition)
            if key in completed:
                continue
            row = _feature_row(item, condition, values, sample_rate=sample_rate)
            with checkpoint.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            completed[key] = row
        print(f"[{index}/{len(manifest)}] {item['sample_id']}", flush=True)
    return [
        completed[(str(item["sample_id"]), condition)]
        for item in manifest
        for condition in conditions
        if (str(item["sample_id"]), condition) in completed
    ]


def _has_features(row: Mapping[str, Any]) -> bool:
    try:
        return all(row.get(name) not in (None, "") and np.isfinite(float(row[name])) for name in FEATURES)
    except (TypeError, ValueError):
        return False


def _matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in FEATURES] for row in rows], dtype=float)


def _labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["label_native_like"]) for row in rows], dtype=int)


def _select_threshold(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.05, 0.95, 181)
    scored: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predicted = probabilities >= threshold
        balanced = balanced_accuracy_score(y, predicted)
        native_recall = float(np.mean(predicted[y == 1]))
        scored.append((float(balanced), native_recall, float(threshold)))
    balanced, _native_recall, threshold = max(scored, key=lambda item: (item[0], item[1], -item[2]))
    return threshold, balanced


def _apply_model(model: Pipeline, rows: Sequence[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    usable = [row for row in rows if _has_features(row)]
    if usable:
        probabilities = model.predict_proba(_matrix(usable))[:, 1]
        for row, probability in zip(usable, probabilities):
            row["model_native_probability"] = round(float(probability), 6)
            row["model_candidate_score"] = int(round(100.0 * float(probability)))
            row["model_native_like_at_threshold"] = bool(probability >= threshold)
    for row in rows:
        if "model_native_probability" not in row:
            row["model_native_probability"] = ""
            row["model_candidate_score"] = ""
            row["model_native_like_at_threshold"] = ""
    return list(rows)


def _distribution(rows: Sequence[Mapping[str, Any]], field: str = "model_candidate_score") -> dict[str, Any]:
    values = np.asarray([float(row[field]) for row in rows if row.get(field) not in (None, "")], dtype=float)
    if not values.size:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "n": int(values.size),
        "mean": round(float(np.mean(values)), 4),
        "p10": round(float(np.percentile(values, 10)), 4),
        "p50": round(float(np.percentile(values, 50)), 4),
        "p90": round(float(np.percentile(values, 90)), 4),
    }


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("split")), str(row.get("condition")))].append(row)
    return [{"split": split, "condition": condition, **_distribution(items)} for (split, condition), items in sorted(grouped.items())]


def _paired_win_rate(rows: Sequence[Mapping[str, Any]], condition: str) -> float | None:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("model_candidate_score") not in (None, ""):
            grouped[str(row["sample_id"])][str(row["condition"])] = float(row["model_candidate_score"])
    pairs = [item for item in grouped.values() if "native_normal" in item and condition in item]
    if not pairs:
        return None
    return round(sum(item["native_normal"] > item[condition] for item in pairs) / len(pairs), 4)


def _load_locked_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    out: list[dict[str, Any]] = []
    for source in rows:
        condition = source["condition"]
        row: dict[str, Any] = dict(source)
        if source["group"] == "jvs_native_heldout":
            row["split"] = "locked_test"
            row["label_native_like"] = 1 if condition == "native_normal" else 0 if condition in {"flat", "shuffled"} else ""
        else:
            row["split"] = "external_janon_native" if source["group"] == "janon_native_external" else "external_janon_learner"
            row["label_native_like"] = ""
        row["manual_formula_score"] = source.get("score") or ""
        out.append(row)
    return out


def _model_artifact(
    model: Pipeline,
    threshold: float,
    metrics: Mapping[str, Any],
    *,
    train_speakers: Sequence[str],
    dev_speakers: Sequence[str],
    test_speakers: Sequence[str],
) -> dict[str, Any]:
    scaler: StandardScaler = model.named_steps["scale"]
    classifier: LogisticRegression = model.named_steps["classifier"]
    return {
        "schema_version": 1,
        "model_type": "standard_scaler_plus_logistic_regression",
        "status": "offline_candidate_not_active_in_runtime",
        "target": "jvs_native_normal_vs_paired_flat_or_shuffled_proxy",
        "not_a_target": "teacher_pitch_accent_correctness_or_learner_ability",
        "features": list(FEATURES),
        "scaler_mean": [round(float(value), 10) for value in scaler.mean_],
        "scaler_scale": [round(float(value), 10) for value in scaler.scale_],
        "coefficients": [round(float(value), 10) for value in classifier.coef_[0]],
        "intercept": round(float(classifier.intercept_[0]), 10),
        "decision_threshold": round(float(threshold), 6),
        "speaker_splits": {
            "train": list(train_speakers),
            "dev": list(dev_speakers),
            "locked_test": list(test_speakers),
        },
        "metrics": dict(metrics),
    }


def _report(path: Path, summary: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    lookup = {(row["split"], row["condition"]): row for row in summary}
    def stat(split: str, condition: str, key: str = "mean") -> Any:
        return lookup.get((split, condition), {}).get(key)
    test_rows = [row for row in rows if row.get("split") == "locked_test"]
    train_rows = [row for row in rows if row.get("split") == "train"]
    dev_rows = [row for row in rows if row.get("split") == "dev"]
    janon_rows = [row for row in rows if str(row.get("split", "")).startswith("external_janon")]
    split_counts = {
        "train": (len({str(row["speaker_id"]) for row in train_rows}), sum(row.get("condition") == "native_normal" for row in train_rows)),
        "dev": (len({str(row["speaker_id"]) for row in dev_rows}), sum(row.get("condition") == "native_normal" for row in dev_rows)),
        "locked_test": (len({str(row["speaker_id"]) for row in test_rows}), sum(row.get("condition") == "native_normal" for row in test_rows)),
        "janon": (len({str(row["speaker_id"]) for row in janon_rows}), len(janon_rows)),
    }
    manual = {
        (split, condition): _distribution(
            [row for row in rows if row.get("split") == split and row.get("condition") == condition],
            field="manual_formula_score",
        )
        for split, condition in (
            ("locked_test", "native_normal"),
            ("locked_test", "flat"),
            ("locked_test", "shuffled"),
            ("locked_test", "wrong_drop"),
            ("external_janon_native", "native_normal"),
            ("external_janon_learner", "learner_observed"),
        )
    }
    lines = [
        "# Pitch Naturalness Lightweight Calibration Experiment",
        "",
        "## Scope",
        "",
        "- Offline experiment only; the runtime scoring formula is unchanged.",
        "- Model: standardized logistic regression.",
        "- Training target: distinguish real JVS native contours from paired flat/shuffled F0 controls.",
        "- This is a native-likeness proxy, not teacher-rated pitch-accent correctness.",
        "- No JVS/JANON speaker appears in more than one internal split.",
        "",
        "## Data Split",
        "",
        "| split | speakers | real native recordings | training use |",
        "|---|---:|---:|---|",
        f"| train (`jvs001–030`) | {split_counts['train'][0]} | {split_counts['train'][1]} | fit scaler and coefficients |",
        f"| dev (`jvs031–040`) | {split_counts['dev'][0]} | {split_counts['dev'][1]} | choose decision threshold |",
        f"| locked test (`jvs041–100`) | {split_counts['locked_test'][0]} | {split_counts['locked_test'][1]} | final internal evaluation only |",
        f"| JANON external | {split_counts['janon'][0]} | {split_counts['janon'][1]} | distribution audit only; no labels |",
        "",
        "Each available JVS real recording has paired flat and shuffled controls. Wrong-drop is never used for fitting and remains a challenge set.",
        "The local nonpara/lab requirements excluded jvs006 and jvs028 from training, so the actual training set has 28 speakers rather than the planned 30.",
        "",
        "## Candidate Score Distribution",
        "",
        "| split | condition | n | mean | p10 | p50 | p90 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['split']} | {row['condition']} | {row['n']} | {row['mean']} | {row['p10']} | {row['p50']} | {row['p90']} |")
    metrics = artifact["metrics"]
    lines.extend([
        "",
        "## Locked-Test Results",
        "",
        f"- ROC AUC for native vs flat/shuffled: `{metrics['locked_test_auc']}`.",
        f"- Balanced accuracy at the dev-selected threshold `{artifact['decision_threshold']}`: `{metrics['locked_test_balanced_accuracy']}`.",
        f"- Native beats paired flat: `{_paired_win_rate(test_rows, 'flat')}`.",
        f"- Native beats paired shuffled: `{_paired_win_rate(test_rows, 'shuffled')}`.",
        f"- Native beats paired wrong-drop: `{_paired_win_rate(test_rows, 'wrong_drop')}`.",
        f"- Locked JVS native candidate mean: `{stat('locked_test', 'native_normal')}`.",
        f"- Flat candidate mean: `{stat('locked_test', 'flat')}`; shuffled mean: `{stat('locked_test', 'shuffled')}`.",
        f"- Wrong-drop candidate mean: `{stat('locked_test', 'wrong_drop')}`.",
        "",
        "## Existing Formula vs Trained Candidate",
        "",
        "| evaluation group | existing mean | trained candidate mean | observation |",
        "|---|---:|---:|---|",
        f"| held-out JVS native | {manual[('locked_test', 'native_normal')]['mean']} | {stat('locked_test', 'native_normal')} | native is higher |",
        f"| paired flat | {manual[('locked_test', 'flat')]['mean']} | {stat('locked_test', 'flat')} | stronger rejection |",
        f"| paired shuffled | {manual[('locked_test', 'shuffled')]['mean']} | {stat('locked_test', 'shuffled')} | stronger rejection |",
        f"| wrong-drop challenge | {manual[('locked_test', 'wrong_drop')]['mean']} | {stat('locked_test', 'wrong_drop')} | separation is not improved |",
        f"| JANON native | {manual[('external_janon_native', 'native_normal')]['mean']} | {stat('external_janon_native', 'native_normal')} | mean similar, but candidate p10 is {stat('external_janon_native', 'native_normal', 'p10')} |",
        f"| JANON learner | {manual[('external_janon_learner', 'learner_observed')]['mean']} | {stat('external_janon_learner', 'learner_observed')} | no teacher labels; not accuracy evidence |",
        "",
        "## Interpretation",
        "",
        "- Training is useful if held-out native remains high while flat/shuffled remain low.",
        "- The trained probabilities are strongly saturated near 0 or 100; they are classifier confidence, not calibrated 0–100 educational scores.",
        "- JANON native/learner overlap is expected because JANON has no teacher pitch labels and uses approximate equal-mora timing. The very low JANON-native p10 shows domain sensitivity remains.",
        "- A high wrong-drop score means this model still measures broad contour naturalness, not lexical pitch-accent correctness.",
        "- Do not activate this candidate in the product until human ratings are collected and score calibration is validated.",
        "",
        "## Next Data Needed",
        "",
        "1. Teacher/listener ratings for naturalness and intelligibility on real learner recordings.",
        "2. Independently recorded flat, unstable, and wrong-accent speech rather than F0-only counterfactuals.",
        "3. More female/mixed-condition external speech with reliable mora timing.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an offline speaker-disjoint weak pitch-naturalness calibrator.")
    parser.add_argument("--jvs-root", type=Path, default=PROJECT_ROOT / "JVS")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/pitch_calibration_train_dev.jsonl")
    parser.add_argument("--locked-results", type=Path, default=ROOT / "results/calibration/pitch_heldout_validation.csv")
    parser.add_argument("--out-rows", type=Path, default=ROOT / "results/calibration/pitch_naturalness_calibration_rows.csv")
    parser.add_argument("--out-summary", type=Path, default=ROOT / "data/calibration_candidates/pitch_naturalness_calibration_summary.csv")
    parser.add_argument("--out-model", type=Path, default=ROOT / "data/calibration_candidates/pitch_naturalness_calibrator_candidate.json")
    parser.add_argument("--out-report", type=Path, default=ROOT / "reports/pitch_naturalness_calibration_experiment.md")
    args = parser.parse_args()

    development = extract_development_rows(args.jvs_root, args.checkpoint, sample_rate=args.sample_rate)
    train = [row for row in development if row["split"] == "train" and row["condition"] in {"native_normal", "flat", "shuffled"} and _has_features(row)]
    dev = [row for row in development if row["split"] == "dev" and row["condition"] in {"native_normal", "flat", "shuffled"} and _has_features(row)]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=20260623)),
    ])
    model.fit(_matrix(train), _labels(train))
    dev_prob = model.predict_proba(_matrix(dev))[:, 1]
    threshold, dev_balanced = _select_threshold(_labels(dev), dev_prob)

    locked = _load_locked_rows(args.locked_results)
    locked_labeled = [row for row in locked if row["split"] == "locked_test" and row["condition"] in {"native_normal", "flat", "shuffled"} and _has_features(row)]
    test_prob = model.predict_proba(_matrix(locked_labeled))[:, 1]
    test_y = _labels(locked_labeled)
    metrics = {
        "train_rows": len(train),
        "dev_rows": len(dev),
        "locked_test_rows": len(locked_labeled),
        "dev_auc": round(float(roc_auc_score(_labels(dev), dev_prob)), 6),
        "dev_balanced_accuracy": round(float(dev_balanced), 6),
        "locked_test_auc": round(float(roc_auc_score(test_y, test_prob)), 6),
        "locked_test_balanced_accuracy": round(float(balanced_accuracy_score(test_y, test_prob >= threshold)), 6),
    }
    artifact = _model_artifact(
        model,
        threshold,
        metrics,
        train_speakers=sorted({str(row["speaker_id"]) for row in train}),
        dev_speakers=sorted({str(row["speaker_id"]) for row in dev}),
        test_speakers=sorted({str(row["speaker_id"]) for row in locked_labeled}),
    )
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    args.out_model.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    all_rows = _apply_model(model, [dict(row) for row in development + locked], threshold)
    summary = _summary_rows(all_rows)
    _write_csv(args.out_rows, all_rows)
    _write_csv(args.out_summary, summary)
    _report(args.out_report, summary, artifact, all_rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"wrote {args.out_model}")
    print(f"wrote {args.out_report}")


if __name__ == "__main__":
    main()
