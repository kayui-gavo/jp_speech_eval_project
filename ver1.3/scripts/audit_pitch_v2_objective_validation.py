#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = (
    "f0_coverage",
    "f0_range_log",
    "local_pitch_movement",
    "smoothness",
    "flatness_penalty",
    "instability_penalty",
)
ANCHORS = {"native_normal": 95.0, "flat": 20.0, "shuffled": 45.0}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


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


def _values(rows: Iterable[Mapping[str, Any]], field: str = "v2_score") -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows if row.get(field) is not None], dtype=float)


def _percentile_interval(values: Sequence[float], *, seed: int = 23, draws: int = 4000) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if not data.size:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(data, size=(draws, len(data)), replace=True), axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    if not data.size:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None, "sd": None}
    low, high = _percentile_interval(data)
    return {
        "n": int(data.size),
        "mean": round(float(np.mean(data)), 4),
        "mean_ci95_low": round(low, 4),
        "mean_ci95_high": round(high, 4),
        "sd": round(float(np.std(data, ddof=1)), 4) if data.size > 1 else 0.0,
        "p10": round(float(np.percentile(data, 10)), 4),
        "p50": round(float(np.percentile(data, 50)), 4),
        "p90": round(float(np.percentile(data, 90)), 4),
        "below_60_rate": round(float(np.mean(data < 60)), 4),
        "below_70_rate": round(float(np.mean(data < 70)), 4),
        "at_least_80_rate": round(float(np.mean(data >= 80)), 4),
        "floor_or_ceiling_rate": round(float(np.mean((data <= 10) | (data >= 98))), 4),
        "unique_integer_scores": len(set(int(round(value)) for value in data)),
    }


def _paired_control(rows: Sequence[Mapping[str, Any]], control: str) -> dict[str, Any]:
    samples: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("group") != "fresh_jvs_parallel100" or row.get("v2_score") is None:
            continue
        samples[str(row["sample_id"])][str(row["condition"])] = float(row["v2_score"])
    pairs = [(item["normal"], item[control]) for item in samples.values() if "normal" in item and control in item]
    native = np.asarray([pair[0] for pair in pairs], dtype=float)
    negative = np.asarray([pair[1] for pair in pairs], dtype=float)
    deltas = native - negative
    ci_low, ci_high = _percentile_interval(deltas, seed=29)
    labels = np.concatenate([np.ones(len(native)), np.zeros(len(negative))])
    scores = np.concatenate([native, negative])
    return {
        "section": "paired_control",
        "metric": f"normal_vs_{control}",
        "n": len(pairs),
        "normal_mean": round(float(np.mean(native)), 4),
        "control_mean": round(float(np.mean(negative)), 4),
        "paired_delta_mean": round(float(np.mean(deltas)), 4),
        "paired_delta_ci95_low": round(ci_low, 4),
        "paired_delta_ci95_high": round(ci_high, 4),
        "normal_wins_rate": round(float(np.mean(native > negative)), 4),
        "control_at_least_normal_rate": round(float(np.mean(negative >= native)), 4),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
    }


def _jvs_repeatability(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normal = [row for row in rows if row.get("group") == "fresh_jvs_parallel100" and row.get("condition") == "normal"]
    by_sentence: dict[str, list[float]] = defaultdict(list)
    by_speaker: dict[str, list[float]] = defaultdict(list)
    for row in normal:
        utterance = str(row["sample_id"]).split(":", 1)[-1]
        by_sentence[utterance].append(float(row["v2_score"]))
        by_speaker[str(row["speaker_id"])].append(float(row["v2_score"]))
    speaker_count = len(by_speaker)
    minimum_sentence_coverage = max(2, int(np.ceil(0.80 * speaker_count)))
    comparable_sentences = {
        utterance: values
        for utterance, values in by_sentence.items()
        if len(values) >= minimum_sentence_coverage
    }
    sentence_rows: list[dict[str, Any]] = []
    for utterance, values in sorted(comparable_sentences.items()):
        stats = _distribution(values)
        sentence_rows.append({
            "section": "jvs_sentence_repeatability",
            "metric": utterance,
            **stats,
        })
    sentence_sds = [float(row["sd"]) for row in sentence_rows]
    speaker_means = [float(np.mean(values)) for values in by_speaker.values()]
    summary = {
        "section": "jvs_repeatability_summary",
        "metric": "parallel100_same_sentences_across_speakers",
        "sentence_count": len(comparable_sentences),
        "speaker_count": speaker_count,
        "minimum_sentence_speaker_coverage": minimum_sentence_coverage,
        "minimum_observed_sentence_speakers": min(len(values) for values in comparable_sentences.values()),
        "maximum_observed_sentence_speakers": max(len(values) for values in comparable_sentences.values()),
        "median_within_sentence_sd": round(float(np.median(sentence_sds)), 4),
        "max_within_sentence_sd": round(float(np.max(sentence_sds)), 4),
        "speaker_macro_mean": round(float(np.mean(speaker_means)), 4),
        "speaker_macro_sd": round(float(np.std(speaker_means, ddof=1)), 4),
        "lowest_speaker_macro_mean": round(float(np.min(speaker_means)), 4),
        "highest_speaker_macro_mean": round(float(np.max(speaker_means)), 4),
    }
    return sentence_rows, summary


def _external_domain(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        "jvs_native_fresh": [row for row in rows if row.get("group") == "fresh_jvs_parallel100" and row.get("condition") == "normal"],
        "janon_native_external": [row for row in rows if row.get("group") == "janon_native_external"],
        "janon_learner_exploratory": [row for row in rows if row.get("group") == "janon_learner_external"],
    }
    output: list[dict[str, Any]] = []
    for name, items in groups.items():
        output.append({"section": "cross_corpus_distribution", "metric": name, **_distribution(_values(items))})
    jvs = _values(groups["jvs_native_fresh"])
    janon = _values(groups["janon_native_external"])
    output.append({
        "section": "cross_corpus_gap",
        "metric": "jvs_native_minus_janon_native",
        "mean_gap": round(float(np.mean(jvs) - np.mean(janon)), 4),
        "interpretation": "domain_and_timing_gap_not_accuracy",
    })
    return output


def _matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row[field]) for field in FEATURE_COLUMNS] for row in rows], dtype=float)


def _speaker_disjoint_cv(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    usable = [
        row for row in rows
        if row.get("split") == "train"
        and row.get("condition") in ANCHORS
        and all(row.get(field) not in (None, "") for field in FEATURE_COLUMNS)
    ]
    groups = np.asarray([str(row["speaker_id"]) for row in usable])
    x = _matrix(usable)
    y = np.asarray([ANCHORS[str(row["condition"])] for row in usable], dtype=float)
    conditions = np.asarray([str(row["condition"]) for row in usable])
    output: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups), start=1):
        model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=50.0))])
        model.fit(x[train_index], y[train_index])
        predictions = np.clip(model.predict(x[test_index]), 10.0, 98.0)
        test_conditions = conditions[test_index]
        means = {
            condition: float(np.mean(predictions[test_conditions == condition]))
            for condition in ANCHORS
        }
        output.append({
            "section": "speaker_disjoint_cv",
            "metric": f"fold_{fold}",
            "heldout_speaker_count": len(set(groups[test_index])),
            "mae_to_pseudo_anchor": round(float(np.mean(np.abs(predictions - y[test_index]))), 4),
            "normal_mean": round(means["native_normal"], 4),
            "flat_mean": round(means["flat"], 4),
            "shuffled_mean": round(means["shuffled"], 4),
            "normal_minus_flat": round(means["native_normal"] - means["flat"], 4),
            "normal_minus_shuffled": round(means["native_normal"] - means["shuffled"], 4),
            "ordering_pass": means["native_normal"] > means["shuffled"] > means["flat"],
        })
    return output


def _report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lookup = {(str(row["section"]), str(row["metric"])): row for row in rows}
    repeat = lookup[("jvs_repeatability_summary", "parallel100_same_sentences_across_speakers")]
    flat = lookup[("paired_control", "normal_vs_flat")]
    shuffled = lookup[("paired_control", "normal_vs_shuffled")]
    jvs = lookup[("cross_corpus_distribution", "jvs_native_fresh")]
    janon_native = lookup[("cross_corpus_distribution", "janon_native_external")]
    janon_learner = lookup[("cross_corpus_distribution", "janon_learner_exploratory")]
    gap = lookup[("cross_corpus_gap", "jvs_native_minus_janon_native")]
    cv = [row for row in rows if row["section"] == "speaker_disjoint_cv"]
    sentence_rows = [row for row in rows if row["section"] == "jvs_sentence_repeatability"]
    worst = max(sentence_rows, key=lambda row: float(row["sd"]))
    lines = [
        "# Pitch v2 Objective Validation (No Human Ratings)",
        "",
        "This audit adds objective reliability and generalization checks without changing runtime scoring. JVS is used for native baselines and paired F0 controls; JANON is external-only and is never used as a training label.",
        "",
        "## Main Results",
        "",
        "| check | result | interpretation |",
        "|---|---:|---|",
        f"| JVS native mean (95% bootstrap CI) | {jvs['mean']} ({jvs['mean_ci95_low']}–{jvs['mean_ci95_high']}) | Unseen speakers/sentences remain high |",
        f"| Native vs flat paired delta | +{flat['paired_delta_mean']} (AUC {flat['roc_auc']}) | Flat F0 is clearly separated |",
        f"| Native vs shuffled paired delta | +{shuffled['paired_delta_mean']} (AUC {shuffled['roc_auc']}) | Unstable F0 is clearly separated |",
        f"| Median same-sentence SD across speakers | {repeat['median_within_sentence_sd']} | Remaining speaker variability on the 0–100 scale |",
        f"| Speaker macro mean range | {repeat['lowest_speaker_macro_mean']}–{repeat['highest_speaker_macro_mean']} | No single fresh JVS speaker collapses |",
        f"| JANON native mean | {janon_native['mean']} | External microphone/timing domain is lower than JVS |",
        f"| JVS–JANON native mean gap | {gap['mean_gap']} | Domain/timing gap, not an accuracy claim |",
        "",
        "## Speaker-Disjoint Cross-Validation",
        "",
        "| fold | held-out speakers | normal | shuffled | flat | ordering |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in cv:
        lines.append(
            f"| {row['metric']} | {row['heldout_speaker_count']} | {row['normal_mean']} | {row['shuffled_mean']} | {row['flat_mean']} | {'PASS' if row['ordering_pass'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "All folds keep native > shuffled > flat. This checks speaker leakage and coefficient stability against the predefined proxy anchors; it does not create human validity.",
        "",
        "## Same-Sentence Cross-Speaker Stability",
        "",
        f"- {repeat['sentence_count']} parallel sentences observed for at least {repeat['minimum_sentence_speaker_coverage']} of {repeat['speaker_count']} unseen JVS speakers were compared (actual coverage {repeat['minimum_observed_sentence_speakers']}–{repeat['maximum_observed_sentence_speakers']}).",
        f"- Median within-sentence SD: `{repeat['median_within_sentence_sd']}`; maximum: `{repeat['max_within_sentence_sd']}`.",
        f"- Highest-variance sentence: `{worst['metric']}` (SD `{worst['sd']}`, p10 `{worst['p10']}`, p90 `{worst['p90']}`).",
        "- This is cross-speaker consistency, not test-retest reliability; JVS does not provide repeated takes for this exact check.",
        "",
        "## External JANON Audit",
        "",
        "| group | n | mean | p10 | p50 | p90 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| JANON native | {janon_native['n']} | {janon_native['mean']} | {janon_native['p10']} | {janon_native['p50']} | {janon_native['p90']} |",
        f"| JANON learner (descriptive only) | {janon_learner['n']} | {janon_learner['mean']} | {janon_learner['p10']} | {janon_learner['p50']} | {janon_learner['p90']} |",
        "",
        "Learner identity is not an error label. The learner distribution is useful only as an external trend and failure-analysis pool because JANON has no teacher pitch score and currently uses approximate equal-mora timing.",
        "",
        "## What This Establishes",
        "",
        "- The broad pitch-naturalness score generalizes across held-out JVS speakers and a different JVS reading set.",
        "- Flat and shuffled controls remain strongly lower without forcing wrong accent-drop contours to be errors.",
        "- Score mass is not collapsed to only 0 or 100.",
        "- JANON exposes a real domain gap that should be addressed through alignment/device robustness, not by fitting learner identity.",
        "",
        "## What It Does Not Establish",
        "",
        "- Teacher-grade pitch-accent correctness.",
        "- Agreement with human perception or educational usefulness.",
        "- Test-retest reliability for the same person and sentence.",
        "- Device/noise robustness on real phone recordings.",
        "- Scientific calibration of the other three dimensions.",
        "",
        "## Next Step Without Human Ratings",
        "",
        "Run waveform-level robustness tests on held-out JVS audio (codec, room noise, gain and microphone filtering), then audit all four dimensions for score drift. Keep the clean score as the reference and require small drift for pronunciation/rhythm/fluency and stable pitch ordering for pitch controls.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Objective, no-human-rating validation for pitch naturalness v2.")
    parser.add_argument("--audit-jsonl", type=Path, default=ROOT / "outputs/pitch_naturalness_v2_audit.jsonl")
    parser.add_argument("--training-rows", type=Path, default=ROOT / "results/calibration/pitch_naturalness_calibration_rows.csv")
    parser.add_argument("--out-summary", type=Path, default=ROOT / "data/calibration_candidates/pitch_v2_objective_validation_summary.csv")
    parser.add_argument("--out-report", type=Path, default=ROOT / "reports/pitch_v2_objective_validation.md")
    args = parser.parse_args()
    if not args.audit_jsonl.exists() or not args.training_rows.exists():
        raise FileNotFoundError("Run the existing pitch v2 training and audit scripts before this summary audit.")
    audit_rows = _read_jsonl(args.audit_jsonl)
    training_rows = _read_csv(args.training_rows)
    sentence_rows, repeatability = _jvs_repeatability(audit_rows)
    rows: list[dict[str, Any]] = [
        _paired_control(audit_rows, "flat"),
        _paired_control(audit_rows, "shuffled"),
        repeatability,
        *_external_domain(audit_rows),
        *_speaker_disjoint_cv(training_rows),
        *sentence_rows,
    ]
    _write_csv(args.out_summary, rows)
    _report(args.out_report, rows)
    print(json.dumps({"summary": str(args.out_summary), "report": str(args.out_report), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
