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

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_cross_speaker_prosody_reference import _lab_f0, flat_f0, low_f0_coverage, shuffled_f0, wrong_drop_f0  # noqa: E402
from audit_pitch_heldout_validation import _janon_f0, _read_csv, _transcripts  # noqa: E402
from jp_speech_eval.pitch_naturalness_v2 import combine_naturalness_with_hint, load_pitch_naturalness_v2_config, score_pitch_naturalness_v2  # noqa: E402
from jp_speech_eval.scoring import score_weak_reference_native_likeness  # noqa: E402
from jp_speech_eval.text_frontend import build_text_info  # noqa: E402


FRESH_JVS_SPEAKERS = tuple(f"jvs{index:03d}" for index in range(71, 101))
FRESH_UTTERANCES_PER_SPEAKER = 10


def _fresh_jvs_manifest(jvs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for speaker_id in FRESH_JVS_SPEAKERS:
        directory = jvs_root / speaker_id / "parallel100"
        transcript_path = directory / "transcripts_utf8.txt"
        if not transcript_path.exists():
            continue
        transcripts = _transcripts(transcript_path)
        ids = sorted(
            set(transcripts)
            & {path.stem for path in (directory / "wav24kHz16bit").glob("*.wav")}
            & {path.stem for path in (directory / "lab" / "mon").glob("*.lab")}
        )
        indexes = np.linspace(0, len(ids) - 1, FRESH_UTTERANCES_PER_SPEAKER, dtype=int)
        for index in indexes:
            utterance_id = ids[int(index)]
            rows.append({
                "sample_id": f"{speaker_id}:{utterance_id}",
                "dataset": "JVS",
                "group": "fresh_jvs_parallel100",
                "speaker_id": speaker_id,
                "reference_text": transcripts[utterance_id],
                "audio_path": str((directory / "wav24kHz16bit" / f"{utterance_id}.wav").resolve()),
                "lab_path": str((directory / "lab" / "mon" / f"{utterance_id}.lab").resolve()),
            })
    return rows


def _janon_manifest(janon_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in _read_csv(janon_root / "data.csv"):
        if source["Stimulus Type"] != "sentence":
            continue
        wav = janon_root / source["Path"].replace("/sentence/", "/sentences/")
        if not wav.exists():
            continue
        native = source["Native Language"] == "Japanese"
        rows.append({
            "sample_id": f"{source['Speaker']}:{Path(source['Path']).stem}",
            "dataset": "JANON",
            "group": "janon_native_external" if native else "janon_learner_external",
            "speaker_id": source["Speaker"],
            "reference_text": source["Stmiulus"].strip().lstrip("\ufeff"),
            "audio_path": str(wav.resolve()),
            "lab_path": "",
        })
    return rows


def _score(
    item: Mapping[str, Any],
    condition: str,
    f0_values: Sequence[float],
    text_info: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    v1_score, _feedback, weak = score_weak_reference_native_likeness(list(f0_values))
    v2_score, v2 = score_pitch_naturalness_v2(
        f0_values,
        weak,
        text_info.target_pitch,
        text_info.accent_phrases,
        pitch_target_source=text_info.pitch_target_source,
        is_question=bool(text_info.is_question),
        config=config,
    )
    return {
        **dict(item),
        "condition": condition,
        "mora_count": len(text_info.moras),
        "is_question": bool(text_info.is_question),
        "pitch_target_source": text_info.pitch_target_source,
        "v1_score": v1_score,
        "v2_score": v2_score,
        "v2_data_naturalness": v2.get("data_calibrated_naturalness_score"),
        "accent_hint_score": v2.get("accent_hint_score"),
        "accent_hint_weight": v2.get("accent_hint_weight"),
        "accent_event_count": (v2.get("accent_hint_details") or {}).get("event_count"),
        "f0_coverage": weak.get("f0_coverage"),
        "available": bool(v2.get("available")),
        "unavailable_reason": v2.get("reason"),
    }


def run(
    jvs_root: Path,
    janon_root: Path,
    checkpoint: Path,
    *,
    sample_rate: int,
) -> list[dict[str, Any]]:
    config = load_pitch_naturalness_v2_config()
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            naturalness = row.get("v2_data_naturalness")
            if naturalness is not None:
                updated = combine_naturalness_with_hint(
                    float(naturalness),
                    float(row["accent_hint_score"]) if row.get("accent_hint_score") is not None else None,
                    float(row.get("accent_hint_weight") or 0.0),
                )
                row["v2_score"] = int(round(updated))
                row["v2_combination_policy"] = "mismatch_penalty_only"
            completed[(row["sample_id"], row["condition"])] = row
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    manifest = _fresh_jvs_manifest(jvs_root) + _janon_manifest(janon_root)
    for index, item in enumerate(manifest, start=1):
        conditions = ("normal", "flat", "shuffled", "wrong_drop", "low_f0") if item["dataset"] == "JVS" else ("observed",)
        if all((item["sample_id"], condition) in completed for condition in conditions):
            continue
        text_info = build_text_info(str(item["reference_text"]))
        if item["dataset"] == "JVS":
            f0_values, _meta = _lab_f0(item, text_info.moras, sample_rate=sample_rate)
        else:
            f0_values, _meta = _janon_f0(item, text_info.moras, sample_rate=sample_rate)
        if f0_values is None:
            continue
        variants = {"observed": f0_values}
        if item["dataset"] == "JVS":
            variants = {
                "normal": f0_values,
                "flat": flat_f0(f0_values),
                "shuffled": shuffled_f0(f0_values, seed=sum(ord(char) for char in str(item["sample_id"]))),
                "wrong_drop": wrong_drop_f0(f0_values, f0_values, text_info.accent_phrases),
                "low_f0": low_f0_coverage(f0_values),
            }
        for condition, values in variants.items():
            key = (str(item["sample_id"]), condition)
            if key in completed:
                continue
            row = _score(item, condition, values, text_info, config)
            with checkpoint.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            completed[key] = row
        print(f"[{index}/{len(manifest)}] {item['sample_id']}", flush=True)
    return list(completed.values())


def _stats(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = np.asarray([float(row[field]) for row in rows if row.get(field) not in (None, "")], dtype=float)
    if not values.size:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None, "at_floor": None, "at_ceiling": None}
    return {
        "n": len(values),
        "mean": round(float(np.mean(values)), 4),
        "p10": round(float(np.percentile(values, 10)), 4),
        "p50": round(float(np.percentile(values, 50)), 4),
        "p90": round(float(np.percentile(values, 90)), 4),
        "at_floor": round(float(np.mean(values <= 10)), 4),
        "at_ceiling": round(float(np.mean(values >= 98)), 4),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["group"]), str(row["condition"]))].append(row)
    out: list[dict[str, Any]] = []
    for (group, condition), items in sorted(grouped.items()):
        v1 = _stats(items, "v1_score")
        v2 = _stats(items, "v2_score")
        hint = _stats(items, "accent_hint_score")
        out.append({
            "group": group,
            "condition": condition,
            **{f"v1_{key}": value for key, value in v1.items()},
            **{f"v2_{key}": value for key, value in v2.items()},
            "accent_hint_n": hint["n"],
            "accent_hint_mean": hint["mean"],
            "accent_hint_p10": hint["p10"],
            "accent_hint_p50": hint["p50"],
            "accent_hint_p90": hint["p90"],
            "unavailable_count": sum(row.get("v2_score") is None for row in items),
            "speaker_count": len({str(row["speaker_id"]) for row in items}),
        })
    return out


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


def _report(path: Path, summary: Sequence[Mapping[str, Any]]) -> None:
    lookup = {(row["group"], row["condition"]): row for row in summary}
    jvs = lookup[("fresh_jvs_parallel100", "normal")]
    flat = lookup[("fresh_jvs_parallel100", "flat")]
    shuffled = lookup[("fresh_jvs_parallel100", "shuffled")]
    wrong = lookup[("fresh_jvs_parallel100", "wrong_drop")]
    janon_native = lookup[("janon_native_external", "observed")]
    janon_learner = lookup[("janon_learner_external", "observed")]
    lines = [
        "# Pitch Naturalness v2 Cross-Dataset Audit",
        "",
        "- activation decision: passed the documented guardrails and enabled for weak-reference practice only",
        "- naturalness model: continuous ridge calibration, not binary probability",
        "- accent target: automatic OpenJTalk accent-phrase chain used as an OJAD-style weak mismatch penalty at at most 8% weight",
        "- the accent hint can never boost a low naturalness score",
        "- official OJAD output is not scraped or treated as ground truth",
        "- sentence-final 2 morae (3 for questions) are excluded from accent-hint matching",
        "",
        "## Distribution",
        "",
        "| group | condition | n | v1 mean | v2 mean | v2 p10 | v2 p50 | v2 p90 | accent hint mean | unavailable |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['group']} | {row['condition']} | {row['v2_n']} | {row['v1_mean']} | {row['v2_mean']} | "
            f"{row['v2_p10']} | {row['v2_p50']} | {row['v2_p90']} | {row['accent_hint_mean']} | {row['unavailable_count']} |"
        )
    lines.extend([
        "",
        "## Product Checks",
        "",
        f"- Fresh JVS normal: mean `{jvs['v2_mean']}`, p10 `{jvs['v2_p10']}`, ceiling rate `{jvs['v2_at_ceiling']}`.",
        f"- Flat: mean `{flat['v2_mean']}`; shuffled: `{shuffled['v2_mean']}`.",
        f"- Wrong-drop: mean `{wrong['v2_mean']}`. It is not required to be low for broad naturalness and is not used as a training negative.",
        f"- JANON native: mean `{janon_native['v2_mean']}`, p10 `{janon_native['v2_p10']}`.",
        f"- JANON learner: mean `{janon_learner['v2_mean']}`. No teacher labels exist, so this is not an accuracy metric.",
        "- Low-F0 controls must remain unavailable internally; the UI numeric fallback is a low-confidence practice estimate.",
        "",
        "## Decision Rule",
        "",
        "Activation requires: fresh JVS normal clearly above flat/shuffle, no 0/100 probability collapse, JANON native without catastrophic collapse, and automatic accent hints changing the total only mildly. Strict accent-error claims remain prohibited.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pitch naturalness v2 on fresh JVS recordings and external JANON.")
    parser.add_argument("--jvs-root", type=Path, default=PROJECT_ROOT / "JVS")
    parser.add_argument("--janon-root", type=Path, default=PROJECT_ROOT / "JANON")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/pitch_naturalness_v2_audit.jsonl")
    parser.add_argument("--out-rows", type=Path, default=ROOT / "results/calibration/pitch_naturalness_v2_audit.csv")
    parser.add_argument("--out-summary", type=Path, default=ROOT / "data/calibration_candidates/pitch_naturalness_v2_audit_summary.csv")
    parser.add_argument("--out-report", type=Path, default=ROOT / "reports/pitch_naturalness_v2_audit.md")
    args = parser.parse_args()
    rows = run(args.jvs_root, args.janon_root, args.checkpoint, sample_rate=args.sample_rate)
    summary = summarize(rows)
    _write_csv(args.out_rows, rows)
    _write_csv(args.out_summary, summary)
    _report(args.out_report, summary)
    print(f"wrote {args.out_report}")


if __name__ == "__main__":
    main()
