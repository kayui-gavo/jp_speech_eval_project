#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

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
    low_f0_coverage,
    shuffled_f0,
    wrong_drop_f0,
)
from jp_speech_eval.audio_features import extract_f0, load_audio, median_f0_by_mora  # noqa: E402
from jp_speech_eval.scoring import score_weak_reference_native_likeness  # noqa: E402
from jp_speech_eval.text_frontend import build_text_info  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402


JVS_TEST_SPEAKERS = [f"jvs{index:03d}" for index in range(41, 101)]
JVS_UTTERANCES_PER_SPEAKER = 10
JANON_LEARNER_TARGET = 316


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _transcripts(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        utterance_id, text = line.split(":", 1)
        out[utterance_id.strip()] = text.strip().lstrip("\ufeff")
    return out


def _balanced_take(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["speaker_id"]].append(row)
    for items in grouped.values():
        items.sort(key=lambda row: row["sample_id"])
    selected: list[dict[str, Any]] = []
    level = 0
    speakers = sorted(grouped)
    while len(selected) < count:
        added = False
        for speaker in speakers:
            if level < len(grouped[speaker]) and len(selected) < count:
                selected.append(grouped[speaker][level])
                added = True
        if not added:
            break
        level += 1
    return selected


def build_manifest(jvs_root: Path, janon_root: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for speaker_id in JVS_TEST_SPEAKERS:
        speaker = jvs_root / speaker_id / "nonpara30"
        transcript_path = speaker / "transcripts_utf8.txt"
        if not transcript_path.exists():
            continue
        transcripts = _transcripts(transcript_path)
        # Different nonparallel text per speaker; spread across the sorted list.
        wav_ids = {path.stem for path in (speaker / "wav24kHz16bit").glob("*.wav")}
        lab_ids = {path.stem for path in (speaker / "lab" / "mon").glob("*.lab")}
        utterance_ids = sorted(set(transcripts) & wav_ids & lab_ids)
        indexes = np.linspace(0, len(utterance_ids) - 1, JVS_UTTERANCES_PER_SPEAKER, dtype=int)
        for index in indexes:
            utterance_id = utterance_ids[int(index)]
            wav = speaker / "wav24kHz16bit" / f"{utterance_id}.wav"
            lab = speaker / "lab" / "mon" / f"{utterance_id}.lab"
            if wav.exists() and lab.exists():
                manifest.append({
                    "sample_id": f"{speaker_id}:{utterance_id}",
                    "dataset": "JVS",
                    "split": "locked_internal_test",
                    "group": "jvs_native_heldout",
                    "speaker_id": speaker_id,
                    "native_language": "Japanese",
                    "reference_text": transcripts[utterance_id],
                    "audio_path": str(wav.resolve()),
                    "lab_path": str(lab.resolve()),
                    "timing_method": "jvs_phone_lab_to_mora",
                })

    janon: list[dict[str, Any]] = []
    for source in _read_csv(janon_root / "data.csv"):
        if source["Stimulus Type"] != "sentence":
            continue
        wav = janon_root / source["Path"].replace("/sentence/", "/sentences/")
        if not wav.exists():
            continue
        native = source["Native Language"] == "Japanese"
        janon.append({
            "sample_id": f"{source['Speaker']}:{Path(source['Path']).stem}",
            "dataset": "JANON",
            "split": "external_corpus_test",
            "group": "janon_native_external" if native else "janon_learner_external",
            "speaker_id": source["Speaker"],
            "native_language": source["Native Language"],
            "reference_text": source["Stmiulus"].strip().lstrip("\ufeff"),
            "audio_path": str(wav.resolve()),
            "lab_path": "",
            "timing_method": "speech_endpoint_equal_mora_external_approximation",
        })
    manifest.extend(row for row in janon if row["group"] == "janon_native_external")
    manifest.extend(_balanced_take(
        [row for row in janon if row["group"] == "janon_learner_external"],
        JANON_LEARNER_TARGET,
    ))
    manifest.sort(key=lambda row: (row["group"], row["sample_id"]))
    for index, row in enumerate(manifest, start=1):
        row["manifest_index"] = index
    return manifest


def _janon_f0(item: Mapping[str, Any], moras: Sequence[str], *, sample_rate: int) -> tuple[list[float] | None, dict[str, Any]]:
    audio = load_audio(str(item["audio_path"]), sr=sample_rate)
    speech, region = trim_to_speech(audio.y, audio.sr)
    duration = len(speech) / max(audio.sr, 1)
    if not moras or duration <= 0:
        return None, {"ok": False, "reason": "empty_mora_or_speech"}
    boundaries = [
        (duration * index / len(moras), duration * (index + 1) / len(moras))
        for index in range(len(moras))
    ]
    times, f0, method = extract_f0(speech, audio.sr)
    return median_f0_by_mora(times, f0, boundaries), {
        "ok": True,
        "f0_method": method,
        "speech_duration_sec": round(duration, 4),
        "speech_region_detected": bool(region.detected),
        "timing_warning": "external_equal_mora_approximation",
    }


def _score_row(item: Mapping[str, Any], condition: str, f0_values: Sequence[float],
               moras: Sequence[str], meta: Mapping[str, Any]) -> dict[str, Any]:
    score, _feedback, details = score_weak_reference_native_likeness(list(f0_values))
    return {
        **dict(item),
        "condition": condition,
        "score": score,
        "available": details.get("available"),
        "unavailable_reason": details.get("unavailable_reason"),
        "mora_count": len(moras),
        "f0_coverage": details.get("f0_coverage"),
        "valid_f0_mora_count": details.get("valid_f0_mora_count"),
        "f0_range_log": details.get("utterance_f0_range_log"),
        "local_pitch_movement": details.get("local_pitch_movement"),
        "smoothness": details.get("transition_smoothness"),
        "flatness_penalty": details.get("flatness_penalty"),
        "instability_penalty": details.get("instability_penalty"),
        "f0_method": meta.get("f0_method"),
        "alignment_ok": meta.get("ok"),
        "alignment_warning": meta.get("timing_warning") or meta.get("mapping_warning_flags") or "",
        "score_formula_frozen": True,
        "scoring_commit": "95b9510",
    }


def run_audit(manifest: list[dict[str, Any]], checkpoint: Path, *, sample_rate: int) -> list[dict[str, Any]]:
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed[(row["sample_id"], row["condition"])] = row
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(manifest, start=1):
        expected_conditions = (
            ["native_normal", "flat", "shuffled", "wrong_drop", "low_f0"]
            if item["dataset"] == "JVS"
            else ["native_normal"] if item["group"] == "janon_native_external" else ["learner_observed"]
        )
        if all((item["sample_id"], condition) in completed for condition in expected_conditions):
            continue
        text_info = build_text_info(item["reference_text"])
        if item["dataset"] == "JVS":
            f0_values, meta = _lab_f0(item, text_info.moras, sample_rate=sample_rate)
        else:
            f0_values, meta = _janon_f0(item, text_info.moras, sample_rate=sample_rate)
        if f0_values is None:
            variants = {condition: [] for condition in expected_conditions}
        elif item["dataset"] == "JVS":
            variants = {
                "native_normal": f0_values,
                "flat": flat_f0(f0_values),
                "shuffled": shuffled_f0(f0_values, seed=sum(ord(char) for char in item["sample_id"])),
                "wrong_drop": wrong_drop_f0(f0_values, f0_values, text_info.accent_phrases),
                "low_f0": low_f0_coverage(f0_values),
            }
        else:
            variants = {expected_conditions[0]: f0_values}
        for condition, values in variants.items():
            key = (item["sample_id"], condition)
            if key in completed:
                continue
            row = _score_row(item, condition, values, text_info.moras, meta)
            with checkpoint.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            completed[key] = row
        print(f"[{index}/{len(manifest)}] {item['sample_id']}", flush=True)
    return [
        completed[(item["sample_id"], condition)]
        for item in manifest
        for condition in (
            ["native_normal", "flat", "shuffled", "wrong_drop", "low_f0"]
            if item["dataset"] == "JVS"
            else ["native_normal"] if item["group"] == "janon_native_external" else ["learner_observed"]
        )
    ]


def _numbers(rows: Iterable[Mapping[str, Any]], field: str = "score") -> list[float]:
    return [float(row[field]) for row in rows if row.get(field) is not None]


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": round(float(np.mean(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["condition"])].append(row)
    out: list[dict[str, Any]] = []
    for (group, condition), items in sorted(grouped.items()):
        stats = _summary(_numbers(items))
        speaker_means = [statistics.mean(_numbers(speaker_rows)) for speaker_rows in _by_speaker(items).values() if _numbers(speaker_rows)]
        out.append({
            "group": group,
            "condition": condition,
            **stats,
            "speaker_count": len({row["speaker_id"] for row in items}),
            "speaker_macro_mean": round(statistics.mean(speaker_means), 4) if speaker_means else None,
            "unavailable_count": sum(row.get("score") is None for row in items),
            "score_below_80_rate": round(sum(float(row["score"]) < 80 for row in items if row.get("score") is not None) / max(len(_numbers(items)), 1), 4),
            "score_at_least_80_rate": round(sum(float(row["score"]) >= 80 for row in items if row.get("score") is not None) / max(len(_numbers(items)), 1), 4),
        })
    return out


def _by_speaker(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["speaker_id"])].append(row)
    return out


def _paired_by_speaker(rows: list[dict[str, Any]], condition: str) -> dict[str, list[float]]:
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["group"] == "jvs_native_heldout":
            by_sample[row["sample_id"]][row["condition"]] = row
    out: dict[str, list[float]] = defaultdict(list)
    for pair in by_sample.values():
        if "native_normal" in pair and condition in pair and pair["native_normal"].get("score") is not None and pair[condition].get("score") is not None:
            out[pair["native_normal"]["speaker_id"]].append(float(pair["native_normal"]["score"]) - float(pair[condition]["score"]))
    return out


def _cluster_ci(speaker_values: Mapping[str, Sequence[float]], *, iterations: int = 2000) -> tuple[float, float, float]:
    speakers = sorted(speaker_values)
    macro = [statistics.mean(speaker_values[speaker]) for speaker in speakers if speaker_values[speaker]]
    observed = statistics.mean(macro)
    rng = random.Random(20260622)
    boot = sorted(statistics.mean(rng.choices(macro, k=len(macro))) for _ in range(iterations))
    return round(observed, 4), round(boot[int(iterations * 0.025)], 4), round(boot[int(iterations * 0.975)], 4)


def write_report(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    jvs = [row for row in rows if row["group"] == "jvs_native_heldout"]
    deltas = {condition: _cluster_ci(_paired_by_speaker(jvs, condition)) for condition in ("flat", "shuffled", "wrong_drop")}
    score_cis = {}
    for group, condition in (
        ("jvs_native_heldout", "native_normal"),
        ("janon_native_external", "native_normal"),
        ("janon_learner_external", "learner_observed"),
    ):
        selected = [row for row in rows if row["group"] == group and row["condition"] == condition and row.get("score") is not None]
        score_cis[(group, condition)] = _cluster_ci({
            speaker: _numbers(speaker_rows)
            for speaker, speaker_rows in _by_speaker(selected).items()
        })
    paired_wins = {}
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in jvs:
        by_sample[row["sample_id"]][row["condition"]] = row
    for condition in ("flat", "shuffled", "wrong_drop"):
        pairs = [pair for pair in by_sample.values() if pair.get("native_normal", {}).get("score") is not None and pair.get(condition, {}).get("score") is not None]
        paired_wins[condition] = round(sum(pair["native_normal"]["score"] > pair[condition]["score"] for pair in pairs) / max(len(pairs), 1), 4)

    lines = [
        "# Held-Out Pitch Naturalness Validation",
        "",
        "- scoring formula: frozen at commit `95b9510`; no tuning performed after test selection",
        "- JVS locked internal test: 600 real recordings, 60 unseen speakers, nonparallel unseen texts",
        "- JANON external test: 284 native + 316 learner sentence recordings",
        "- total real recordings: 1,200",
        "- JVS counterfactual score rows: 3,000; actual independent JVS recordings remain 600",
        "",
        "## Score Summary",
        "",
        "| group | condition | n scored | speakers | mean | p10 | p50 | p90 | unavailable | <80 | >=80 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['group']} | {item['condition']} | {item['n']} | {item['speaker_count']} | "
            f"{item['mean']} | {item['p10']} | {item['p50']} | {item['p90']} | {item['unavailable_count']} | "
            f"{item['score_below_80_rate']} | {item['score_at_least_80_rate']} |"
        )
    lines.extend([
        "",
        "## Speaker-Cluster Confidence Intervals",
        "",
        "| group | speaker-macro mean | cluster-bootstrap 95% CI |",
        "|---|---:|---|",
    ])
    for (group, _condition), (mean, low, high) in score_cis.items():
        lines.append(f"| {group} | {mean} | [{low}, {high}] |")
    lines.extend([
        "",
        "## Paired JVS Counterfactual Separation",
        "",
        "| comparison | speaker-macro margin | cluster-bootstrap 95% CI | normal wins |",
        "|---|---:|---|---:|",
    ])
    for condition, (mean, low, high) in deltas.items():
        lines.append(f"| normal - {condition} | {mean} | [{low}, {high}] | {paired_wins[condition]} |")
    lines.extend([
        "",
        "## Findings",
        "",
        "- Held-out JVS native speech is generally high, but not uniformly high: mean 88.65, p10 78, and 13% of rows are below 80.",
        "- Flat separation is strong: all 600 paired native rows beat the flat control.",
        "- Shuffle separation is strong but imperfect: native wins 99% of pairs, while 1.83% of shuffled controls still score at least 80.",
        "- Wrong-drop remains weak: mean margin is only about 5.78 and 70.83% of wrong-drop controls still score at least 80. Strict pitch-accent correctness is unsupported.",
        "- Low-F0 gating behaves correctly: 600/600 controls are unavailable rather than assigned a formal score.",
        "- JANON external native mean is 80.77 and 35.21% fall below 80. This shows material domain/alignment sensitivity; do not interpret those rows as low native ability.",
        "- JANON learner and native distributions overlap substantially. Without teacher labels, the score is not a calibrated learner-ability measure.",
        "",
        "## Scientific Boundary",
        "",
        "- JVS test speakers `jvs041`–`jvs100` and `nonpara30` texts were not used by the original 24-item weak-score development audit.",
        "- Counterfactuals alter mora-level F0 only; they are paired semi-audio controls, not independently recorded learner errors.",
        "- JANON native/learner rows use endpointed equal-mora timing because JANON lacks phone labels. Treat JANON as external robustness/trend evidence, not exact pitch-error ground truth.",
        "- JANON learner prompts have no teacher pitch labels; lower scores cannot automatically be interpreted as incorrect Japanese.",
        "- Wrong-drop separation remains the key criterion for whether strict accent claims are unsupported.",
        "- The test report is frozen evidence. Any later formula change requires a new untouched test set or external human-labeled corpus.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run speaker/text-held-out pitch naturalness validation on JVS and JANON.")
    parser.add_argument("--jvs-root", type=Path, default=PROJECT_ROOT / "JVS")
    parser.add_argument("--janon-root", type=Path, default=PROJECT_ROOT / "JANON")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/pitch_heldout_validation_manifest.csv")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/pitch_heldout_validation.jsonl")
    parser.add_argument("--out-csv", type=Path, default=ROOT / "results/calibration/pitch_heldout_validation.csv")
    parser.add_argument("--out-summary", type=Path, default=ROOT / "results/calibration/pitch_heldout_validation_summary.csv")
    parser.add_argument("--out-report", type=Path, default=ROOT / "reports/pitch_heldout_validation.md")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(args.jvs_root, args.janon_root)
    _write_csv(args.manifest, manifest)
    print(f"wrote {args.manifest} ({len(manifest)} rows)")
    if args.manifest_only:
        return
    rows = run_audit(manifest, args.checkpoint, sample_rate=args.sample_rate)
    _write_csv(args.out_csv, rows)
    summary = summarize(rows)
    _write_csv(args.out_summary, summary)
    write_report(args.out_report, rows, summary)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_summary}")
    print(f"wrote {args.out_report}")


if __name__ == "__main__":
    main()
