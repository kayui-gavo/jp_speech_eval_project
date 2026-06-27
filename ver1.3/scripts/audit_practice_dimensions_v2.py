#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.alignment import estimate_mora_boundaries_equal  # noqa: E402
from jp_speech_eval.alignment_evidence.phone_mora_mapper import map_phones_to_moras  # noqa: E402
from jp_speech_eval.alignment_evidence.textgrid_parser import parse_lab_phone_segments  # noqa: E402
from jp_speech_eval.audio_features import load_audio  # noqa: E402
from jp_speech_eval.practice_dimensions import (  # noqa: E402
    score_pronunciation_clarity_practice,
    score_rhythm_timing_practice,
    weighted_four_dimension_overall,
)
from jp_speech_eval.recording_quality import assess_recording_quality  # noqa: E402
from jp_speech_eval.scoring import score_fluency  # noqa: E402
from jp_speech_eval.text_frontend import build_text_info  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def _lab_pause_info(path: Path) -> tuple[float, dict[str, Any]]:
    segments: list[tuple[float, float, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            segments.append((float(parts[0]), float(parts[1]), str(parts[2])))
        except ValueError:
            continue
    speech = [item for item in segments if item[2] not in {"sil", "sp", "<eps>"}]
    if not speech:
        return 0.0, {"pause_count": 0, "pause_total": 0.0, "pause_ratio": 0.0}
    start, end = speech[0][0], speech[-1][1]
    duration = max(0.0, end - start)
    pauses = [(s, e) for s, e, phone in segments if phone == "pau" and s >= start and e <= end and e - s >= 0.30]
    pause_total = sum(e - s for s, e in pauses)
    return duration, {
        "pause_count": len(pauses),
        "pause_total": pause_total,
        "pause_ratio": pause_total / max(duration, 1e-6),
    }


def _boundaries_from_durations(durations: np.ndarray) -> list[tuple[float, float]]:
    cursor = 0.0
    rows = []
    for duration in durations:
        rows.append((cursor, cursor + float(duration)))
        cursor += float(duration)
    return rows


def _timing_jitter(boundaries: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    durations = np.asarray([end - start for start, end in boundaries], dtype=float)
    factors = np.asarray([0.35 if index % 3 == 0 else 1.75 if index % 3 == 1 else 0.90 for index in range(len(durations))])
    changed = np.maximum(0.01, durations * factors)
    changed *= float(np.sum(durations)) / max(float(np.sum(changed)), 1e-8)
    return _boundaries_from_durations(changed)


def _paired_auc(rows: Sequence[Mapping[str, Any]], normal: str, control: str) -> float:
    pairs = [(float(row[normal]), float(row[control])) for row in rows if row.get(normal) is not None and row.get(control) is not None]
    positive = np.asarray([item[0] for item in pairs], dtype=float)
    negative = np.asarray([item[1] for item in pairs], dtype=float)
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    scores = np.concatenate([positive, negative])
    return float(roc_auc_score(labels, scores))


def _summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "n": len(array),
        "mean": round(float(np.mean(array)), 4),
        "p10": round(float(np.percentile(array, 10)), 4),
        "p50": round(float(np.percentile(array, 50)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "floor_rate": round(float(np.mean(array <= 0)), 4),
        "ceiling_rate": round(float(np.mean(array >= 100)), 4),
    }


def run() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = ROOT / "outputs/pitch_naturalness_v2_audit.jsonl"
    manifest = [
        row for row in _read_jsonl(source)
        if row.get("group") == "fresh_jvs_parallel100" and row.get("condition") == "normal"
    ]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260627)
    for index, item in enumerate(manifest):
        info = build_text_info(str(item["reference_text"]))
        phones = parse_lab_phone_segments(item["lab_path"])
        segments, mapping = map_phones_to_moras(phones, info.moras)
        if len(segments) != len(info.moras):
            continue
        boundaries = [(float(segment.start), float(segment.end)) for segment in segments]
        duration, pause_info = _lab_pause_info(Path(item["lab_path"]))
        fluency, _, fluency_details = score_fluency(len(info.moras), duration, pause_info)
        slow, _, _ = score_fluency(len(info.moras), duration * 1.8, pause_info)
        fast, _, _ = score_fluency(len(info.moras), duration * 0.55, pause_info)
        added_pause = max(1.5, duration * 0.30)
        hesitant_pause = {
            "pause_count": int(pause_info["pause_count"]) + 5,
            "pause_total": float(pause_info["pause_total"]) + added_pause,
            "pause_ratio": (float(pause_info["pause_total"]) + added_pause) / max(duration + added_pause, 1e-6),
        }
        hesitant, _, _ = score_fluency(len(info.moras), duration + added_pause, hesitant_pause)
        rhythm, rhythm_details = score_rhythm_timing_practice(
            boundaries=boundaries,
            alignment_mode="mfa_phone_lab",
            rate_score=fluency_details.get("rate_score"),
        )
        rhythm_jitter, _ = score_rhythm_timing_practice(
            boundaries=_timing_jitter(boundaries),
            alignment_mode="mfa_phone_lab",
            rate_score=fluency_details.get("rate_score"),
        )
        equal = estimate_mora_boundaries_equal(sum(end - start for start, end in boundaries), len(boundaries))
        rhythm_equal, _ = score_rhythm_timing_practice(
            boundaries=equal,
            alignment_mode="cached_dtw_fallback_equal",
            rate_score=fluency_details.get("rate_score"),
        )

        audio = load_audio(item["audio_path"], sr=16000)
        _, speech_region = trim_to_speech(audio.y, audio.sr)
        clean_quality = assess_recording_quality(audio.y, audio.sr, speech_region)
        evidence = {
            "mora_count": len(info.moras),
            "judgement_available_count": len(info.moras),
            "mean_energy_coverage": 0.95,
        }
        pronunciation, _ = score_pronunciation_clarity_practice(
            recording_quality=clean_quality,
            mora_evidence_summary=evidence,
            alignment_mode="mfa_phone_lab",
            content_match={"status": "pass", "kana_similarity": 1.0},
        )
        pronunciation_noisy = None
        pronunciation_low_gain = None
        if index < 60:
            signal_rms = float(np.sqrt(np.mean(audio.y * audio.y)))
            noise_rms = signal_rms / (10.0 ** (5.0 / 20.0))
            noisy = np.clip(audio.y + rng.normal(0.0, noise_rms, size=audio.y.shape), -1.0, 1.0)
            noisy_quality = assess_recording_quality(noisy, audio.sr, speech_region)
            pronunciation_noisy, _ = score_pronunciation_clarity_practice(
                recording_quality=noisy_quality,
                mora_evidence_summary={**evidence, "mean_energy_coverage": 0.55},
                alignment_mode="mfa_phone_lab",
                content_match={"status": "pass", "kana_similarity": 1.0},
            )
            low_quality = assess_recording_quality(audio.y * 0.08, audio.sr, speech_region)
            pronunciation_low_gain, _ = score_pronunciation_clarity_practice(
                recording_quality=low_quality,
                mora_evidence_summary={**evidence, "mean_energy_coverage": 0.60},
                alignment_mode="mfa_phone_lab",
                content_match={"status": "pass", "kana_similarity": 1.0},
            )

        overall, _ = weighted_four_dimension_overall(
            pronunciation=pronunciation,
            rhythm=rhythm,
            fluency=fluency,
            pitch=float(item["v2_score"]),
        )
        rows.append({
            "sample_id": item["sample_id"],
            "speaker_id": item["speaker_id"],
            "mora_count": len(info.moras),
            "mapping_success": bool(mapping.get("mapping_success")),
            "pronunciation_clarity_normal": pronunciation,
            "pronunciation_clarity_5db_noise": pronunciation_noisy,
            "pronunciation_clarity_low_gain": pronunciation_low_gain,
            "rhythm_normal": rhythm,
            "rhythm_timing_jitter": rhythm_jitter,
            "rhythm_equal_fallback": rhythm_equal,
            "rhythm_log_duration_mad": (rhythm_details.get("components") or {}).get("log_duration_mad"),
            "fluency_normal": fluency,
            "fluency_slow": slow,
            "fluency_fast": fast,
            "fluency_hesitation": hesitant,
            "pitch_normal": item["v2_score"],
            "overall_four_dimension": overall,
        })

    metric_fields = [
        "pronunciation_clarity_normal",
        "pronunciation_clarity_5db_noise",
        "pronunciation_clarity_low_gain",
        "rhythm_normal",
        "rhythm_timing_jitter",
        "rhythm_equal_fallback",
        "fluency_normal",
        "fluency_slow",
        "fluency_fast",
        "fluency_hesitation",
        "pitch_normal",
        "overall_four_dimension",
    ]
    summary_rows = []
    for field in metric_fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        summary_rows.append({"metric": field, **_summary(values)})
    janon_path = ROOT / "outputs/three_dimensions_janon.jsonl"
    if janon_path.exists():
        janon = _read_jsonl(janon_path)
        for group in ("janon_native", "janon_learner"):
            scores: list[float] = []
            for item in janon:
                if item.get("group") != group:
                    continue
                score, _, _ = score_fluency(
                    int(item.get("mora_count") or 0),
                    float(item.get("duration_sec") or 0.0),
                    {
                        "pause_count": int(item.get("pause_count") or 0),
                        "pause_ratio": float(item.get("pause_ratio") or 0.0),
                    },
                )
                scores.append(float(score))
            summary_rows.append({"metric": f"{group}_fluency_external", **_summary(scores)})
    for dimension, normal, control in (
        ("pronunciation", "pronunciation_clarity_normal", "pronunciation_clarity_5db_noise"),
        ("pronunciation", "pronunciation_clarity_normal", "pronunciation_clarity_low_gain"),
        ("rhythm", "rhythm_normal", "rhythm_timing_jitter"),
        ("fluency", "fluency_normal", "fluency_slow"),
        ("fluency", "fluency_normal", "fluency_fast"),
        ("fluency", "fluency_normal", "fluency_hesitation"),
    ):
        pairs = [(float(row[normal]), float(row[control])) for row in rows if row.get(normal) is not None and row.get(control) is not None]
        summary_rows.append({
            "metric": f"paired_{normal}_vs_{control}",
            "dimension": dimension,
            "n": len(pairs),
            "mean": round(float(np.mean([a - b for a, b in pairs])), 4),
            "p10": "",
            "p50": round(float(np.median([a - b for a, b in pairs])), 4),
            "p90": "",
            "floor_rate": "",
            "ceiling_rate": "",
            "roc_auc": round(_paired_auc(rows, normal, control), 4),
        })
    return rows, summary_rows


def write_report(rows: Sequence[Mapping[str, Any]], summary: Sequence[Mapping[str, Any]]) -> None:
    lookup = {str(row["metric"]): row for row in summary}
    d = lambda key: lookup[key]
    lines = [
        "# Practice dimensions v2 engineering sanity validation",
        "",
        f"- JVS native rows: {len(rows)} (speaker-disjoint fresh parallel100 audit set)",
        f"- JANON sentence rows: {int(d('janon_native_fluency_external')['n']) + int(d('janon_learner_fluency_external')['n'])} (external fluency distribution only)",
        "- Human ratings: none",
        "- Purpose: validate implementation behavior and controlled negative-control separation, not teacher-grade or phoneme-error accuracy",
        "",
        "## Distribution",
        "",
        "| metric | n | mean | p10 | p50 | p90 | ceiling rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in (
        "pronunciation_clarity_normal",
        "pronunciation_clarity_5db_noise",
        "pronunciation_clarity_low_gain",
        "rhythm_normal",
        "rhythm_timing_jitter",
        "rhythm_equal_fallback",
        "fluency_normal",
        "fluency_slow",
        "fluency_fast",
        "fluency_hesitation",
        "pitch_normal",
        "overall_four_dimension",
        "janon_native_fluency_external",
        "janon_learner_fluency_external",
    ):
        row = d(key)
        lines.append(f"| {key} | {row['n']} | {row['mean']} | {row['p10']} | {row['p50']} | {row['p90']} | {row['ceiling_rate']} |")
    lines.extend([
        "",
        "## Paired separation",
        "",
        "| comparison | mean normal-control delta | AUC |",
        "|---|---:|---:|",
    ])
    for key in (
        "paired_pronunciation_clarity_normal_vs_pronunciation_clarity_5db_noise",
        "paired_pronunciation_clarity_normal_vs_pronunciation_clarity_low_gain",
        "paired_rhythm_normal_vs_rhythm_timing_jitter",
        "paired_fluency_normal_vs_fluency_slow",
        "paired_fluency_normal_vs_fluency_fast",
        "paired_fluency_normal_vs_fluency_hesitation",
    ):
        row = d(key)
        lines.append(f"| {key.removeprefix('paired_')} | {row['mean']} | {row['roc_auc']} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Pronunciation clarity no longer uses global mora-duration CV. It reacts to recording/evidence degradation but still cannot detect a cleanly recorded phone substitution.",
        "- The 5 dB noise and low-gain conditions are synthetic channel controls. Their AUC values show deterministic response to those controls, not real pronunciation-error validity.",
        "- Rhythm uses robust log-duration dispersion with reliable phone-label timing. Equal fallback is a low-confidence neutral estimate, not perfect rhythm.",
        "- Fluency uses a continuous rate curve and excess-pause penalty, removing the previous broad 100-point plateau while allowing natural phrase pauses.",
        "- The practice overall is computed from the same four dimensions shown in the UI. There is no hidden high-score floor.",
        "- Special-mora user feedback is safe by default. Long-vowel/nasal candidate feedback requires explicit opt-in and reliable non-fallback boundaries; sokuon and yoon remain blocked.",
        "",
        "## Remaining scientific limits",
        "",
        "- Pronunciation clarity is an intelligibility/judgeability proxy, not phone-level GOP or phoneme correctness.",
        "- JVS negative controls are synthetic paired perturbations; they validate sensitivity, not educational validity.",
        "- JANON has no teacher score and is not used as a bad-pronunciation label.",
        "- Special-mora learner feedback still needs reliable phone boundaries and real learner error labels.",
        "- Pitch remains broad naturalness with a soft accent hint, not strict pitch-accent correctness.",
    ])
    (ROOT / "reports/practice_dimensions_v2_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows, summary = run()
    _write_csv(ROOT / "results/calibration/practice_dimensions_v2_validation.csv", rows)
    _write_csv(ROOT / "data/calibration_candidates/practice_dimensions_v2_summary.csv", summary)
    write_report(rows, summary)
    print({"rows": len(rows), "summary_rows": len(summary)})


if __name__ == "__main__":
    main()
