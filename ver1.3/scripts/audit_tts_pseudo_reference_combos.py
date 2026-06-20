#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(statistics.mean(values), 4) if values else None


def _metric_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "reference_audio_readable": any(row.get("audio_readable") == "True" for row in rows),
        "duration_sec": _mean(float(row["duration_sec"]) for row in rows if row.get("duration_sec")),
        "f0_coverage": _mean(float(row["f0_coverage"]) for row in rows if row.get("f0_coverage")),
        "mora_timing_quality": ";".join(sorted({row["mora_timing_quality"] for row in rows if row.get("mora_timing_quality")})) or None,
        "pitch_range_log": _mean(float(row["pitch_range_log"]) for row in rows if row.get("pitch_range_log")),
        "local_pitch_movement": _mean(float(row["local_pitch_movement"]) for row in rows if row.get("local_pitch_movement")),
        "final_intonation_score": _mean(float(row["final_intonation_score"]) for row in rows if row.get("final_intonation_score")),
    }


def audit_rows(plan_path: Path, tts_quality_path: Path, jvs_path: Path) -> list[dict[str, Any]]:
    quality = _read(tts_quality_path)
    jvs = _read(jvs_path)
    current_quality = [row for row in quality if row["provider_slot"] == "current_pyopenjtalk" and row["fixture_available"] == "True"]
    native_scores = [float(row["prosody_score"]) for row in jvs if row["case_name"] == "native_cross_speaker_reference_audio_f0_cache"]
    native_rows = [row for row in jvs if row["case_name"] == "native_cross_speaker_reference_audio_f0_cache"]
    openjtalk_proxy = [float(row["prosody_score"]) for row in jvs if row["case_name"] == "native_cross_speaker_openjtalk_target"]

    rows: list[dict[str, Any]] = []
    for combo in _read(plan_path):
        combo_id = combo["combo_id"]
        metrics: dict[str, Any] = {
            "reference_audio_readable": None,
            "duration_sec": None,
            "f0_coverage": None,
            "mora_timing_quality": None,
            "pitch_range_log": None,
            "local_pitch_movement": None,
            "final_intonation_score": None,
            "native_against_reference_prosody_score": None,
            "openjtalk_symbolic_target_proxy_score": None,
            "score_evidence_status": "not_measured",
        }
        if combo_id == "current_tts_pseudo_reference":
            metrics.update(_metric_summary(current_quality))
            metrics["openjtalk_symbolic_target_proxy_score"] = _mean(openjtalk_proxy)
            metrics["score_evidence_status"] = "audio_metrics_only_no_same_sentence_native_vs_tts_f0_score"
        elif combo_id == "verified_native_reference_oracle":
            metrics["reference_audio_readable"] = True
            metrics["f0_coverage"] = _mean(float(row["f0_coverage"]) for row in native_rows if row.get("f0_coverage"))
            metrics["mora_timing_quality"] = "lab_phone_mora"
            metrics["final_intonation_score"] = _mean(
                float(row["final_intonation_score"]) for row in native_rows if row.get("final_intonation_score")
            )
            metrics["native_against_reference_prosody_score"] = _mean(native_scores)
            metrics["score_evidence_status"] = "test_only_jvs_cross_speaker_oracle"
        rows.append({
            "benchmark_mode": "fixed_reference_pseudo_reference",
            "combo_id": combo_id,
            "provider_name": combo["provider_name"],
            "model_name": combo["model_name"],
            "provenance": combo["provenance"],
            "reference_kind": combo["reference_kind"],
            "fixture_status": combo["fixture_status"],
            "asr_setting": combo["asr_setting"],
            "tts_is_primary_scoring_variable": True,
            "pitch_target_reliability": "reliable" if combo["provenance"] == "verified_native" else "weak",
            "strong_pitch_reference_allowed": combo["provenance"] == "verified_native",
            "external_provider_called": False,
            "api_key_required": False,
            **metrics,
            "notes": combo["notes"],
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    by_id = {row["combo_id"]: row for row in rows}
    current = by_id["current_tts_pseudo_reference"]
    oracle = by_id["verified_native_reference_oracle"]
    lines = [
        "# TTS pseudo-reference combo audit",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- mode: fixed-reference pseudo-reference",
        "- external provider calls: 0",
        "- API keys required: no",
        "- ASR setting: fixed/current or oracle transcript; ASR is not the comparison variable.",
        "",
        "## Results",
        "",
        "| combo | fixture | readable | duration | F0 coverage | timing | range | movement | final | native score | reliability | evidence |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['combo_id']} | {row['fixture_status']} | {row['reference_audio_readable']} | "
            f"{row['duration_sec']} | {row['f0_coverage']} | {row['mora_timing_quality']} | "
            f"{row['pitch_range_log']} | {row['local_pitch_movement']} | {row['final_intonation_score']} | "
            f"{row['native_against_reference_prosody_score']} | {row['pitch_target_reliability']} | {row['score_evidence_status']} |"
        )
    lines.extend([
        "",
        "## Evidence boundary",
        "",
        f"- Current packaged pyopenjtalk fixture metrics are available for one sentence: duration {current['duration_sec']} s, F0 coverage {current['f0_coverage']}, timing `{current['mora_timing_quality']}`.",
        f"- The existing OpenJTalk symbolic-target JVS proxy mean is {current['openjtalk_symbolic_target_proxy_score']}. It is not a TTS-audio-F0 pseudo-reference score and is not used as proof that current TTS reaches that ceiling.",
        f"- The test-only verified native cross-speaker oracle mean is {oracle['native_against_reference_prosody_score']}. It is an upper-bound proxy, not packaged-demo evidence.",
        "- No same-sentence best-candidate TTS WAV is present. Therefore better TTS versus current TTS, and either TTS versus the verified oracle, cannot yet be quantified.",
        "",
        "## Answers",
        "",
        "- Current TTS is a plausible strict/reference-based ceiling bottleneck because its provenance is weak and its mora timing is approximate. The present files do not prove the size of that bottleneck.",
        "- Better TTS is not demonstrated to be materially better in this offline audit; the candidate fixture is missing.",
        "- No synthetic TTS can become a reliable human/native pitch baseline solely by improving audio quality. It may improve imitation UX, timing/alignment behavior, and possibly pseudo-reference score consistency.",
        "- A decisive A/B needs the same texts synthesized by current and candidate TTS, the same native/user recordings scored against each, plus listening review. Verified human/native reference remains the strict baseline.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TTS pseudo-reference combinations without provider calls.")
    parser.add_argument("--plan", default="data/tts_pseudo_reference_combo_plan.csv")
    parser.add_argument("--tts-quality", default="results/calibration/tts_reference_quality_audit.csv")
    parser.add_argument("--jvs", default="results/calibration/jvs_verified_pitch_demo_summary.csv")
    parser.add_argument("--out-csv", default="results/calibration/tts_pseudo_reference_combo_audit.csv")
    parser.add_argument("--out-report", default="reports/tts_pseudo_reference_combo_audit.md")
    args = parser.parse_args()
    rows = audit_rows(ROOT / args.plan, ROOT / args.tts_quality, ROOT / args.jvs)
    _write_csv(ROOT / args.out_csv, rows)
    _write_report(ROOT / args.out_report, rows)
    print(f"wrote {ROOT / args.out_csv}")
    print(f"wrote {ROOT / args.out_report}")


if __name__ == "__main__":
    main()
