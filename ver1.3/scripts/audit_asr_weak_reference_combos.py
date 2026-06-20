#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(statistics.mean(values), 4) if values else None


def audit_rows(plan_path: Path, source_path: Path) -> list[dict[str, Any]]:
    source = defaultdict(list)
    for row in _read(source_path):
        source[row["provider_slot"]].append(row)

    rows: list[dict[str, Any]] = []
    for combo in _read(plan_path):
        for item in source[combo["source_provider_slot"]]:
            row = dict(item)
            row.update({
                "benchmark_mode": "asr_confirmed_weak_reference",
                "combo_id": combo["combo_id"],
                "provider_name": combo["provider_name"],
                "model_name": combo["model_name"],
                "provider_provenance": combo["provenance"],
                "tts_setting": combo["tts_setting"],
                "tts_is_primary_scoring_variable": False,
                "external_provider_called": False,
                "api_key_required": False,
                "combo_notes": combo["notes"],
            })
            rows.append(row)
    return rows


def combo_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["combo_id"])].append(row)
    out = []
    for combo_id, items in grouped.items():
        good = [row for row in items if row["input_class"] == "good_japanese"]
        bad = [row for row in items if str(row["input_class"]).startswith("bad_")]
        expected_no_score = [
            row for row in items
            if row["input_class"] != "good_japanese" or row["input_class"] == "low_evidence_japanese"
        ]
        out.append({
            "combo_id": combo_id,
            "n": len(items),
            "normalized_match_rate": _mean(float(_bool(row["transcript_exact_normalized"])) for row in good),
            "japanese_accept_rate": _mean(float(row["content_gate_status"] == "pass") for row in good),
            "bad_input_reject_rate": _mean(float(row["content_gate_status"] != "pass") for row in bad),
            "mean_good_kana_similarity": _mean(float(row["kana_similarity"]) for row in good if row["transcript"]),
            "hallucinated_japanese_risk_rate": _mean(float(_bool(row["hallucination_risk"])) for row in bad),
            "weak_score_visible_now_rate": _mean(float(_bool(row["weak_score_visibility_now"])) for row in items),
            "weak_score_visible_if_confirmed_good_rate": _mean(
                float(_bool(row["weak_score_visibility_if_user_confirms_unchanged"])) for row in good
            ),
            "no_score_correct_rate": _mean(float(_bool(row["no_score_correct"])) for row in expected_no_score),
            "timestamp_useful_rate": _mean(float(_bool(row["word_timestamps_available"])) for row in good),
            "feedback_location_useful_rate": _mean(float(_bool(row["feedback_location_may_improve"])) for row in good),
            "false_reject_count": sum(_bool(row["false_reject_good_japanese"]) for row in items),
            "false_accept_if_confirmed_count": sum(_bool(row["false_accept_bad_input_if_confirmed"]) for row in items),
        })
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    summary = combo_summary(rows)
    by_id = {row["combo_id"]: row for row in summary}
    current = by_id["current_asr"]
    best = by_id["best_candidate_asr"]
    oracle = by_id["oracle_transcript"]
    lines = [
        "# ASR weak-reference combo audit",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- rows: {len(rows)}",
        "- mode: `asr_confirmed_weak_reference`",
        "- external provider calls: 0",
        "- API keys required: no",
        "- evidence class: offline plan replay; candidate transcripts are fixtures, not empirical provider outputs.",
        "- TTS setting: fixed current demo pseudo-reference; TTS is not the comparison variable.",
        "",
        "## Results",
        "",
        "| combo | normalized match | Japanese accept | bad reject | kana similarity | hallucination risk | good visible if confirmed | no-score correctness | timestamps useful | feedback location useful |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['combo_id']} | {item['normalized_match_rate']} | {item['japanese_accept_rate']} | "
            f"{item['bad_input_reject_rate']} | {item['mean_good_kana_similarity']} | "
            f"{item['hallucinated_japanese_risk_rate']} | {item['weak_score_visible_if_confirmed_good_rate']} | "
            f"{item['no_score_correct_rate']} | "
            f"{item['timestamp_useful_rate']} | {item['feedback_location_useful_rate']} |"
        )
    lines.extend([
        "",
        "## Gaps",
        "",
        f"- Best-candidate minus current Japanese accept: {best['japanese_accept_rate'] - current['japanese_accept_rate']:.4f}.",
        f"- Best-candidate minus current normalized match: {best['normalized_match_rate'] - current['normalized_match_rate']:.4f}.",
        f"- Oracle minus best Japanese accept: {oracle['japanese_accept_rate'] - best['japanese_accept_rate']:.4f}.",
        f"- Oracle minus best bad-input reject: {oracle['bad_input_reject_rate'] - best['bad_input_reject_rate']:.4f}.",
        "",
        "## Answers",
        "",
        "- The best-candidate slot looks better than current ASR on good-Japanese eligibility and provides timestamp anchors. Statistical significance cannot be claimed: the fixture candidates are deliberately near-oracle and no real provider processed the audio.",
        "- Oracle still leaves a bad-input edge case in this synthetic plan, showing that transcript correctness alone does not replace language/content evidence and user confirmation.",
        "- ASR is a primary ceiling for arbitrary-speech eligibility, kana/mora stability, and feedback localization. It is not the only product ceiling: F0 extraction, timing proxies, learner calibration, and pedagogy remain downstream limitations.",
        "- TTS is not wholly absent from the current implementation: the fixed pseudo-reference cache can indirectly influence alignment/timing proxies. It is held constant here and is not a strict pitch judge in weak native-likeness scoring.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ASR combinations for weak-reference practice mode.")
    parser.add_argument("--plan", default="data/asr_weak_reference_combo_plan.csv")
    parser.add_argument("--source", default="results/calibration/asr_bottleneck_audit.csv")
    parser.add_argument("--out-csv", default="results/calibration/asr_weak_reference_combo_audit.csv")
    parser.add_argument("--out-report", default="reports/asr_weak_reference_combo_audit.md")
    args = parser.parse_args()
    rows = audit_rows(ROOT / args.plan, ROOT / args.source)
    _write_csv(ROOT / args.out_csv, rows)
    _write_report(ROOT / args.out_report, rows)
    print(f"wrote {ROOT / args.out_csv}")
    print(f"wrote {ROOT / args.out_report}")


if __name__ == "__main__":
    main()
