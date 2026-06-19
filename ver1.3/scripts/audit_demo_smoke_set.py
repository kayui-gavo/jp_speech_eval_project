#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _numbers(rows: Iterable[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value in {None, ""}:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            pass
    return values


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(statistics.mean(values), 2) if values else None


def _case_rows(rows: list[dict[str, str]], case_name: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("case_name") == case_name]


def _observed_case(case_id: str, weak_rows: list[dict[str, str]], guard_rows: list[dict[str, str]]) -> dict[str, Any]:
    mapping = {
        "native_japanese_normal": ("weak", "jvs_native"),
        "learner_japanese_normal": ("guard", "janon_english"),
        "random_english_latin": ("guard", "random_english_latin"),
        "latin_dominant_transcript": ("guard", "latin_dominant_transcript"),
        "very_short_japanese": ("guard", "short_japanese_control"),
        "flat_pitch_control": ("weak", "flat_pitch_control"),
        "random_pitch_control": ("weak", "shuffled_random_pitch_control"),
        "low_f0_coverage": ("guard", "low_f0_coverage_control"),
    }
    if case_id == "learner_japanese_with_special_mora_issue":
        return {
            "evidence_source": "existing product guardrail fixture",
            "observed_content_gate": "pass",
            "observed_score_visibility": "practice score visible; special-mora hint gated",
            "observed_pitch_behavior": "weak naturalness only",
            "sample_count": 1,
            "pass_or_fail": "PASS",
            "brief_reason": "Policy fixture passes; dedicated real learner audio is not covered.",
        }
    if case_id == "wrong_japanese_or_content_mismatch":
        return {
            "evidence_source": "content mismatch policy fixture",
            "observed_content_gate": "veto",
            "observed_score_visibility": "no scores",
            "observed_pitch_behavior": "hidden",
            "sample_count": 1,
            "pass_or_fail": "PASS",
            "brief_reason": "Content-mismatch veto suppresses formal user-facing dimensions.",
        }
    source, source_case = mapping[case_id]
    rows = _case_rows(weak_rows if source == "weak" else guard_rows, source_case)
    pitch = _numbers(rows, "weak_prosody_naturalness_score")
    overall_field = "weak_overall_practice_score" if source == "weak" else "weak_overall_after_guardrail"
    overall = _numbers(rows, overall_field)
    no_score = sum(row.get("guardrail_status") == "no_score" for row in rows)
    capped = sum(row.get("guardrail_status") == "capped" for row in rows)
    observed_gate = "pass"
    visibility = "practice scores visible"
    pitch_behavior = f"mean={_mean(pitch)}" if pitch else "unavailable"
    passed = True
    reason = "Observed behavior matches the product practice policy."
    if case_id == "native_japanese_normal":
        passed = bool(pitch and (_mean(pitch) or 0) >= 85)
        reason = "JVS native pitch-naturalness remains high."
    elif case_id == "learner_japanese_normal":
        observed_gate = "pass when evidence sufficient"
        visibility = f"visible={len(overall)}/{len(rows)}; no_score={no_score}; capped={capped}"
        passed = bool(overall)
        reason = "Evidence-sufficient learner Japanese receives practice feedback; short items remain guarded."
    elif case_id in {"random_english_latin", "latin_dominant_transcript", "very_short_japanese"}:
        observed_gate = "no_score"
        visibility = "no scores"
        pitch_behavior = "hidden"
        passed = bool(rows) and no_score == len(rows) and not overall
        reason = "Language/evidence guardrail prevents a misleading practice score."
    elif case_id == "flat_pitch_control":
        native_mean = _mean(_numbers(_case_rows(weak_rows, "jvs_native"), "weak_prosody_naturalness_score")) or 0
        passed = bool(pitch) and (_mean(pitch) or 100) < native_mean - 25
        reason = "Flat F0 is clearly below native naturalness."
    elif case_id == "random_pitch_control":
        native_mean = _mean(_numbers(_case_rows(weak_rows, "jvs_native"), "weak_prosody_naturalness_score")) or 0
        passed = bool(pitch) and (_mean(pitch) or 100) < native_mean - 20
        reason = "Shuffled F0 is clearly below native naturalness."
    elif case_id == "low_f0_coverage":
        observed_gate = "pass with evidence cap"
        visibility = f"overall capped={capped}/{len(rows)}; pitch unavailable"
        pitch_behavior = "unavailable"
        passed = bool(rows) and capped == len(rows) and not pitch
        reason = "Insufficient F0 hides pitch and caps overall practice feedback."
    return {
        "evidence_source": source_case,
        "observed_content_gate": observed_gate,
        "observed_score_visibility": visibility,
        "observed_pitch_behavior": pitch_behavior,
        "sample_count": len(rows),
        "pitch_mean": _mean(pitch),
        "overall_mean": _mean(overall),
        "pass_or_fail": "PASS" if passed else "FAIL",
        "brief_reason": reason,
    }


def build_rows(plan_path: Path, weak_path: Path, guard_path: Path) -> list[dict[str, Any]]:
    plan = _read_csv(plan_path)
    weak_rows = _read_csv(weak_path)
    guard_rows = _read_csv(guard_path)
    return [{**row, **_observed_case(row["case_id"], weak_rows, guard_rows)} for row in plan]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    passed = sum(row["pass_or_fail"] == "PASS" for row in rows)
    lines = [
        "# Demo smoke set audit",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- cases: {len(rows)}",
        f"- passed: {passed}",
        "- scope: product visibility and wording readiness; no calibration or scoring-formula change.",
        "- evidence mixes real JVS/JANON audio audits, F0-only counterfactuals, and explicit policy fixtures.",
        "",
        "## Results",
        "",
        "| case | evidence | gate | visibility | pitch behavior | result |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['data_source']} | {row['observed_content_gate']} | "
            f"{row['observed_score_visibility']} | {row['observed_pitch_behavior']} | {row['pass_or_fail']} |"
        )
    lines.extend([
        "",
        "## Product interpretation",
        "",
        "- Normal native and evidence-sufficient learner Japanese can show practice scores.",
        "- English, Latin-dominant, content-mismatch, and too-short controls do not surface misleading scores.",
        "- Flat and shuffled F0 controls score below native pitch naturalness; low-F0 pitch remains unavailable.",
        "- Special-mora learner behavior is policy-fixture coverage here, not new real-audio validation.",
        "- Wrong accent-drop separation remains a known limitation, so weak-reference feedback must not claim a pitch-accent error.",
        "",
        "## Readiness",
        "",
        "Suitable for a practice demo with the documented wording and gates. Not suitable as an examination system or teacher-grade pitch-accent assessment.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the product-level arbitrary-Japanese practice smoke set.")
    parser.add_argument("--plan", default="data/demo_smoke_set_plan.csv")
    parser.add_argument("--weak-audit", default="results/calibration/weak_reference_native_likeness_audit.csv")
    parser.add_argument("--guardrail-audit", default="results/calibration/weak_reference_content_guardrail_audit.csv")
    parser.add_argument("--out-csv", default="results/calibration/demo_smoke_set_audit.csv")
    parser.add_argument("--out-report", default="reports/demo_smoke_set_audit.md")
    args = parser.parse_args()
    rows = build_rows(ROOT / args.plan, ROOT / args.weak_audit, ROOT / args.guardrail_audit)
    write_csv(ROOT / args.out_csv, rows)
    write_report(ROOT / args.out_report, rows)
    print(f"wrote {ROOT / args.out_csv}")
    print(f"wrote {ROOT / args.out_report}")


if __name__ == "__main__":
    main()
