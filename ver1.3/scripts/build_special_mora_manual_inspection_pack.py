#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent


def _resolve_jvs_audio(row: Mapping[str, Any]) -> str:
    """Best-effort local JVS audio lookup for review playback.

    The audit CSVs often keep only speaker_id/utterance_id, so the manual
    inspection pack reconstructs the usual JVS parallel100 path when possible.
    """

    audio_path = str(row.get("audio_path") or "")
    if audio_path and Path(audio_path).exists():
        return audio_path
    speaker_id = str(row.get("speaker_id") or "")
    utterance_id = str(row.get("utterance_id") or "")
    dataset = str(row.get("dataset") or "").lower()
    if dataset == "jvs" and speaker_id and utterance_id:
        candidate = PROJECT_ROOT / "JVS" / speaker_id / "parallel100" / "wav24kHz16bit" / f"{utterance_id}.wav"
        if candidate.exists():
            return str(candidate)
    return audio_path


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _item(source: str, row: Mapping[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "item_id": f"{source}_{idx:03d}",
        "source": source,
        "dataset": row.get("dataset"),
        "speaker_id": row.get("speaker_id"),
        "utterance_id": row.get("utterance_id"),
        "audio_path": _resolve_jvs_audio(row),
        "transcript": row.get("transcript") or row.get("text"),
        "special_mora_type": row.get("special_mora_type"),
        "surface_mora": row.get("surface_mora") or row.get("mora"),
        "decision": row.get("decision") or row.get("user_decision") or row.get("debug_decision"),
        "feature_value": row.get("feature_value") or row.get("ratio_to_avg_mora") or row.get("ratio_to_avg"),
        "threshold_user_low": row.get("user_low") or row.get("threshold_user_low"),
        "threshold_user_high": row.get("user_high") or row.get("threshold_user_high"),
        "near_boundary": row.get("near_boundary"),
        "evidence_confidence": row.get("evidence_confidence"),
        "phone_sequence_for_mora": row.get("phone_sequence_for_mora"),
        "suggested_human_check": "seems_valid_feedback? | false_alarm? | unclear_alignment? | audio_quality_issue? | comment",
    }


def run(args: argparse.Namespace) -> List[Dict[str, Any]]:
    false_cases = _read_csv(args.false_alarm_cases)
    v2_decisions = _read_csv(args.v2_jvs_decisions)
    janon = _read_csv(args.janon_v2)
    counter = _read_csv(args.counterfactual)
    sample = _read_csv(args.sample_audit)
    items: List[Dict[str, Any]] = []
    for row in false_cases:
        items.append(_item("JVS_false_alarm", row, len(items) + 1))
    for row in [r for r in v2_decisions if str(r.get("user_feedback_allowed")).lower() == "true"][:20]:
        items.append(_item("JVS_allowed_candidate", row, len(items) + 1))
    for row in [r for r in janon if str(r.get("user_decision")) in {"too_short", "too_long"} or str(r.get("decision")) in {"too_short", "too_long"}][:20]:
        items.append(_item("JANON_outlier", row, len(items) + 1))
    for row in [r for r in counter if str(r.get("too_short")).lower() == "true"][:20]:
        items.append(_item("counterfactual_positive", row, len(items) + 1))
    for row in [r for r in v2_decisions if str(r.get("near_boundary")).lower() == "true"][:20]:
        items.append(_item("near_boundary", row, len(items) + 1))
    if not items:
        for row in sample[:10]:
            items.append(_item("sample_audit", row, len(items) + 1))
    _write_csv(args.output_csv, items)
    lines = [
        "# Special mora manual inspection pack",
        "",
        "This is not a formal listener test. It is a developer/teacher inspection pack before any limited candidate rollout.",
        "",
        f"- total items: {len(items)}",
        f"- JVS false alarms: {sum(1 for x in items if x['source'] == 'JVS_false_alarm')}",
        f"- JVS allowed candidates: {sum(1 for x in items if x['source'] == 'JVS_allowed_candidate')}",
        f"- JANON outliers: {sum(1 for x in items if x['source'] == 'JANON_outlier')}",
        f"- counterfactual positives: {sum(1 for x in items if x['source'] == 'counterfactual_positive')}",
        f"- near-boundary suppressed examples: {sum(1 for x in items if x['source'] == 'near_boundary')}",
        "",
        "Suggested review labels are included in the CSV: seems_valid_feedback, false_alarm, unclear_alignment, audio_quality_issue, comment.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manual inspection pack for special mora limited candidates")
    parser.add_argument("--false-alarm-cases", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "special_mora_false_alarm_cases.csv")
    parser.add_argument("--v2-jvs-decisions", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "jvs_shadow_decisions_v2.csv")
    parser.add_argument("--janon-v2", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "janon_shadow_trend_v2.csv")
    parser.add_argument("--counterfactual", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "counterfactual_feature_sensitivity.csv")
    parser.add_argument("--sample-audit", type=Path, default=ROOT / "results" / "calibration" / "special_mora_sample_audit.csv")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "manual_inspection_items.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "special_mora_manual_inspection_pack.md")
    args = parser.parse_args()
    print(json.dumps({"items": len(run(args)), "output_csv": str(args.output_csv), "report": str(args.report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
