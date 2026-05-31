#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(value, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _summary_row(path: Path) -> Dict[str, str]:
    rows = _read_csv(path)
    return rows[0] if rows else {}


def evaluate_gate(profile_csv: Path, false_alarm_csv: Path, annotation_summary_csv: Path) -> Dict[str, object]:
    profile_rows = _read_csv(profile_csv)
    false_rows = {row.get("special_mora_type"): row for row in _read_csv(false_alarm_csv)}
    ann = _summary_row(annotation_summary_csv)
    annotated = int(_float(ann.get("total_annotated"), 0))
    allow_rate = _float(ann.get("should_allow_user_facing_rate"), 0)
    human_false_alarm = _float(ann.get("false_alarm_rate"), 1)
    alignment_issue = _float(ann.get("alignment_issue_rate"), 1)
    profile_limited = [r for r in profile_rows if r.get("profile_name") == "v2_limited_candidate" and r.get("flag_enabled") == "True"]
    leakage_ok = bool(profile_limited) and all(_float(profile_limited[0].get(k), 1) == 0 for k in ("sokuon_yoon_leakage", "too_long_leakage", "near_boundary_leakage"))
    decisions: Dict[str, Dict[str, object]] = {}
    for special_type in ("long_vowel", "moraic_nasal"):
        fa = _float(false_rows.get(special_type, {}).get("false_alarm_proxy_rate_all"), 1)
        ok = (
            fa <= 0.05
            and leakage_ok
            and annotated >= 10
            and allow_rate >= 0.80
            and human_false_alarm <= 0.10
            and alignment_issue < 0.25
        )
        reason = []
        if fa > 0.05:
            reason.append("jvs_false_alarm_above_5_percent")
        if not leakage_ok:
            reason.append("profile_leakage_or_missing_profile_validation")
        if annotated < 10:
            reason.append("blocked_pending_manual_inspection")
        if allow_rate < 0.80:
            reason.append("manual_allow_rate_below_80_percent")
        if human_false_alarm > 0.10:
            reason.append("manual_false_alarm_rate_above_10_percent")
        decisions[special_type] = {
            "rollout_status": "limited_user_facing_candidate" if ok else "blocked_pending_manual_inspection",
            "jvs_false_alarm_rate": fa,
            "manual_annotated_count": annotated,
            "manual_should_allow_rate": allow_rate,
            "manual_false_alarm_rate": human_false_alarm,
            "reason": reason or ["passed_limited_gate"],
        }
    decisions["sokuon"] = {"rollout_status": "blocked_insufficient_native_evidence"}
    decisions["yoon"] = {"rollout_status": "blocked_debug_only_duration_not_valid"}
    return {
        "gate_version": "special_mora_rollout_gate_v1",
        "profile_leakage_ok": leakage_ok,
        "manual_annotation_summary_available": annotated > 0,
        "decisions": decisions,
    }


def write_outputs(gate: Dict[str, object], json_path: Path, report: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Special mora rollout gate decision", ""]
    for special_type, item in gate["decisions"].items():
        lines.append(f"## {special_type}")
        for key, value in item.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend([
        "## Limitations",
        "- manual inspection is not a full listener study",
        "- JVS controls native false alarm risk but does not prove learner benefit",
        "- JANON trend is not ground truth",
        "- counterfactual feature perturbation is not human validation",
        "- limited_candidate is not full rollout",
    ])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate special mora rollout gate")
    parser.add_argument("--profile-validation", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "profile_validation_summary.csv")
    parser.add_argument("--false-alarm-v2", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "jvs_false_alarm_by_type_v2.csv")
    parser.add_argument("--annotation-summary", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "manual_inspection_annotation_summary.csv")
    parser.add_argument("--out-json", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "special_mora_rollout_gate.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "special_mora_rollout_gate_decision.md")
    args = parser.parse_args()
    gate = evaluate_gate(args.profile_validation, args.false_alarm_v2, args.annotation_summary)
    write_outputs(gate, args.out_json, args.report)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
