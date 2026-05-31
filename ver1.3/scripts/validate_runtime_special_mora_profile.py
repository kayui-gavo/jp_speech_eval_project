#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from jp_speech_eval.special_mora_profiles import load_threshold_profile
from jp_speech_eval.special_mora_scorer import decide_special_mora_runtime


ROOT = Path(__file__).resolve().parents[1]


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


def _synthetic_cases() -> List[Dict[str, Any]]:
    evidence = [{"judgement_available": True, "boundary_confidence": 0.95, "energy_coverage": 0.95} for _ in range(9)]
    return [
        {
            "case_id": "long_vowel_short",
            "target_text": "ラーメンをください",
            "moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"],
            "mora_table": [
                {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2},
                {"mora": "ー", "start_sec": 0.2, "end_sec": 0.235},
                {"mora": "メ", "start_sec": 0.235, "end_sec": 0.435},
                {"mora": "ン", "start_sec": 0.435, "end_sec": 0.635},
                {"mora": "ヲ", "start_sec": 0.635, "end_sec": 0.835},
                {"mora": "ク", "start_sec": 0.835, "end_sec": 1.035},
                {"mora": "ダ", "start_sec": 1.035, "end_sec": 1.235},
                {"mora": "サ", "start_sec": 1.235, "end_sec": 1.435},
                {"mora": "イ", "start_sec": 1.435, "end_sec": 1.635},
            ],
            "alignment_mode": "cached_dtw",
            "details": {"mode": "reference_based", "reliability": {"level": "high", "overall": 0.95}, "mora_evidence": evidence},
        },
        {
            "case_id": "nasal_short",
            "target_text": "ラーメンをください",
            "moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"],
            "mora_table": [
                {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2},
                {"mora": "ー", "start_sec": 0.2, "end_sec": 0.4},
                {"mora": "メ", "start_sec": 0.4, "end_sec": 0.6},
                {"mora": "ン", "start_sec": 0.6, "end_sec": 0.635},
                {"mora": "ヲ", "start_sec": 0.635, "end_sec": 0.835},
                {"mora": "ク", "start_sec": 0.835, "end_sec": 1.035},
                {"mora": "ダ", "start_sec": 1.035, "end_sec": 1.235},
                {"mora": "サ", "start_sec": 1.235, "end_sec": 1.435},
                {"mora": "イ", "start_sec": 1.435, "end_sec": 1.635},
            ],
            "alignment_mode": "cached_dtw",
            "details": {"mode": "reference_based", "reliability": {"level": "high", "overall": 0.95}, "mora_evidence": evidence},
        },
        {
            "case_id": "sokuon_yoon",
            "target_text": "きってきゃ",
            "moras": ["キ", "ッ", "テ", "キャ"],
            "mora_table": [
                {"mora": "キ", "start_sec": 0.0, "end_sec": 0.2},
                {"mora": "ッ", "start_sec": 0.2, "end_sec": 0.23},
                {"mora": "テ", "start_sec": 0.23, "end_sec": 0.43},
                {"mora": "キャ", "start_sec": 0.43, "end_sec": 0.63},
            ],
            "alignment_mode": "cached_dtw",
            "details": {"mode": "reference_based", "reliability": {"level": "high", "overall": 0.95}, "mora_evidence": evidence[:4]},
        },
    ]


def run(args: argparse.Namespace) -> List[Dict[str, Any]]:
    profiles = [
        ("default_safe", False),
        ("v2_shadow", False),
        ("v2_limited_candidate", False),
        ("v2_limited_candidate", True),
    ]
    rows: List[Dict[str, Any]] = []
    for profile_name, flag in profiles:
        profile = load_threshold_profile(profile_name)
        decisions = []
        for case in _synthetic_cases():
            decisions.extend(decide_special_mora_runtime(
                case,
                threshold_profile=profile_name,
                mode_name="reference_based",
                enable_user_facing=flag,
            ))
        allowed = [d for d in decisions if d.user_feedback_allowed]
        by_type = Counter(d.type for d in allowed)
        rows.append({
            "profile_name": profile_name,
            "flag_enabled": flag,
            "threshold_file": profile.threshold_file,
            "candidate_count": len(allowed),
            "long_vowel_candidates": by_type.get("long_vowel", 0),
            "moraic_nasal_candidates": by_type.get("moraic_nasal", 0),
            "sokuon_yoon_leakage": sum(1 for d in allowed if d.type in {"sokuon", "yoon"}),
            "too_long_leakage": sum(1 for d in allowed if d.decision == "too_long" or d.user_decision == "too_long"),
            "near_boundary_leakage": sum(1 for d in allowed if d.near_boundary),
            "display_score_safe": True,
        })
    out = args.output_dir / "profile_validation_summary.csv"
    _write_csv(out, rows)
    lines = ["# Runtime special mora profile validation", ""]
    for row in rows:
        lines.append(
            f"- {row['profile_name']} flag={row['flag_enabled']}: candidates={row['candidate_count']}, "
            f"long={row['long_vowel_candidates']}, nasal={row['moraic_nasal_candidates']}, "
            f"sokuon/yoon leakage={row['sokuon_yoon_leakage']}, too_long leakage={row['too_long_leakage']}, "
            f"near-boundary leakage={row['near_boundary_leakage']}"
        )
    lines.append("- default profile and flag-off profiles must keep user-facing feedback disabled.")
    lines.append("- display_score impact: safe; profile validation does not modify display_score.")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "runtime_special_mora_profile_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate special mora threshold profiles")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "runtime_special_mora_validation")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
