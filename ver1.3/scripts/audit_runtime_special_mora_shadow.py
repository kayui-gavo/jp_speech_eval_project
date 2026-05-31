#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from jp_speech_eval.text_frontend import build_text_info
from jp_speech_eval.special_mora_scorer import (
    decide_special_mora_runtime,
    load_runtime_special_mora_thresholds,
    select_special_mora_feedback_candidate,
    special_mora_score_from_decisions,
)


ROOT = Path(__file__).resolve().parents[1]


def _read_json_cases(path: Path | None) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return [_synthetic_case()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        data = data.get("cases", [])
    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        case = dict(item)
        if not case.get("moras") and case.get("text"):
            case = _case_from_text(case)
        out.append(case)
    return out


def _case_from_text(case: Mapping[str, Any]) -> Dict[str, Any]:
    info = build_text_info(str(case.get("text") or ""))
    moras = list(info.moras)
    duration = 0.18
    table = [
        {
            "mora": mora,
            "start_sec": round(i * duration, 4),
            "end_sec": round((i + 1) * duration, 4),
        }
        for i, mora in enumerate(moras)
    ]
    return {
        "case_id": str(case.get("id") or case.get("case_id") or info.text),
        "target_text": info.text,
        "moras": moras,
        "mora_table": table,
        "alignment_mode": "cached_dtw",
        "details": {
            "mode": "reference_based",
            "weak_reference": False,
            "verified_level": "ojad_checked",
            "pitch_target_source": "ojad_checked",
            "reliability": {"level": "high", "overall": 0.95, "alignment": 0.9, "f0_coverage": 0.9},
            "mora_evidence": [
                {"judgement_available": True, "boundary_confidence": 0.9, "energy_coverage": 0.9}
                for _ in moras
            ],
        },
    }


def _synthetic_case() -> Dict[str, Any]:
    return {
        "case_id": "synthetic_ramen_shadow",
        "target_text": "ラーメンをください",
        "moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"],
        "mora_table": [
            {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2},
            {"mora": "ー", "start_sec": 0.2, "end_sec": 0.24},
            {"mora": "メ", "start_sec": 0.24, "end_sec": 0.44},
            {"mora": "ン", "start_sec": 0.44, "end_sec": 0.62},
            {"mora": "ヲ", "start_sec": 0.62, "end_sec": 0.82},
            {"mora": "ク", "start_sec": 0.82, "end_sec": 1.02},
            {"mora": "ダ", "start_sec": 1.02, "end_sec": 1.22},
            {"mora": "サ", "start_sec": 1.22, "end_sec": 1.42},
            {"mora": "イ", "start_sec": 1.42, "end_sec": 1.62},
        ],
        "alignment_mode": "cached_dtw",
        "details": {
            "mode": "reference_based",
            "weak_reference": False,
            "reliability": {"level": "high", "overall": 0.95, "alignment": 0.9, "f0_coverage": 0.9},
            "mora_evidence": [
                {"judgement_available": True, "boundary_confidence": 0.9, "energy_coverage": 0.9}
                for _ in range(9)
            ],
        },
    }


def _load_csv(path: Path) -> List[Dict[str, str]]:
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
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _decision_rows(cases: List[Dict[str, Any]], threshold_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases):
        case_id = str(case.get("case_id") or case.get("id") or f"case_{idx + 1}")
        decisions = decide_special_mora_runtime(
            case,
            threshold_path=threshold_path,
            weak_reference=bool(case.get("details", {}).get("weak_reference", False)),
            enable_runtime_shadow=True,
            enable_user_facing=True,
        )
        score = special_mora_score_from_decisions(decisions)
        selected = select_special_mora_feedback_candidate(decisions)
        for item in decisions:
            row = item.to_dict()
            row.update({
                "case_id": case_id,
                "target_text": case.get("target_text", ""),
                "special_mora_score": score,
                "selected_feedback_candidate": bool(selected and selected.mora_index == item.mora_index and selected.type == item.type),
            })
            rows.append(row)
    return rows


def _sample_audit_summary(rows: List[Dict[str, str]], label: str) -> List[str]:
    if not rows:
        return [f"- {label}: no sample audit rows found"]
    by_type = Counter(row.get("special_mora_type", "") for row in rows)
    decisions = Counter(row.get("decision", "") for row in rows)
    return [
        f"- {label}: rows={len(rows)}, by_type={dict(by_type)}",
        f"- {label}: offline sample decisions={dict(decisions)}",
    ]


def _write_report(
    path: Path,
    thresholds: Mapping[str, Mapping[str, Any]],
    decision_rows: List[Dict[str, Any]],
    jvs_rows: List[Dict[str, str]],
    janon_rows: List[Dict[str, str]],
) -> None:
    status_by_type = {key: value.get("status") for key, value in thresholds.items()}
    allowed_by_type: Dict[str, int] = defaultdict(int)
    suppressed_by_reason: Counter[str] = Counter()
    leaked = []
    for row in decision_rows:
        if row.get("user_feedback_allowed"):
            allowed_by_type[str(row.get("type"))] += 1
        else:
            suppressed_by_reason[str(row.get("suppression_reason") or "none")] += 1
        if row.get("type") in {"sokuon", "yoon"} and row.get("user_feedback_allowed"):
            leaked.append(row)

    jvs_false_alarm = [
        row for row in decision_rows
        if str(row.get("case_id", "")).lower().startswith("jvs")
        and row.get("user_feedback_allowed")
        and row.get("decision") in {"too_short", "too_long"}
    ]
    display_score_safety = "pass: unavailable special_mora_score is not emitted as zero"
    if any(row.get("special_mora_score") == 0 for row in decision_rows):
        display_score_safety = "check: at least one runtime special_mora_score is 0"

    lines = [
        "# Runtime special mora shadow report",
        "",
        "## Threshold status",
    ]
    for key in sorted(status_by_type):
        th = thresholds[key]
        lines.append(
            f"- {key}: status={th.get('status')}, low={th.get('low_ratio')}, high={th.get('high_ratio')}, "
            f"sample_count={th.get('sample_count')}, source={th.get('source_dataset')}"
        )
    lines.extend([
        "",
        "## Runtime shadow decisions",
        f"- total decisions: {len(decision_rows)}",
        f"- user_feedback_allowed by type: {dict(allowed_by_type)}",
        f"- suppressed by reason: {dict(suppressed_by_reason)}",
        f"- sokuon/yoon user-facing leakage: {len(leaked)}",
        f"- display score safety: {display_score_safety}",
        f"- JVS native false alarm under runtime rows: {len(jvs_false_alarm)}",
        "",
        "## Sample audit trend inputs",
        *_sample_audit_summary(jvs_rows, "JVS"),
        *_sample_audit_summary(janon_rows, "JANON"),
        "",
        "## Allowed feedback examples",
    ])
    allowed = [row for row in decision_rows if row.get("user_feedback_allowed")]
    if not allowed:
        lines.append("- none")
    for row in allowed[:8]:
        lines.append(f"- {row.get('case_id')}: {row.get('type')} {row.get('surface_mora')} -> {row.get('feedback_candidate_text')}")
    lines.append("")
    lines.append("## Suppressed examples")
    for row in [r for r in decision_rows if not r.get("user_feedback_allowed")][:8]:
        lines.append(f"- {row.get('case_id')}: {row.get('type')} {row.get('surface_mora')} suppressed={row.get('suppression_reason')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit calibrated special-mora runtime shadow decisions")
    parser.add_argument("--eval-test-cases", type=Path, default=ROOT / "data" / "eval_test_cases.json")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "results" / "calibration" / "special_mora_thresholds.json")
    parser.add_argument("--jvs-sample-audit", type=Path, default=ROOT / "results" / "calibration" / "special_mora_sample_audit.csv")
    parser.add_argument("--janon-sample-audit", type=Path, default=ROOT / "results" / "calibration" / "janon_special_mora_metrics.csv")
    parser.add_argument("--out-csv", type=Path, default=ROOT / "results" / "runtime_special_mora_shadow_eval.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "runtime_special_mora_shadow_report.md")
    args = parser.parse_args()

    cases = _read_json_cases(args.eval_test_cases)
    thresholds = load_runtime_special_mora_thresholds(args.thresholds)
    rows = _decision_rows(cases, args.thresholds)
    jvs_rows = _load_csv(args.jvs_sample_audit)
    janon_rows = _load_csv(args.janon_sample_audit)
    _write_csv(args.out_csv, rows)
    _write_report(args.report, thresholds, rows, jvs_rows, janon_rows)
    print(json.dumps({
        "cases": len(cases),
        "decisions": len(rows),
        "out_csv": str(args.out_csv),
        "report": str(args.report),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
