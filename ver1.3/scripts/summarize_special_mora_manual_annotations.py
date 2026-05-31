#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_FIELDS = [
    "seems_valid_feedback",
    "false_alarm",
    "alignment_issue",
    "audio_quality_issue",
    "wording_ok",
    "should_allow_user_facing",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _annotated(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [row for row in rows if any(str(row.get(field, "")).strip() for field in ANNOTATION_FIELDS + ["severity", "comment"])]


def _rate(rows: List[Dict[str, str]], field: str, value: str = "yes") -> float | None:
    answered = [row for row in rows if row.get(field) in {"yes", "no", "unsure"}]
    if not answered:
        return None
    return round(sum(1 for row in answered if row.get(field) == value) / len(answered), 4)


def summarize(annotation_csv: Path) -> Dict[str, object]:
    rows = _annotated(_read_csv(annotation_csv))
    by_source = Counter(row.get("source", "unknown") for row in rows)
    by_type = Counter(row.get("special_mora_type", "unknown") for row in rows)
    unsure = sum(1 for row in rows for field in ANNOTATION_FIELDS if row.get(field) == "unsure")
    answered = sum(1 for row in rows for field in ANNOTATION_FIELDS if row.get(field) in {"yes", "no", "unsure"})
    return {
        "total_annotated": len(rows),
        "annotated_by_source": dict(by_source),
        "annotated_by_special_mora_type": dict(by_type),
        "valid_feedback_rate": _rate(rows, "seems_valid_feedback"),
        "false_alarm_rate": _rate(rows, "false_alarm"),
        "alignment_issue_rate": _rate(rows, "alignment_issue"),
        "audio_quality_issue_rate": _rate(rows, "audio_quality_issue"),
        "wording_ok_rate": _rate(rows, "wording_ok"),
        "should_allow_user_facing_rate": _rate(rows, "should_allow_user_facing"),
        "unsure_rate": None if answered == 0 else round(unsure / answered, 4),
        "comments": [row.get("comment", "") for row in rows if row.get("comment")],
    }


def write_outputs(summary: Dict[str, object], out_csv: Path, report: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    lines = ["# Special mora manual annotation summary", ""]
    if int(summary["total_annotated"]) == 0:
        lines.append("no human annotations yet")
    else:
        for key, value in summary.items():
            lines.append(f"- {key}: {value}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize manual special mora annotations")
    parser.add_argument("--annotations", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "manual_inspection_annotations.csv")
    parser.add_argument("--out-csv", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "manual_inspection_annotation_summary.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "special_mora_manual_annotation_summary.md")
    args = parser.parse_args()
    summary = summarize(args.annotations)
    write_outputs(summary, args.out_csv, args.report)
    print(summary)


if __name__ == "__main__":
    main()
