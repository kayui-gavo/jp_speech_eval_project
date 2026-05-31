#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
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
    "severity",
    "comment",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    for field in ANNOTATION_FIELDS:
        if field not in fields:
            fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{**row, **{k: row.get(k, "") for k in ANNOTATION_FIELDS}} for row in rows])


def build_annotation_template(items_csv: Path, output_csv: Path) -> bool:
    """Create a blank annotation CSV. Returns False if it already existed."""

    if output_csv.exists():
        return False
    _write_csv(output_csv, _read_csv(items_csv))
    return True


def _source_label(source: str) -> str:
    labels = {
        "JVS_false_alarm": "JVS false alarm",
        "JVS_allowed_candidate": "JVS allowed candidate",
        "JANON_outlier": "JANON trend-only",
        "counterfactual_positive": "counterfactual synthetic",
        "near_boundary": "near-boundary suppressed",
    }
    return labels.get(source, source or "unknown")


def build_review_viewer(items_csv: Path, output_html: Path) -> int:
    rows = _read_csv(items_csv)
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get("source", "unknown"), []).append(row)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Special mora manual review</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;line-height:1.45}"
        "table{border-collapse:collapse;width:100%;margin:12px 0 28px}td,th{border:1px solid #ddd;padding:6px;vertical-align:top}"
        "th{background:#f6f6f6}.tag{font-weight:700}.muted{color:#666}audio{max-width:220px}</style>",
        "</head><body>",
        "<h1>Special mora manual review</h1>",
        "<p class='muted'>Developer/teacher inspection pack only. This is not a formal listener test.</p>",
    ]
    for source, items in sorted(groups.items()):
        parts.append(f"<h2>{html.escape(_source_label(source))} ({len(items)})</h2>")
        parts.append("<table><tr><th>Item</th><th>Utterance</th><th>Evidence</th><th>Audio</th><th>Human check fields</th></tr>")
        for row in items:
            audio_path = row.get("audio_path", "")
            audio_cell = html.escape(audio_path)
            if audio_path and Path(audio_path).exists():
                uri = Path(audio_path).resolve().as_uri()
                audio_cell = f"<audio controls src='{html.escape(uri)}'></audio><br><span class='muted'>{html.escape(audio_path)}</span>"
            fields = "<br>".join(f"{html.escape(k)}: ______" for k in ANNOTATION_FIELDS)
            evidence = (
                f"type={html.escape(row.get('special_mora_type',''))}<br>"
                f"mora={html.escape(row.get('surface_mora',''))}<br>"
                f"decision={html.escape(row.get('decision',''))}<br>"
                f"feature={html.escape(row.get('feature_value',''))}<br>"
                f"user_low={html.escape(row.get('threshold_user_low',''))}<br>"
                f"near_boundary={html.escape(row.get('near_boundary',''))}<br>"
                f"evidence={html.escape(row.get('evidence_confidence',''))}<br>"
                f"phones={html.escape(row.get('phone_sequence_for_mora',''))}"
            )
            parts.append(
                "<tr>"
                f"<td><span class='tag'>{html.escape(row.get('item_id',''))}</span><br>{html.escape(source)}</td>"
                f"<td>{html.escape(row.get('transcript',''))}<br><span class='muted'>{html.escape(row.get('speaker_id',''))}/{html.escape(row.get('utterance_id',''))}</span></td>"
                f"<td>{evidence}</td><td>{audio_cell}</td><td>{fields}</td>"
                "</tr>"
            )
        parts.append("</table>")
    parts.append("</body></html>")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text("\n".join(parts), encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build special mora manual review template and static HTML viewer")
    parser.add_argument("--items-csv", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "manual_inspection_items.csv")
    parser.add_argument("--annotation-template", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "manual_inspection_annotations_template.csv")
    parser.add_argument("--output-html", type=Path, default=ROOT / "reports" / "special_mora_manual_review_viewer.html")
    args = parser.parse_args()
    created = build_annotation_template(args.items_csv, args.annotation_template)
    count = build_review_viewer(args.items_csv, args.output_html)
    print({"items": count, "template_created": created, "template": str(args.annotation_template), "html": str(args.output_html)})


if __name__ == "__main__":
    main()
