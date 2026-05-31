#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import os
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


def build_annotation_template(items_csv: Path, output_csv: Path, *, overwrite: bool = False) -> bool:
    """Create a blank annotation CSV.

    Returns False if it already existed and overwrite is disabled.
    """

    if output_csv.exists() and not overwrite:
        return False
    _write_csv(output_csv, _read_csv(items_csv))
    return True


def _source_label(source: str) -> str:
    labels = {
        "JVS_false_alarm": "母语者误伤候选：系统可能不该提示",
        "JVS_allowed_candidate": "可考虑开放候选：系统认为证据较强",
        "JANON_outlier": "学习者趋势样本：只看倾向，不当真值",
        "counterfactual_positive": "合成压力样本：只测规则敏感度",
        "near_boundary": "边界样本：默认应压制不提示",
    }
    return labels.get(source, source or "未知来源")


def _type_label(special_mora_type: str) -> str:
    labels = {
        "long_vowel": "长音",
        "moraic_nasal": "拨音「ん」",
        "sokuon": "促音「っ」",
        "yoon": "拗音",
    }
    return labels.get(special_mora_type, special_mora_type or "未知")


def _decision_label(decision: str) -> str:
    labels = {
        "too_short": "系统怀疑：读得偏短",
        "too_long": "系统怀疑：读得偏长",
        "ok": "系统判断：大致正常",
        "uncertain": "系统判断：证据不足",
    }
    return labels.get(decision, decision or "未记录")


def _review_prompt(row: Dict[str, str]) -> str:
    mora_type = _type_label(row.get("special_mora_type", ""))
    mora = row.get("surface_mora", "")
    decision = _decision_label(row.get("decision", ""))
    if row.get("source") == "JVS_false_alarm":
        return f"请听原音：这是母语者样本。如果听起来正常，就说明这条「{mora_type}」提示可能会误伤用户。"
    if row.get("source") == "near_boundary":
        return "这是接近阈值边界的样本。除非问题非常明显，否则应该继续压制，不给用户看。"
    if row.get("source") == "counterfactual_positive":
        return "这是合成/反事实检查样本，主要用来确认规则会不会响应，不用于判断真实发音好坏。"
    return f"请听原音并判断：{decision} 这条提示是否足够可靠，可以展示给用户。"


def _audio_cell(audio_path: str, output_html: Path) -> str:
    if not audio_path:
        return "<span class='warn'>没有音频路径：需要重新生成 inspection pack。</span>"
    path = Path(audio_path)
    if not path.exists():
        return f"<span class='warn'>找不到音频文件</span><br><code>{html.escape(audio_path)}</code>"
    rel = os.path.relpath(path.resolve(), output_html.parent.resolve())
    return (
        f"<audio controls preload='none' src='{html.escape(rel)}'></audio>"
        f"<br><code>{html.escape(rel)}</code>"
    )


def build_review_viewer(items_csv: Path, output_html: Path) -> int:
    rows = _read_csv(items_csv)
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get("source", "unknown"), []).append(row)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>特殊拍人工复核</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Noto Sans CJK SC',sans-serif;margin:24px;line-height:1.55;color:#1f2937;background:#fafafa}"
        "h1{margin-bottom:6px}.summary{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px 16px;margin:12px 0 18px}"
        "table{border-collapse:separate;border-spacing:0;width:100%;margin:12px 0 28px;background:#fff;border:1px solid #ddd;border-radius:8px;overflow:hidden}"
        "td,th{border-bottom:1px solid #e5e7eb;padding:10px;vertical-align:top}tr:last-child td{border-bottom:0}"
        "th{background:#f3f4f6;text-align:left}.tag{font-weight:700}.muted{color:#6b7280}.warn{color:#b45309;font-weight:700}"
        ".hint{background:#f8fafc;border-left:4px solid #2563eb;padding:8px 10px;margin-top:8px}.metric{font-size:13px;color:#4b5563}audio{width:260px;max-width:100%}code{font-size:12px;word-break:break-all}</style>",
        "</head><body>",
        "<h1>特殊拍人工复核</h1>",
        "<div class='summary'>"
        "<p><b>这个页面是做什么的：</b>检查系统的长音、拨音等特殊拍提示，会不会误伤用户。</p>"
        "<p><b>怎么判断：</b>先播放原音，再看系统怀疑的问题是否真的听得出来。不要看见数值就相信系统，耳朵判断优先。</p>"
        "<p><b>标注建议：</b>如果提示可以给用户看，填 <code>should_allow_user_facing=yes</code>；如果母语者听起来正常却被系统判错，填 <code>false_alarm=yes</code>；如果对齐或录音有问题，填对应问题字段。</p>"
        "<p class='muted'>这不是正式听辨实验，只是上线前的人工安全检查。</p>"
        "</div>",
    ]
    for source, items in sorted(groups.items()):
        parts.append(f"<h2>{html.escape(_source_label(source))} ({len(items)})</h2>")
        parts.append("<table><tr><th>样本</th><th>句子</th><th>系统怀疑什么</th><th>原音</th><th>人工标注怎么填</th></tr>")
        for row in items:
            audio_path = row.get("audio_path", "")
            audio_cell = _audio_cell(audio_path, output_html)
            fields = (
                "<b>最低限建议填这几项：</b><br>"
                "should_allow_user_facing: yes / no / unsure<br>"
                "false_alarm: yes / no / unsure<br>"
                "alignment_issue: yes / no / unsure<br>"
                "severity: none / mild / clear / severe<br>"
                "comment: 自由备注"
            )
            evidence = (
                f"<b>{html.escape(_type_label(row.get('special_mora_type','')))}</b> / 目标拍：{html.escape(row.get('surface_mora',''))}<br>"
                f"{html.escape(_decision_label(row.get('decision','')))}<br>"
                f"<div class='hint'>{html.escape(_review_prompt(row))}</div>"
                f"<div class='metric'>内部数值：feature={html.escape(row.get('feature_value',''))}，下阈值={html.escape(row.get('threshold_user_low',''))}，上阈值={html.escape(row.get('threshold_user_high',''))}<br>"
                f"边界附近={html.escape(row.get('near_boundary',''))}，证据置信度={html.escape(row.get('evidence_confidence',''))}，phone={html.escape(row.get('phone_sequence_for_mora',''))}</div>"
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
    parser.add_argument("--refresh-template", action="store_true", help="Overwrite the blank annotation template from current items.")
    args = parser.parse_args()
    created = build_annotation_template(args.items_csv, args.annotation_template, overwrite=args.refresh_template)
    count = build_review_viewer(args.items_csv, args.output_html)
    print({"items": count, "template_created": created, "template": str(args.annotation_template), "html": str(args.output_html)})


if __name__ == "__main__":
    main()
