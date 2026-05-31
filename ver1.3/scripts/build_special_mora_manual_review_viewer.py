#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
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

ANNOTATION_FIELDS_V2 = [
    "intelligibility",
    "naturalness",
    "communication_impact",
    "variation_type",
    "audible_variation",
    "should_feedback",
    "feedback_strength",
    "wording_ok",
    "alignment_issue",
    "audio_quality_issue",
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


def _write_csv_with_fields(path: Path, rows: List[Dict[str, str]], annotation_fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    for field in annotation_fields:
        if field not in fields:
            fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{**row, **{k: row.get(k, "") for k in annotation_fields}} for row in rows])


def build_annotation_template(items_csv: Path, output_csv: Path, *, overwrite: bool = False) -> bool:
    """Create a blank annotation CSV.

    Returns False if it already existed and overwrite is disabled.
    """

    if output_csv.exists() and not overwrite:
        return False
    _write_csv(output_csv, _read_csv(items_csv))
    return True


def build_annotation_template_v2(items_csv: Path, output_csv: Path, *, overwrite: bool = False) -> bool:
    """Create a v2 annotation CSV that separates variation from true errors."""

    if output_csv.exists() and not overwrite:
        return False
    rows: List[Dict[str, str]] = []
    for row in _read_csv(items_csv):
        rows.append({
            **row,
            "severity_from_system": row.get("severity_from_system", ""),
            "feedback_candidate_text": row.get("feedback_candidate_text", ""),
        })
    _write_csv_with_fields(output_csv, rows, ANNOTATION_FIELDS_V2)
    return True


def write_guideline_v2(path: Path) -> None:
    lines = [
        "# 特殊拍人工复核指南 v2",
        "",
        "目标：区分“听得出长短变化”和“真的应该给学习者反馈的问题”。",
        "",
        "## 核心原则",
        "- 听得出变化，不等于发音错误。",
        "- 母语者语流中的长音、拨音会自然伸缩。",
        "- near-boundary 和 mild variation 默认通过，不扣分，不强提示。",
        "- too_long 默认 debug-only，不进入用户端纠错。",
        "- JANON 只看学习者趋势，不当 ground truth。",
        "- counterfactual 只测规则敏感度，不是真人验证。",
        "",
        "## 字段怎么填",
        "- intelligibility: 是否听得懂。clear / mostly_clear / hard_to_understand / unsure。",
        "- naturalness: 听起来是否自然。natural / slightly_unnatural / unnatural / unsure。",
        "- communication_impact: 是否影响交流。none / minor / clear / severe / unsure。",
        "- variation_type: natural_variation / acceptable_variation / possible_issue / likely_error / alignment_uncertain / unsure。",
        "- audible_variation: 是否听得出长短变化。yes / no / unsure。",
        "- should_feedback: 是否值得给用户提示。yes / no / unsure。",
        "- feedback_strength: none / gentle_tip / practice_focus / correction。",
        "- wording_ok: 当前候选文案是否可以接受。",
        "- alignment_issue: 对齐是否疑似有问题。",
        "- audio_quality_issue: 原音质量是否影响判断。",
        "",
        "## 上线规则",
        "- natural_variation / acceptable_variation 不应扣分。",
        "- communication_impact none/minor 不应强反馈。",
        "- 只有 clear/severe + should_feedback=yes 才可能进入用户端提示候选。",
        "- C 端当前最多使用 gentle_tip / practice_focus，不使用 correction。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _audio_cell(audio_path: str, output_html: Path, asset_dir: Path | None) -> str:
    if not audio_path:
        return "<span class='warn'>没有音频路径：需要重新生成 inspection pack。</span>"
    path = Path(audio_path)
    if not path.exists():
        return f"<span class='warn'>找不到音频文件</span><br><code>{html.escape(audio_path)}</code>"
    src = path.resolve().as_uri()
    link = src
    shown_path = str(path)
    if asset_dir is not None:
        asset_dir.mkdir(parents=True, exist_ok=True)
        speaker_prefix = path.parent.parent.parent.name if path.parent.name == "wav24kHz16bit" else "audio"
        dest = asset_dir / f"{speaker_prefix}_{path.stem}{path.suffix}"
        if not dest.exists() or dest.stat().st_size != path.stat().st_size:
            shutil.copy2(path, dest)
        src = dest.relative_to(output_html.parent).as_posix()
        link = src
        shown_path = str(dest)
    return (
        f"<audio controls preload='metadata' src='{html.escape(src)}'></audio>"
        f"<br><a href='{html.escape(link)}' target='_blank'>如果播放器报错，点这里直接打开原音</a>"
        f"<br><code>{html.escape(shown_path)}</code>"
    )


def _select(name: str, item_id: str, options: List[str]) -> str:
    opts = ["<option value=''></option>"] + [f"<option value='{html.escape(o)}'>{html.escape(o)}</option>" for o in options]
    return f"<select data-field='{html.escape(name)}' data-item='{html.escape(item_id)}'>{''.join(opts)}</select>"


def _annotation_controls(row: Dict[str, str]) -> str:
    item_id = row.get("item_id", "")
    yn = ["yes", "no", "unsure"]
    severity = ["none", "mild", "clear", "severe"]
    return (
        "<div class='formbox'>"
        "<label>这条提示能给用户看吗？<br>"
        f"{_select('should_allow_user_facing', item_id, yn)}</label>"
        "<label>这是误伤吗？<br>"
        f"{_select('false_alarm', item_id, yn)}</label>"
        "<label>对齐有问题吗？<br>"
        f"{_select('alignment_issue', item_id, yn)}</label>"
        "<label>录音质量有问题吗？<br>"
        f"{_select('audio_quality_issue', item_id, yn)}</label>"
        "<label>这句反馈文案可以接受吗？<br>"
        f"{_select('wording_ok', item_id, yn)}</label>"
        "<label>问题严重程度<br>"
        f"{_select('severity', item_id, severity)}</label>"
        "<label>备注<br>"
        f"<textarea data-field='comment' data-item='{html.escape(item_id)}' placeholder='例如：母语者听起来正常，建议不要提示。'></textarea></label>"
        "</div>"
    )


def build_review_viewer(items_csv: Path, output_html: Path, *, copy_audio_assets: bool = True) -> int:
    rows = _read_csv(items_csv)
    asset_dir = output_html.parent / "special_mora_manual_review_audio" if copy_audio_assets else None
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get("source", "unknown"), []).append(row)
    base_rows_json = json.dumps(rows, ensure_ascii=False)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>特殊拍人工复核</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Noto Sans CJK SC',sans-serif;margin:24px;line-height:1.55;color:#1f2937;background:#fafafa}"
        "h1{margin-bottom:6px}.summary{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px 16px;margin:12px 0 18px}"
        ".toolbar{position:sticky;top:0;z-index:10;background:#fff;border:1px solid #d1d5db;border-radius:8px;padding:10px 12px;margin:12px 0;box-shadow:0 2px 8px rgba(0,0,0,.06)}"
        "button{border:1px solid #2563eb;background:#2563eb;color:white;border-radius:6px;padding:8px 12px;font-weight:700;cursor:pointer}"
        "button.secondary{background:white;color:#2563eb}"
        "table{border-collapse:separate;border-spacing:0;width:100%;margin:12px 0 28px;background:#fff;border:1px solid #ddd;border-radius:8px;overflow:hidden}"
        "td,th{border-bottom:1px solid #e5e7eb;padding:10px;vertical-align:top}tr:last-child td{border-bottom:0}"
        "th{background:#f3f4f6;text-align:left}.tag{font-weight:700}.muted{color:#6b7280}.warn{color:#b45309;font-weight:700}"
        ".hint{background:#f8fafc;border-left:4px solid #2563eb;padding:8px 10px;margin-top:8px}.metric{font-size:13px;color:#4b5563}audio{width:260px;max-width:100%}code{font-size:12px;word-break:break-all}"
        ".formbox{display:grid;grid-template-columns:1fr 1fr;gap:8px;min-width:260px}.formbox label{font-size:13px;font-weight:700;color:#374151}.formbox select,.formbox textarea{width:100%;box-sizing:border-box;margin-top:3px}.formbox textarea{grid-column:1/-1;min-height:58px}</style>",
        "</head><body>",
        "<h1>特殊拍人工复核</h1>",
        "<div class='summary'>"
        "<p><b>这个页面是做什么的：</b>检查系统的长音、拨音等特殊拍提示，会不会误伤用户。</p>"
        "<p><b>怎么判断：</b>先播放原音，再看系统怀疑的问题是否真的听得出来。不要看见数值就相信系统，耳朵判断优先。</p>"
        "<p><b>标注建议：</b>如果提示可以给用户看，填 <code>should_allow_user_facing=yes</code>；如果母语者听起来正常却被系统判错，填 <code>false_alarm=yes</code>；如果对齐或录音有问题，填对应问题字段。</p>"
        "<p class='muted'>这不是正式听辨实验，只是上线前的人工安全检查。</p>"
        "</div>",
        "<div class='toolbar'>"
        "<button onclick='downloadAnnotations()'>导出当前标注 CSV</button> "
        "<button class='secondary' onclick='markVisibleUnsure()'>把空白项临时填成 unsure</button>"
        "<span class='muted'> 填完后点击导出，把 CSV 保存为 <code>manual_inspection_annotations.csv</code>。</span>"
        "</div>",
    ]
    for source, items in sorted(groups.items()):
        parts.append(f"<h2>{html.escape(_source_label(source))} ({len(items)})</h2>")
        parts.append("<table><tr><th>样本</th><th>句子</th><th>系统怀疑什么</th><th>原音</th><th>人工标注怎么填</th></tr>")
        for row in items:
            audio_path = row.get("audio_path", "")
            audio_cell = _audio_cell(audio_path, output_html, asset_dir)
            fields = _annotation_controls(row)
            evidence = (
                f"<b>{html.escape(_type_label(row.get('special_mora_type','')))}</b> / 目标拍：{html.escape(row.get('surface_mora',''))}<br>"
                f"{html.escape(_decision_label(row.get('decision','')))}<br>"
                f"<div class='hint'>{html.escape(_review_prompt(row))}</div>"
                f"<div class='metric'>内部数值：feature={html.escape(row.get('feature_value',''))}，下阈值={html.escape(row.get('threshold_user_low',''))}，上阈值={html.escape(row.get('threshold_user_high',''))}<br>"
                f"边界附近={html.escape(row.get('near_boundary',''))}，证据置信度={html.escape(row.get('evidence_confidence',''))}，phone={html.escape(row.get('phone_sequence_for_mora',''))}</div>"
            )
            parts.append(
                f"<tr class='review-row' data-item='{html.escape(row.get('item_id',''))}'>"
                f"<td><span class='tag'>{html.escape(row.get('item_id',''))}</span><br>{html.escape(source)}</td>"
                f"<td>{html.escape(row.get('transcript',''))}<br><span class='muted'>{html.escape(row.get('speaker_id',''))}/{html.escape(row.get('utterance_id',''))}</span></td>"
                f"<td>{evidence}</td><td>{audio_cell}</td><td>{fields}</td>"
                "</tr>"
            )
        parts.append("</table>")
    parts.append(
        "<script>"
        f"const BASE_ROWS = {base_rows_json};"
        "const ANNOTATION_FIELDS = ['seems_valid_feedback','false_alarm','alignment_issue','audio_quality_issue','wording_ok','should_allow_user_facing','severity','comment'];"
        "function csvEscape(v){v=(v??'').toString();return /[\",\\n]/.test(v)?'\"'+v.replaceAll('\"','\"\"')+'\"':v;}"
        "function collectRows(){const byId={}; for(const r of BASE_ROWS){byId[r.item_id]={...r}; for(const f of ANNOTATION_FIELDS){if(!(f in byId[r.item_id])) byId[r.item_id][f]='';}}"
        "document.querySelectorAll('[data-item][data-field]').forEach(el=>{const id=el.dataset.item; const f=el.dataset.field; if(byId[id]) byId[id][f]=el.value;}); return Object.values(byId);}"
        "function downloadAnnotations(){const rows=collectRows(); const headers=[]; rows.forEach(r=>Object.keys(r).forEach(k=>{if(!headers.includes(k))headers.push(k);}));"
        "const csv=[headers.join(',')].concat(rows.map(r=>headers.map(h=>csvEscape(r[h])).join(','))).join('\\n');"
        "const blob=new Blob(['\\ufeff'+csv],{type:'text/csv;charset=utf-8'}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='manual_inspection_annotations.csv'; a.click(); URL.revokeObjectURL(a.href);}"
        "function markVisibleUnsure(){document.querySelectorAll('select[data-field]').forEach(s=>{if(!s.value)s.value='unsure';});}"
        "</script></body></html>"
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text("\n".join(parts), encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build special mora manual review template and static HTML viewer")
    parser.add_argument("--items-csv", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "manual_inspection_items.csv")
    parser.add_argument("--annotation-template", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "manual_inspection_annotations_template.csv")
    parser.add_argument("--annotation-v2-template", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "manual_inspection_annotations_v2_template.csv")
    parser.add_argument("--guideline-v2", type=Path, default=ROOT / "reports" / "manual_inspection_guideline_v2.md")
    parser.add_argument("--output-html", type=Path, default=ROOT / "reports" / "special_mora_manual_review_viewer.html")
    parser.add_argument("--refresh-template", action="store_true", help="Overwrite the blank annotation template from current items.")
    parser.add_argument("--no-copy-audio-assets", action="store_true", help="Do not copy review audio next to the HTML.")
    args = parser.parse_args()
    created = build_annotation_template(args.items_csv, args.annotation_template, overwrite=args.refresh_template)
    created_v2 = build_annotation_template_v2(args.items_csv, args.annotation_v2_template, overwrite=args.refresh_template)
    write_guideline_v2(args.guideline_v2)
    count = build_review_viewer(args.items_csv, args.output_html, copy_audio_assets=not args.no_copy_audio_assets)
    print({
        "items": count,
        "template_created": created,
        "template": str(args.annotation_template),
        "template_v2_created": created_v2,
        "template_v2": str(args.annotation_v2_template),
        "guideline_v2": str(args.guideline_v2),
        "html": str(args.output_html),
    })


if __name__ == "__main__":
    main()
