from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .unified_result import ANNOTATION_FIELDS, UnifiedEvaluationResult


def append_jsonl(path: str | Path, result: UnifiedEvaluationResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_log_record(), ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def _scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def flatten_log_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    input_info = record.get("input_info") or {}
    features = record.get("features") or {}
    scores = record.get("scores") or {}
    reliability = record.get("reliability") or {}

    row: Dict[str, Any] = {
        "mode": record.get("mode"),
        "audio_path": input_info.get("audio_path"),
        "target_text": input_info.get("target_text"),
        "asr_transcript": input_info.get("asr_transcript"),
        "kana": input_info.get("kana"),
        "input_mora_count": input_info.get("mora_count"),
        "latency_ms": record.get("latency_ms"),
        "reliability_overall": reliability.get("overall"),
        "reliability_level": reliability.get("level"),
        "reliability_endpointing": reliability.get("endpointing"),
        "reliability_alignment": reliability.get("alignment"),
        "reliability_f0_coverage": reliability.get("f0_coverage"),
        "score_total": scores.get("total"),
        "score_pronunciation": scores.get("pronunciation"),
        "score_prosody": scores.get("prosody"),
        "score_fluency": scores.get("fluency"),
        "score_expression": scores.get("expression"),
        "warnings": " | ".join(str(x) for x in (record.get("warnings") or [])),
        "feedback": " | ".join(str(x) for x in (record.get("feedback") or [])),
    }
    for key, value in features.items():
        row[f"feature_{key}"] = _scalar(value)
    for field_name in ANNOTATION_FIELDS:
        row[field_name] = record.get(field_name)
    return row


def export_feature_table(jsonl_paths: Iterable[str | Path], csv_path: str | Path) -> int:
    records = load_jsonl(jsonl_paths)
    rows = [flatten_log_record(record) for record in records]
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return 0

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
