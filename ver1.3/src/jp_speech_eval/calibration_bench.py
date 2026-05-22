from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .evaluator import EvaluationResult


DEFAULT_NUMERIC_FIELDS = [
    "score_total",
    "score_pronunciation",
    "score_prosody",
    "score_fluency",
    "score_expression",
    "reliability_overall",
    "f0_coverage",
    "speech_rate_mora_per_sec",
    "avg_mora_duration_sec",
    "mora_duration_cv",
    "special_mora_penalty",
    "strong_special_mora_count",
    "weak_special_mora_count",
    "contour_corr",
    "transition_agreement",
    "pause_ratio",
]


def read_manifest(path: str | Path) -> List[Dict[str, str]]:
    """Read a dataset audit manifest.

    Expected columns:
      audio_path,text,dataset,split,speaker_id,cache_path

    Only `audio_path` and either `text` or `cache_path` are required. `dataset`
    should usually be values such as `jvs` or `janon`, while `split` can be
    `native`, `l2`, `train`, or any label useful for audit grouping.
    """

    rows: List[Dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {str(k): str(v).strip() for k, v in row.items() if k is not None and v is not None}
            if clean.get("audio_path") and (clean.get("text") or clean.get("cache_path")):
                rows.append(clean)
    return rows


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def evaluation_to_audit_row(manifest_row: Dict[str, str], result: EvaluationResult) -> Dict[str, Any]:
    details = result.details or {}
    reliability = details.get("reliability") or {}
    fluency = details.get("fluency") or {}
    pronunciation = details.get("pronunciation") or {}
    prosody = details.get("prosody") or {}
    evidence = details.get("mora_evidence_summary") or {}

    return {
        "dataset": manifest_row.get("dataset") or "unknown",
        "split": manifest_row.get("split") or manifest_row.get("speaker_type") or "unknown",
        "speaker_id": manifest_row.get("speaker_id") or "",
        "utterance_id": manifest_row.get("utterance_id") or Path(manifest_row.get("audio_path", "")).stem,
        "audio_path": manifest_row.get("audio_path") or "",
        "target_text": result.target_text,
        "kana": result.kana,
        "mora_count": len(result.moras),
        "alignment_mode": result.alignment_mode,
        "cache_prefix": result.cache_prefix,
        "score_total": float(result.total_score),
        "score_pronunciation": float(result.pronunciation_score),
        "score_prosody": float(result.prosody_score),
        "score_fluency": float(result.fluency_score),
        "score_expression": float(result.tone_score),
        "reliability_overall": _safe_float(reliability.get("overall")),
        "reliability_level": reliability.get("level"),
        "score_is_diagnostic": bool(reliability.get("score_is_diagnostic")),
        "f0_coverage": _safe_float(reliability.get("f0_coverage")),
        "speech_rate_mora_per_sec": _safe_float(fluency.get("speech_rate_mora_per_sec")),
        "avg_mora_duration_sec": _safe_float(fluency.get("avg_mora_duration_sec")),
        "mora_duration_cv": _safe_float(pronunciation.get("mora_duration_cv")),
        "special_mora_penalty": _safe_float(pronunciation.get("special_mora_penalty")),
        "strong_special_mora_count": int(evidence.get("strong_special_mora_count", evidence.get("special_mora_count", 0)) or 0),
        "weak_special_mora_count": int(evidence.get("weak_special_mora_count", 0) or 0),
        "special_mora_judgement_available_count": int(evidence.get("special_mora_judgement_available_count", 0) or 0),
        "contour_corr": _safe_float(prosody.get("contour_corr")),
        "transition_agreement": _safe_float(prosody.get("transition_agreement")),
        "accent_drop_agreement": _safe_float(prosody.get("accent_drop_agreement")),
        "pause_ratio": _safe_float((result.pause_info or {}).get("pause_ratio")),
        "feedback": " | ".join(str(item) for item in result.feedback),
        "warnings": " | ".join(str(item) for item in reliability.get("warnings", [])),
    }


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
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


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def summarize_audit_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    group_fields: Sequence[str] = ("dataset", "split"),
    numeric_fields: Sequence[str] = tuple(DEFAULT_NUMERIC_FIELDS),
) -> List[Dict[str, Any]]:
    """Summarize native/L2 audit rows with robust distribution statistics."""

    groups: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        groups.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        out: Dict[str, Any] = {field: value for field, value in zip(group_fields, key)}
        out["n_utterances"] = len(group)
        out["diagnostic_rate"] = round(
            float(np.mean([bool(row.get("score_is_diagnostic")) for row in group])),
            6,
        )
        for field in numeric_fields:
            values = np.asarray([
                value for row in group
                if (value := _safe_float(row.get(field))) is not None
            ], dtype=float)
            out[f"{field}_n"] = int(values.size)
            if values.size:
                out[f"{field}_mean"] = round(float(np.mean(values)), 6)
                out[f"{field}_p10"] = round(float(np.percentile(values, 10)), 6)
                out[f"{field}_p50"] = round(float(np.percentile(values, 50)), 6)
                out[f"{field}_p90"] = round(float(np.percentile(values, 90)), 6)
            else:
                out[f"{field}_mean"] = None
                out[f"{field}_p10"] = None
                out[f"{field}_p50"] = None
                out[f"{field}_p90"] = None
        summary_rows.append(out)
    return summary_rows
