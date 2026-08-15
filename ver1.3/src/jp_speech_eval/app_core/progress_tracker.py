from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from jp_speech_eval.evaluator import EvaluationResult
from jp_speech_eval.score_contract import comparison_context
from jp_speech_eval.user_score_policy import apply_user_score_policy

from .user_profile import utc_now_iso


@dataclass
class ProgressRecord:
    user_id: str
    item_id: str
    step: int
    target_text: str
    audio_path: str
    # New records store the product-facing four-score contract here. Legacy
    # JSONL records remain readable, but lack score_context and are therefore
    # intentionally blocked from direct progress-delta claims.
    scores: Dict[str, float]
    features: Dict[str, Optional[float]]
    feedback: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    raw_summary: Dict[str, Any] = field(default_factory=dict)
    score_context: Dict[str, Any] = field(default_factory=dict)
    legacy_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _legacy_scores_from_result(result: EvaluationResult) -> Dict[str, float]:
    return {
        "total": float(result.total_score),
        "pronunciation": float(result.pronunciation_score),
        "prosody": float(result.prosody_score),
        "fluency": float(result.fluency_score),
        "expression": float(result.tone_score),
    }


def _effective_mode(result: EvaluationResult) -> str:
    details = result.details or {}
    return str(details.get("mode") or "fixed_reference")


def _consumer_scores_from_result(result: EvaluationResult) -> tuple[Dict[str, float], Dict[str, Any]]:
    raw = result.to_dict()
    mode = _effective_mode(result)
    policy = apply_user_score_policy(raw, mode=mode)
    components = policy.get("component_scores") or {}
    scores: Dict[str, float] = {}
    mapping = {
        "clarity": "clarity",
        "rhythm": "mora_timing",
        "fluency": "delivery_fluency",
        "intonation": "intonation",
    }
    total = _to_float(policy.get("display_score"))
    if total is not None:
        scores["total"] = total
    for public_key, component_key in mapping.items():
        item = components.get(component_key) if isinstance(components.get(component_key), dict) else {}
        value = _to_float(item.get("value"))
        if value is not None:
            scores[public_key] = value
    context = comparison_context(raw, mode=mode)
    context.update({
        "score_available": bool(policy.get("score_available")),
        "confidence_label": str(policy.get("confidence_label") or "unknown"),
        "component_confidence": {
            public_key: str((components.get(component_key) or {}).get("confidence") or "unknown")
            for public_key, component_key in mapping.items()
        },
        "component_evidence_tier": {
            public_key: str((components.get(component_key) or {}).get("evidence_tier") or "unknown")
            for public_key, component_key in mapping.items()
        },
    })
    return scores, context


def _features_from_result(result: EvaluationResult) -> Dict[str, Optional[float]]:
    details = result.details or {}
    fluency = details.get("fluency") or {}
    tone = details.get("tone") or {}
    reliability = details.get("reliability") or {}
    return {
        "mora_rate": _to_float(fluency.get("speech_rate_mora_per_sec")),
        "avg_mora_duration_sec": _to_float(fluency.get("avg_mora_duration_sec")),
        "f0_range_log": _to_float(tone.get("pitch_range_log")),
        "pause_ratio": _to_float((result.pause_info or {}).get("pause_ratio")),
        "reliability_overall": _to_float(reliability.get("overall")),
    }


def record_from_evaluation(
    *,
    user_id: str,
    item_id: str,
    step: int,
    audio_path: str | Path,
    result: EvaluationResult,
    extra_feedback: Optional[Iterable[str]] = None,
) -> ProgressRecord:
    feedback = list(result.feedback)
    if extra_feedback:
        feedback.extend(str(item) for item in extra_feedback if str(item).strip())
    consumer_scores, score_context = _consumer_scores_from_result(result)
    return ProgressRecord(
        user_id=user_id,
        item_id=item_id,
        step=int(step),
        target_text=result.target_text,
        audio_path=str(audio_path),
        scores=consumer_scores,
        features=_features_from_result(result),
        feedback=feedback,
        score_context=score_context,
        legacy_scores=_legacy_scores_from_result(result),
        raw_summary={
            "kana": result.kana,
            "mora_count": len(result.moras),
            "alignment_mode": result.alignment_mode,
            "cache_prefix": result.cache_prefix,
            "timing": result.timing,
            "reliability": (result.details or {}).get("reliability") or {},
        },
    )


def append_progress_record(path: str | Path, record: ProgressRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def load_progress_records(
    path: str | Path,
    *,
    user_id: Optional[str] = None,
    item_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if user_id is not None and row.get("user_id") != user_id:
                continue
            if item_id is not None and row.get("item_id") != item_id:
                continue
            rows.append(row)
    return rows


def latest_record(
    records: Iterable[Dict[str, Any]],
    *,
    item_id: Optional[str] = None,
    step: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    filtered = []
    for record in records:
        if item_id is not None and record.get("item_id") != item_id:
            continue
        if step is not None and int(record.get("step", -1)) != int(step):
            continue
        filtered.append(record)
    return filtered[-1] if filtered else None
