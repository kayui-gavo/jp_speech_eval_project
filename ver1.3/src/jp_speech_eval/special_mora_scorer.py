from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .phonology import classify_mora_sequence


DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "long_vowel": {"low_ratio": 0.45, "high_ratio": 2.15, "min_evidence_confidence": 0.45},
    "sokuon": {"low_ratio": 0.45, "high_ratio": 1.85, "min_evidence_confidence": 0.45},
    "moraic_nasal": {"low_ratio": 0.45, "high_ratio": 2.15, "min_evidence_confidence": 0.45},
    "yoon": {"low_ratio": 0.45, "high_ratio": 1.85, "min_evidence_confidence": 0.45},
    "vowel_lengthening_candidate": {"low_ratio": 0.55, "high_ratio": 2.15, "min_evidence_confidence": 0.45},
}

RUNTIME_THRESHOLD_STATUSES = {"active", "tentative", "insufficient", "debug_only"}
USER_FEEDBACK_TYPES = {"long_vowel", "moraic_nasal"}
SEVERE_MAPPING_WARNINGS = {
    "phone_mora_count_mismatch",
    "not_enough_phone_segments",
    "missing_mora_segments",
    "mapping_failed",
}


def default_special_mora_threshold_path() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "calibration" / "special_mora_thresholds.json"


@dataclass(frozen=True)
class SpecialMoraFeedback:
    index: int
    mora: str
    type: str
    status: str
    confidence: str
    message: str
    duration_ratio: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeSpecialMoraDecision:
    type: str
    surface_mora: str
    mora_index: int
    phone_sequence_for_mora: str
    feature_name: str
    feature_value: Optional[float]
    threshold_low: Optional[float]
    threshold_high: Optional[float]
    threshold_status: str
    decision: str
    evidence_confidence: float
    confidence: str
    alignment_method: str
    mapping_success: bool
    mapping_warning_flags: List[str]
    user_feedback_allowed: bool
    suppression_reason: str
    feedback_candidate_text: str
    threshold_metadata_warning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _confidence(row: Mapping[str, Any] | None) -> str:
    if not row:
        return "low"
    boundary = float(row.get("boundary_confidence", 0.0) or 0.0)
    energy = float(row.get("energy_coverage", 0.0) or 0.0)
    if boundary >= 0.70 and energy >= 0.40:
        return "high"
    if boundary >= 0.45 and energy >= 0.20:
        return "medium"
    return "low"


def _confidence_score(row: Mapping[str, Any] | None) -> float:
    if not row:
        return 0.0
    boundary = float(row.get("boundary_confidence", 0.0) or 0.0)
    energy = float(row.get("energy_coverage", 0.0) or 0.0)
    return max(0.0, min(1.0, 0.7 * boundary + 0.3 * energy))


def _canonical_type(mora_type: str, mora: str) -> str:
    if mora_type == "explicit_long_vowel":
        return "long_vowel"
    if mora_type == "nasal":
        return "moraic_nasal"
    if any(ch in mora for ch in "ャュョゃゅょ"):
        return "yoon"
    return mora_type


def load_special_mora_thresholds(path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Load conservative special-mora timing thresholds.

    The JSON is expected to contain `thresholds` keyed by canonical special
    mora type. Missing or insufficient entries fall back to defaults.
    """

    thresholds = {key: dict(value) for key, value in DEFAULT_THRESHOLDS.items()}
    if path is None:
        return thresholds
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("thresholds", data)
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        entry = thresholds.setdefault(str(key), {})
        for field in ("low_ratio", "high_ratio", "min_evidence_confidence"):
            if field in value and value[field] is not None:
                entry[field] = float(value[field])
        for field in (
            "type",
            "status",
            "feature_name",
            "feature_definition",
            "denominator",
            "source_dataset",
            "alignment_backend",
            "sample_count",
            "reliable_count",
            "percentile_used",
            "generated_at",
            "warnings",
            "warning",
        ):
            if field in value:
                entry[field] = value[field]
    return thresholds


def load_runtime_special_mora_thresholds(path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Load calibrated runtime threshold metadata without heuristic fallback.

    Missing or malformed threshold metadata is represented explicitly so the
    runtime scorer can return `uncertain` and keep user-facing feedback off.
    """

    threshold_path = Path(path) if path is not None else default_special_mora_threshold_path()
    invalid = {
        key: {
            "type": key,
            "status": "invalid",
            "low_ratio": None,
            "high_ratio": None,
            "feature_name": "",
            "feature_definition": "",
            "denominator": "",
            "warnings": ["missing_or_invalid_threshold_metadata"],
        }
        for key in DEFAULT_THRESHOLDS
    }
    if not threshold_path.exists():
        return invalid
    try:
        data = json.loads(threshold_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return invalid
    raw = data.get("thresholds", data)
    if not isinstance(raw, Mapping):
        return invalid
    out = {key: dict(value) for key, value in invalid.items()}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        status = str(value.get("status") or "")
        low = value.get("low_ratio")
        high = value.get("high_ratio")
        entry = {
            "type": str(value.get("type") or key),
            "status": status if status in RUNTIME_THRESHOLD_STATUSES else "invalid",
            "low_ratio": float(low) if low is not None else None,
            "high_ratio": float(high) if high is not None else None,
            "feature_name": str(value.get("feature_name") or ""),
            "feature_definition": str(value.get("feature_definition") or ""),
            "denominator": str(value.get("denominator") or ""),
            "source_dataset": value.get("source_dataset"),
            "alignment_backend": value.get("alignment_backend"),
            "sample_count": value.get("sample_count", value.get("coverage_count")),
            "reliable_count": value.get("reliable_count"),
            "percentile_used": value.get("percentile_used"),
            "generated_at": value.get("generated_at", data.get("generated_at")),
            "warnings": list(value.get("warnings") or ([] if not value.get("warning") else [value.get("warning")])),
        }
        if entry["status"] == "active" and (entry["low_ratio"] is None or entry["high_ratio"] is None):
            entry["status"] = "invalid"
            entry["warnings"].append("missing_or_invalid_threshold_metadata")
        out[str(key)] = entry
    return out


def _alignment_method(result: Mapping[str, Any], details: Mapping[str, Any]) -> str:
    alignment_mode = str(result.get("alignment_mode") or details.get("alignment_mode") or "")
    if not alignment_mode:
        alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}
        alignment_mode = str(alignment.get("mode") or "")
    return alignment_mode


def _feature_value(special_type: str, duration: float, avg: float, prev_duration: Optional[float], next_duration: Optional[float]) -> Optional[float]:
    if duration <= 0:
        return None
    if special_type == "sokuon":
        neighbors = [v for v in (prev_duration, next_duration) if v is not None and v > 0]
        denom = sum(neighbors) / len(neighbors) if neighbors else avg
    else:
        denom = avg
    if denom <= 0:
        return None
    return float(duration / max(denom, 1e-8))


def _candidate_text(mora: str, special_type: str, decision: str, weak_reference: bool) -> str:
    if decision not in {"too_short", "too_long"}:
        return ""
    if special_type == "long_vowel":
        core = f"「{mora}」をもう少し伸ばすと自然です。" if decision == "too_short" else f"「{mora}」が少し長く聞こえます。"
    elif special_type == "moraic_nasal":
        core = f"「{mora}」を少し残して言うと聞き取りやすくなります。" if decision == "too_short" else f"「{mora}」が少し長く聞こえます。"
    else:
        return ""
    if weak_reference:
        return "もしその表現を言いたい場合は，" + core
    return core


def decide_special_mora_feature_value(threshold: Mapping[str, Any], feature_value: Optional[float]) -> str:
    """Apply the same calibrated threshold decision used by runtime shadow.

    This helper exists so validation scripts can test counterfactual feature
    perturbations without duplicating threshold logic.
    """

    status = str(threshold.get("status") or "invalid")
    low = threshold.get("low_ratio")
    high = threshold.get("high_ratio")
    if status != "active" or feature_value is None or low is None or high is None:
        return "uncertain"
    value = float(feature_value)
    if value < float(low):
        return "too_short"
    if value > float(high):
        return "too_long"
    return "ok"


def _suppression_reason(
    *,
    threshold_status: str,
    alignment_method: str,
    evidence_confidence: float,
    min_confidence: float,
    judgement_available: bool,
    mapping_success: bool,
    mapping_warning_flags: List[str],
    mora_count: int,
    reliability: Mapping[str, Any],
    weak_reference: bool,
    special_type: str,
    decision: str,
    enable_user_facing: bool,
) -> str:
    if threshold_status in {"invalid", ""}:
        return "missing_or_invalid_threshold_metadata"
    if threshold_status == "insufficient":
        return "insufficient_native_evidence"
    if threshold_status == "debug_only":
        return "debug_only_threshold"
    if threshold_status == "tentative":
        return "tentative_threshold_debug_only"
    if not enable_user_facing:
        return "shadow_mode_user_facing_disabled"
    if special_type not in USER_FEEDBACK_TYPES:
        return "special_mora_type_not_user_facing"
    if decision not in {"too_short", "too_long"}:
        return "no_correction_needed"
    if alignment_method.endswith("fallback_equal") or alignment_method == "equal_fallback":
        return "fallback_alignment"
    if not judgement_available:
        return "insufficient_mora_evidence"
    if evidence_confidence < min_confidence:
        return "evidence_confidence_low"
    if not mapping_success:
        return "mapping_failed"
    if SEVERE_MAPPING_WARNINGS.intersection(mapping_warning_flags):
        return "severe_mapping_warning"
    if mora_count <= 3:
        return "short_utterance"
    if str(reliability.get("level") or "") == "low" or float(reliability.get("overall", 1.0) or 1.0) < 0.45:
        return "reliability_gate_low"
    if weak_reference and evidence_confidence < 0.75:
        return "weak_reference_requires_high_evidence"
    return ""


def decide_special_mora_runtime(
    result: Mapping[str, Any],
    *,
    threshold_path: str | Path | None = None,
    weak_reference: bool | None = None,
    enable_runtime_shadow: bool = True,
    enable_user_facing: bool = False,
) -> List[RuntimeSpecialMoraDecision]:
    """Compute calibrated special-mora decisions for debug/candidate use.

    This intentionally separates acoustic evidence from product display. The
    returned decisions can be logged even when `user_feedback_allowed` is false.
    """

    if not enable_runtime_shadow:
        return []
    moras = [str(m) for m in (result.get("moras") or [])]
    mora_table = result.get("mora_table") or []
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    evidence_rows = details.get("mora_evidence") if isinstance(details.get("mora_evidence"), list) else []
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    alignment_method = _alignment_method(result, details)
    if weak_reference is None:
        weak_reference = bool(details.get("weak_reference"))
    thresholds = load_runtime_special_mora_thresholds(threshold_path)
    phonology = classify_mora_sequence(moras)
    durations: List[float] = []
    for row in mora_table:
        if isinstance(row, Mapping):
            start = float(row.get("start_sec", 0.0) or 0.0)
            end = float(row.get("end_sec", 0.0) or 0.0)
        else:
            start = float(getattr(row, "start_sec", 0.0) or 0.0)
            end = float(getattr(row, "end_sec", 0.0) or 0.0)
        durations.append(max(0.0, end - start))
    avg = sum(durations) / len(durations) if durations else 0.0
    decisions: List[RuntimeSpecialMoraDecision] = []
    for i, ph in enumerate(phonology):
        special_type = _canonical_type(ph.mora_type, ph.mora)
        if ph.strength == "none" and special_type != "yoon":
            continue
        ev = evidence_rows[i] if i < len(evidence_rows) and isinstance(evidence_rows[i], Mapping) else {}
        duration = durations[i] if i < len(durations) else 0.0
        prev_duration = durations[i - 1] if i > 0 and i - 1 < len(durations) else None
        next_duration = durations[i + 1] if i + 1 < len(durations) else None
        feature_value = _feature_value(special_type, duration, avg, prev_duration, next_duration)
        threshold = thresholds.get(special_type, {})
        status = str(threshold.get("status") or "invalid")
        low = threshold.get("low_ratio")
        high = threshold.get("high_ratio")
        decision = decide_special_mora_feature_value(threshold, feature_value)
        min_conf = float(threshold.get("min_evidence_confidence", 0.45) or 0.45)
        evidence_confidence = _confidence_score(ev)
        confidence = _confidence(ev)
        mapping_success = bool(ev.get("mapping_success", True))
        raw_warnings = ev.get("mapping_warning_flags", ev.get("warnings", []))
        if isinstance(raw_warnings, str):
            mapping_warning_flags = [x for x in raw_warnings.split("|") if x]
        else:
            mapping_warning_flags = [str(x) for x in (raw_warnings or [])]
        suppression = _suppression_reason(
            threshold_status=status,
            alignment_method=alignment_method,
            evidence_confidence=evidence_confidence,
            min_confidence=min_conf,
            judgement_available=bool(ev.get("judgement_available")),
            mapping_success=mapping_success,
            mapping_warning_flags=mapping_warning_flags,
            mora_count=len(moras),
            reliability=reliability,
            weak_reference=bool(weak_reference),
            special_type=special_type,
            decision=decision,
            enable_user_facing=enable_user_facing,
        )
        candidate = _candidate_text(ph.mora, special_type, decision, bool(weak_reference))
        decisions.append(RuntimeSpecialMoraDecision(
            type=special_type,
            surface_mora=ph.mora,
            mora_index=ph.index,
            phone_sequence_for_mora=str(ev.get("phone_sequence_for_mora") or ""),
            feature_name=str(threshold.get("feature_name") or ""),
            feature_value=None if feature_value is None else round(float(feature_value), 4),
            threshold_low=None if low is None else float(low),
            threshold_high=None if high is None else float(high),
            threshold_status=status,
            decision=decision,
            evidence_confidence=round(float(evidence_confidence), 4),
            confidence=confidence,
            alignment_method=alignment_method,
            mapping_success=mapping_success,
            mapping_warning_flags=mapping_warning_flags,
            user_feedback_allowed=not suppression,
            suppression_reason=suppression,
            feedback_candidate_text=candidate,
            threshold_metadata_warning="|".join(str(x) for x in (threshold.get("warnings") or [])),
        ))
    return decisions


def special_mora_score_from_decisions(decisions: List[RuntimeSpecialMoraDecision]) -> Optional[float]:
    judged = [d for d in decisions if d.threshold_status == "active" and d.decision in {"ok", "too_short", "too_long"} and d.evidence_confidence >= 0.45]
    if not judged:
        return None
    bad = sum(1 for d in judged if d.decision in {"too_short", "too_long"})
    return round(max(0.0, 100.0 - 22.0 * bad), 2)


def select_special_mora_feedback_candidate(decisions: List[RuntimeSpecialMoraDecision]) -> Optional[RuntimeSpecialMoraDecision]:
    priority = {"long_vowel": 3, "moraic_nasal": 2, "sokuon": 1, "yoon": 0}
    candidates = [
        item for item in decisions
        if item.user_feedback_allowed and item.decision in {"too_short", "too_long"} and item.feedback_candidate_text
    ]
    candidates.sort(key=lambda item: (priority.get(item.type, 0), item.evidence_confidence), reverse=True)
    return candidates[0] if candidates else None


def score_special_mora_timing(
    result: Mapping[str, Any],
    *,
    threshold_path: str | Path | None = None,
    weak_reference: bool | None = None,
) -> List[SpecialMoraFeedback]:
    moras = [str(m) for m in (result.get("moras") or [])]
    mora_table = result.get("mora_table") or []
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    evidence_rows = details.get("mora_evidence") if isinstance(details.get("mora_evidence"), list) else []
    reliability = details.get("reliability") if isinstance(details.get("reliability"), Mapping) else {}
    alignment_mode = str(result.get("alignment_mode") or details.get("alignment_mode") or "")
    if not alignment_mode:
        alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}
        alignment_mode = str(alignment.get("mode") or "")
    if weak_reference is None:
        weak_reference = bool(details.get("weak_reference"))
    thresholds = load_special_mora_thresholds(threshold_path)
    phonology = classify_mora_sequence(moras)
    durations = []
    for row in mora_table:
        if isinstance(row, Mapping):
            start = float(row.get("start_sec", 0.0) or 0.0)
            end = float(row.get("end_sec", 0.0) or 0.0)
        else:
            start = float(getattr(row, "start_sec", 0.0) or 0.0)
            end = float(getattr(row, "end_sec", 0.0) or 0.0)
        durations.append(max(0.0, end - start))
    avg = sum(durations) / len(durations) if durations else 0.0
    out: List[SpecialMoraFeedback] = []
    for i, ph in enumerate(phonology):
        special_type = _canonical_type(ph.mora_type, ph.mora)
        if ph.strength == "none" and special_type != "yoon":
            continue
        ev = evidence_rows[i] if i < len(evidence_rows) and isinstance(evidence_rows[i], Mapping) else None
        conf = _confidence(ev)
        conf_score = _confidence_score(ev)
        min_conf = float(thresholds.get(special_type, {}).get("min_evidence_confidence", 0.45))
        gate_blocked = (
            alignment_mode.endswith("fallback_equal")
            or alignment_mode == "equal_fallback"
            or str(reliability.get("level") or "") == "low"
            or float(reliability.get("overall", 1.0) or 1.0) < 0.45
            or len(moras) <= 3
        )
        if gate_blocked or conf_score < min_conf or not ev or not bool(ev.get("judgement_available")):
            out.append(SpecialMoraFeedback(ph.index, ph.mora, special_type, "uncertain", conf, "今回はこの特殊拍を細かく判定できません。"))
            continue
        ratio = durations[i] / max(avg, 1e-8) if i < len(durations) and avg > 0 else None
        low = float(thresholds.get(special_type, {}).get("low_ratio", 0.45 if ph.strength == "strong" else 0.55))
        high = float(thresholds.get(special_type, {}).get("high_ratio", 1.85 if special_type in {"sokuon", "yoon"} else 2.15))
        if ratio is None:
            status = "uncertain"
            msg = "今回はこの特殊拍を細かく判定できません。"
        elif ratio < low:
            status = "too_short"
            if special_type == "sokuon":
                msg = f"「{ph.mora}」で一瞬止める感じを出すと自然です。"
            elif special_type == "moraic_nasal":
                msg = f"「{ph.mora}」を少し残して言うと聞き取りやすくなります。"
            elif special_type == "yoon":
                msg = f"「{ph.mora}」は一つのまとまりとして、短くなめらかに言うと自然です。"
            else:
                msg = f"「{ph.mora}」の長さをもう少し保つと自然です。"
        elif ratio > high:
            status = "too_long"
            msg = f"「{ph.mora}」が少し長く聞こえます。"
        else:
            status = "ok"
            msg = f"「{ph.mora}」の長さはおおむね自然です。"
        if weak_reference and status in {"too_short", "too_long"}:
            msg = "参考として見ると、" + msg
        out.append(SpecialMoraFeedback(ph.index, ph.mora, special_type, status, conf, msg, None if ratio is None else round(float(ratio), 4)))
    return out
