from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .phonology import classify_mora_sequence


DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "long_vowel": {"low_ratio": 0.45, "high_ratio": 2.15, "min_evidence_confidence": 0.45},
    "sokuon": {"low_ratio": 0.45, "high_ratio": 1.85, "min_evidence_confidence": 0.45},
    "moraic_nasal": {"low_ratio": 0.45, "high_ratio": 2.15, "min_evidence_confidence": 0.45},
    "yoon": {"low_ratio": 0.45, "high_ratio": 1.85, "min_evidence_confidence": 0.45},
    "vowel_lengthening_candidate": {"low_ratio": 0.55, "high_ratio": 2.15, "min_evidence_confidence": 0.45},
}


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


def load_special_mora_thresholds(path: str | Path | None = None) -> Dict[str, Dict[str, float]]:
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
    return thresholds


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
