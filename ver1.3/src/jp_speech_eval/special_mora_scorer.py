from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping

from .phonology import classify_mora_sequence


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


def score_special_mora_timing(result: Mapping[str, Any]) -> List[SpecialMoraFeedback]:
    moras = [str(m) for m in (result.get("moras") or [])]
    mora_table = result.get("mora_table") or []
    details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
    evidence_rows = details.get("mora_evidence") if isinstance(details.get("mora_evidence"), list) else []
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
        if ph.strength == "none":
            continue
        ev = evidence_rows[i] if i < len(evidence_rows) and isinstance(evidence_rows[i], Mapping) else None
        conf = _confidence(ev)
        if conf == "low" or not ev or not bool(ev.get("judgement_available")):
            out.append(SpecialMoraFeedback(ph.index, ph.mora, ph.mora_type, "uncertain", "low", "今回はこの特殊拍を細かく判定できません。"))
            continue
        ratio = durations[i] / max(avg, 1e-8) if i < len(durations) and avg > 0 else None
        low = 0.45 if ph.strength == "strong" else 0.55
        high = 2.15 if ph.mora_type != "sokuon" else 1.85
        if ratio is None:
            status = "uncertain"
            msg = "今回はこの特殊拍を細かく判定できません。"
        elif ratio < low:
            status = "too_short"
            msg = f"「{ph.mora}」の長さをもう少し保つと自然です。"
        elif ratio > high:
            status = "too_long"
            msg = f"「{ph.mora}」が少し長く聞こえます。"
        else:
            status = "ok"
            msg = f"「{ph.mora}」の長さはおおむね自然です。"
        out.append(SpecialMoraFeedback(ph.index, ph.mora, ph.mora_type, status, conf, msg, None if ratio is None else round(float(ratio), 4)))
    return out
