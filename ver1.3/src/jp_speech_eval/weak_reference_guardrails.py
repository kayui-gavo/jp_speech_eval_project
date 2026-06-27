from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Sequence


_LATIN_RE = re.compile(r"[A-Za-z]")
_JA_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff々〆〤ー]")


def _float_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _count_linguistic_chars(text: str) -> tuple[int, int, int]:
    latin = 0
    japanese = 0
    total = 0
    for ch in str(text or ""):
        if ch.isspace() or ch in "、。，．!?！？・「」『』（）()[]":
            continue
        if ch.isalnum() or _JA_RE.match(ch):
            total += 1
            if _LATIN_RE.match(ch):
                latin += 1
            if _JA_RE.match(ch):
                japanese += 1
    return latin, japanese, total


def japanese_likeness(
    *,
    text: str = "",
    kana: str = "",
    moras: Sequence[str] | None = None,
) -> Dict[str, Any]:
    moras = list(moras or [])
    latin, japanese, total = _count_linguistic_chars(text)
    kana_latin, kana_japanese, kana_total = _count_linguistic_chars(kana)
    latin_total = latin + kana_latin
    japanese_total = japanese + kana_japanese + len(moras)
    char_total = total + kana_total + len(moras)
    latin_ratio = latin_total / max(char_total, 1)
    japanese_ratio = japanese_total / max(char_total, 1)
    raw_latin_ratio = latin / max(total, 1)
    raw_japanese_ratio = japanese / max(total, 1)
    raw_latin_dominant = total > 0 and raw_latin_ratio >= 0.50 and raw_japanese_ratio < 0.25
    raw_non_japanese = total > 0 and japanese == 0
    latin_dominant = raw_latin_dominant or (latin_ratio >= 0.50 and japanese_ratio < 0.35)
    return {
        "latin_char_count": latin_total,
        "japanese_char_count": japanese_total,
        "linguistic_char_count": char_total,
        "latin_ratio": round(float(latin_ratio), 4),
        "japanese_ratio": round(float(japanese_ratio), 4),
        "raw_latin_ratio": round(float(raw_latin_ratio), 4),
        "raw_japanese_ratio": round(float(raw_japanese_ratio), 4),
        "raw_latin_dominant": bool(raw_latin_dominant),
        "raw_non_japanese": bool(raw_non_japanese),
        "latin_dominant": bool(latin_dominant),
        "has_kana_or_mora": bool(kana_japanese > 0 or moras),
    }


def apply_weak_overall_guardrail(
    *,
    weak_overall_score: Optional[int],
    target_text: str,
    kana: str,
    moras: Sequence[str],
    duration_sec: Optional[float],
    content_match: Mapping[str, Any] | None,
    weak_prosody_details: Mapping[str, Any] | None,
    mora_evidence_summary: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Apply product guardrails for arbitrary-sentence weak practice scoring.

    This layer is intentionally outside the pitch scoring formula. It prevents
    English/Latin-dominant and insufficient-evidence utterances from surfacing a
    high practice overall built from fluency/rhythm proxies.
    """

    moras = list(moras or [])
    content = content_match or {}
    weak = weak_prosody_details or {}
    evidence = mora_evidence_summary or {}
    score = weak_overall_score
    reasons: list[str] = []
    status = "ok"
    cap: Optional[int] = None
    likeness = japanese_likeness(text=target_text, kana=kana, moras=moras)

    content_status = str(content.get("status") or "unknown")
    if content_status == "fail":
        reasons.append("content_mismatch")

    if likeness["latin_dominant"]:
        reasons.append("latin_dominant_confirmed_text")
    elif likeness["raw_non_japanese"]:
        reasons.append("low_japanese_likeness")
    if not likeness["has_kana_or_mora"] or len(moras) == 0:
        reasons.append("kana_or_mora_unavailable")
    if likeness["japanese_ratio"] < 0.35 and len(moras) < 3:
        reasons.append("low_japanese_likeness")

    mora_count = len(moras)
    duration = _float_or_none(duration_sec)
    f0_coverage = _float_or_none(weak.get("f0_coverage"))
    valid_f0_mora_count = int(weak.get("valid_f0_mora_count") or 0)
    voiced_mora_count = int(weak.get("voiced_mora_count") or valid_f0_mora_count or 0)
    judgement_count = int(evidence.get("judgement_available_count") or 0)

    if mora_count <= 3:
        reasons.append("short_utterance_insufficient_evidence")
    elif mora_count <= 4:
        cap = 70
        reasons.append("short_utterance_practice_cap")
    if duration is not None and duration < 0.55:
        cap = min(cap or 70, 70)
        reasons.append("very_short_duration_practice_cap")
    if weak.get("available") is False or f0_coverage is not None and f0_coverage < 0.50:
        cap = min(cap or 70, 70)
        reasons.append("low_f0_coverage_practice_cap")
    if mora_count >= 4 and valid_f0_mora_count and valid_f0_mora_count < 3:
        cap = min(cap or 70, 70)
        reasons.append("low_valid_f0_mora_practice_cap")
    if mora_count >= 5 and judgement_count and judgement_count < max(3, int(mora_count * 0.40)):
        cap = min(cap or 75, 75)
        reasons.append("low_mora_evidence_practice_cap")

    hard_reasons = {
        "content_mismatch",
        "latin_dominant_confirmed_text",
        "kana_or_mora_unavailable",
        "low_japanese_likeness",
        "short_utterance_insufficient_evidence",
    }
    if any(reason in hard_reasons for reason in reasons):
        status = "no_score"
        score = None
    elif cap is not None:
        status = "capped"
        if score is not None:
            score = min(int(score), int(cap))

    return {
        "status": status,
        "weak_overall_practice_score_before_guardrail": weak_overall_score,
        "weak_overall_practice_score_after_guardrail": score,
        "cap": cap,
        "reasons": sorted(set(reasons)),
        "display_allowed": status != "no_score",
        "japanese_likeness": likeness,
        "mora_count": mora_count,
        "duration_sec": duration,
        "voiced_mora_count": voiced_mora_count,
        "valid_f0_mora_count": valid_f0_mora_count,
        "content_gate_status": content_status,
    }
