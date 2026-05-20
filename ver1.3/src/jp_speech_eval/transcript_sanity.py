from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable


_JA_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff々〆〤ー]")
_CONTENT_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff々〆〤A-Za-z0-9]")
_PUNCT_SPACE_RE = re.compile(r"[\s、。！？!?,.，．・…ー〜~\-]+")


@dataclass(frozen=True)
class TranscriptSanityResult:
    """Lightweight gate before creating an ASR-generated pseudo-reference."""

    ok: bool
    score: float
    reason: str
    normalized_text: str
    metrics: Dict[str, float | int | str]

    def to_dict(self) -> Dict:
        return asdict(self)


def _longest_run_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 1.0
    longest = 1
    current = 1
    for prev, cur in zip(chars, chars[1:]):
        if cur == prev:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest / max(len(chars), 1)


def _unique_ratio(text: str) -> float:
    chars = [ch for ch in text if _CONTENT_CHAR_RE.match(ch)]
    if not chars:
        return 0.0
    return len(set(chars)) / len(chars)


def _contains_any(text: str, items: Iterable[str]) -> bool:
    return any(item in text for item in items)


def check_asr_transcript_sanity(text: str, *, min_chars: int = 3, max_chars: int = 80) -> TranscriptSanityResult:
    """Reject ASR text that is too implausible to synthesize and score.

    This is intentionally lightweight. It prevents obvious failure cases such
    as shouting/noise being hallucinated into a short or repetitive transcript.
    It is not a semantic judge and should later be replaceable by a compact
    Japanese language-model acceptability gate.
    """

    normalized = re.sub(r"\s+", "", str(text or "").strip())
    content_chars = _CONTENT_CHAR_RE.findall(normalized)
    ja_chars = _JA_CHAR_RE.findall(normalized)
    content_len = len(content_chars)
    ja_ratio = len(ja_chars) / max(content_len, 1)
    punct_ratio = len(_PUNCT_SPACE_RE.findall(normalized)) / max(len(normalized), 1)
    run_ratio = _longest_run_ratio(normalized)
    uniq_ratio = _unique_ratio(normalized)

    metrics: Dict[str, float | int | str] = {
        "char_count": len(normalized),
        "content_char_count": content_len,
        "ja_ratio": round(float(ja_ratio), 4),
        "punct_ratio": round(float(punct_ratio), 4),
        "longest_run_ratio": round(float(run_ratio), 4),
        "unique_content_ratio": round(float(uniq_ratio), 4),
    }

    reason = "ok"
    score = 1.0
    ok = True
    if content_len < min_chars:
        ok = False
        reason = "too_short_for_pseudo_reference"
        score = 0.05
    elif content_len > max_chars:
        ok = False
        reason = "too_long_for_realtime_pseudo_reference"
        score = 0.25
    elif ja_ratio < 0.55:
        ok = False
        reason = "not_enough_japanese_content"
        score = 0.20
    elif run_ratio >= 0.45:
        ok = False
        reason = "repetitive_or_shouted_transcript"
        score = 0.20
    elif uniq_ratio < 0.18 and content_len >= 8:
        ok = False
        reason = "low_information_repetition"
        score = 0.25
    elif punct_ratio > 0.45:
        ok = False
        reason = "mostly_punctuation_or_fillers"
        score = 0.20
    elif _contains_any(normalized.lower(), {"www", "ahaha", "哈哈", "呵呵"}):
        ok = False
        reason = "laughter_or_noise_like_transcript"
        score = 0.20

    return TranscriptSanityResult(
        ok=ok,
        score=round(float(score), 4),
        reason=reason,
        normalized_text=normalized,
        metrics=metrics,
    )
