"""Evidence-only spontaneous utterance fluency features.

The C-end fluency score is intentionally *not* remapped in this module.
It exposes a Tavakoli/Skehan-style decomposition into speed, breakdown and
repair evidence so a later human-criterion study can decide what belongs in the
product score.

Japanese adaptation notes
-------------------------
The classical literature usually reports syllables/second.  This project uses
mora count because its Japanese frontend is mora based.  The resulting rates
must therefore be treated as Japanese product/research features, not as
numerically interchangeable with syllable-rate thresholds from English L2
studies.
"""

from __future__ import annotations

import re
import unicodedata
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SCHEMA_VERSION = "spontaneous_fluency_evidence_v1"

# Conservative, fairly lexicalized Japanese filler forms.  We deliberately do
# not count あの/その/まあ/なんか as certain fillers because they also have
# ordinary lexical/discourse uses.  Those forms are reported separately.
_HIGH_PRECISION_FILLER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("eeto", re.compile(r"え(?:ー|え)*と")),
    ("etto", re.compile(r"えっ+と")),
    ("uun", re.compile(r"う(?:ー|う)+ん")),
    ("nn", re.compile(r"んー+")),
)
_AMBIGUOUS_DISCOURSE_MARKERS = ("あの", "その", "まあ", "なんか")
_AMBIGUOUS_REPAIR_MARKERS = ("いや", "というか", "じゃなくて", "じゃなく", "違う")


def _finite_nonnegative(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number < 0 or number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _normalize_transcript(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).strip()


def _frontend_tokens(text: str) -> List[str]:
    """Best-effort surface tokens used only for conservative repair candidates."""
    if not text:
        return []
    try:
        from .text_frontend import run_frontend

        out: List[str] = []
        for item in run_frontend(text):
            token = str(item.get("string") or item.get("orig") or "").strip()
            if token:
                out.append(token)
        return out
    except Exception:
        # The repair channel is shadow-only; text-frontend failure should not
        # make the utterance unavailable or lower a learner score.
        return []


def speed_fluency_features(
    *,
    mora_count: int,
    speech_duration_sec: float,
    silent_pause_total_sec: float,
) -> Dict[str, Any]:
    """Separate composite speech rate from pause-excluding articulation rate."""
    duration = max(_finite_nonnegative(speech_duration_sec), 1e-6)
    pause_total = min(duration, _finite_nonnegative(silent_pause_total_sec))
    phonation_time = max(duration - pause_total, 1e-6)
    count = max(0, int(mora_count))
    return {
        "mora_count": count,
        "speech_duration_sec": round(duration, 6),
        "phonation_time_sec": round(phonation_time, 6),
        "phonation_time_ratio": round(phonation_time / duration, 6),
        "speech_rate_mora_per_sec": None if count <= 0 else round(count / duration, 6),
        "articulation_rate_mora_per_sec": None if count <= 0 else round(count / phonation_time, 6),
        "mean_mora_duration_during_phonation_sec": None if count <= 0 else round(phonation_time / count, 6),
        "rate_unit_note": "Japanese mora rate; do not reuse English syllable-rate cutoffs directly",
    }


def breakdown_fluency_features(
    pause_info: Mapping[str, Any],
    *,
    speech_duration_sec: float,
    mora_count: int,
    silent_pause_threshold_sec: float = 0.30,
) -> Dict[str, Any]:
    """Describe long silent-pause breakdown without inventing clause location."""
    duration = max(_finite_nonnegative(speech_duration_sec), 1e-6)
    raw_segments = pause_info.get("pause_segments") if isinstance(pause_info, Mapping) else []
    segments: List[tuple[float, float]] = []
    for item in raw_segments or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        start = _finite_nonnegative(item[0])
        end = _finite_nonnegative(item[1])
        if end > start:
            segments.append((start, end))
    durations = [end - start for start, end in segments]
    pause_total = sum(durations)
    count = len(durations)
    mora_n = max(0, int(mora_count))
    phonation_time = max(duration - pause_total, 0.0)
    return {
        "silent_pause_threshold_sec": float(silent_pause_threshold_sec),
        "silent_pause_count": count,
        "silent_pause_total_sec": round(pause_total, 6),
        "silent_pause_ratio": round(pause_total / duration, 6),
        "silent_pause_mean_sec": None if not durations else round(mean(durations), 6),
        "silent_pause_median_sec": None if not durations else round(median(durations), 6),
        "silent_pause_max_sec": None if not durations else round(max(durations), 6),
        "silent_pauses_per_min": round(count * 60.0 / duration, 6),
        "silent_pauses_per_100_mora": None if mora_n <= 0 else round(count * 100.0 / mora_n, 6),
        "mean_run_sec_between_long_silent_pauses": round(phonation_time / max(count + 1, 1), 6),
        "pause_location_available": False,
        "pause_location_reason": "no_time_aligned_clause_or_phrase_boundaries",
        "pause_location_note": "temporal pause position is not equivalent to mid-clause vs clause-final location",
    }


def _adjacent_repetition_candidates(tokens: Sequence[str]) -> List[Dict[str, Any]]:
    """Find exact adjacent 1- or 2-token repetitions; high precision, low recall."""
    clean = [str(token).strip() for token in tokens if str(token).strip()]
    candidates: List[Dict[str, Any]] = []
    used: set[tuple[int, int]] = set()
    for width in (2, 1):
        for start in range(0, max(0, len(clean) - 2 * width + 1)):
            left = clean[start : start + width]
            right = clean[start + width : start + 2 * width]
            if left != right:
                continue
            key = (start, width)
            if key in used:
                continue
            # Single Japanese punctuation-like or one-character functional
            # repetitions are too ambiguous to call repair evidence.
            surface = "".join(left)
            if width == 1 and len(surface) <= 1:
                continue
            candidates.append({
                "token_start": start,
                "token_width": width,
                "surface": surface,
                "evidence": "exact_adjacent_token_ngram_repetition",
            })
            used.add(key)
    return candidates


def transcript_repair_features(
    transcript: str,
    *,
    transcript_source: str = "unknown",
) -> Dict[str, Any]:
    """Extract conservative transcript-side filler/repair candidates.

    Whisper-style ASR can omit or normalize disfluencies.  Therefore these
    features are descriptive evidence with explicit missingness/provenance, not
    learner-error counts and not a score.
    """
    text = _normalize_transcript(transcript)
    filler_hits: List[Dict[str, Any]] = []
    for label, pattern in _HIGH_PRECISION_FILLER_PATTERNS:
        for match in pattern.finditer(text):
            filler_hits.append({
                "type": label,
                "surface": match.group(0),
                "start_char": match.start(),
                "end_char": match.end(),
            })
    ambiguous_markers = {
        marker: text.count(marker)
        for marker in _AMBIGUOUS_DISCOURSE_MARKERS
        if text.count(marker) > 0
    }
    repair_markers = {
        marker: text.count(marker)
        for marker in _AMBIGUOUS_REPAIR_MARKERS
        if text.count(marker) > 0
    }
    tokens = _frontend_tokens(text)
    repetitions = _adjacent_repetition_candidates(tokens)
    return {
        "transcript_available": bool(text),
        "transcript_source": str(transcript_source or "unknown"),
        "high_precision_filled_pause_count": len(filler_hits),
        "high_precision_filled_pause_hits": filler_hits,
        "ambiguous_discourse_marker_counts": ambiguous_markers,
        "ambiguous_repair_marker_counts": repair_markers,
        "exact_adjacent_repetition_candidate_count": len(repetitions),
        "exact_adjacent_repetition_candidates": repetitions,
        "frontend_token_count": len(tokens),
        "repair_evidence_confidence": "low",
        "asr_limitation": "ASR transcripts may omit or normalize fillers, false starts, repetitions and cut-offs",
        "score_mapped": False,
    }


def build_spontaneous_fluency_evidence(
    *,
    mora_count: int,
    speech_duration_sec: float,
    pause_info: Mapping[str, Any],
    transcript: str = "",
    transcript_source: str = "unknown",
    silent_pause_threshold_sec: float = 0.30,
) -> Dict[str, Any]:
    """Build the frozen v1 spontaneous-fluency evidence payload."""
    breakdown = breakdown_fluency_features(
        pause_info,
        speech_duration_sec=speech_duration_sec,
        mora_count=mora_count,
        silent_pause_threshold_sec=silent_pause_threshold_sec,
    )
    speed = speed_fluency_features(
        mora_count=mora_count,
        speech_duration_sec=speech_duration_sec,
        silent_pause_total_sec=breakdown["silent_pause_total_sec"],
    )
    repair = transcript_repair_features(transcript, transcript_source=transcript_source)
    return {
        "schema_version": SCHEMA_VERSION,
        "construct": "utterance_fluency_speed_breakdown_repair",
        "speed": speed,
        "breakdown": breakdown,
        "repair": repair,
        "score_mapped": False,
        "product_calibrated": False,
        "user_facing": False,
        "interpretation": "research_and_audit_evidence_not_a_new_product_score",
    }
