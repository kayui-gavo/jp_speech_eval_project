"""Target-independent evidence for spontaneous/general Japanese speech.

This module deliberately separates *evidence* from the learner-facing /100
contract.  Free conversation has no trustworthy target utterance, so the
system must not manufacture phone correctness, mora-local correctness, or
lexical pitch-accent correctness from a transcript hallucinated by ASR.

The evidence here is intended for criterion validation and C-end A/B work:

* clarity: ASR machine-recoverability evidence, not human comprehensibility;
* rhythm: weak ASR-word local-tempo evidence, not equal-mora isochrony;
* fluency: speed/breakdown/repair evidence supplied by spontaneous_fluency;
* intonation: target-independent robust F0 movement evidence, not lexical
  pitch-accent correctness or context-appropriate intonation.

A provisional candidate surface is included only so product engineers can
measure whether target-independent evidence creates useful score dispersion.
It is strongly shrunk toward a neutral anchor, remains uncalibrated, and must
not be copied into ProductScore before construct-matched human validation.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .text_frontend import build_text_info


SCHEMA_VERSION = "free_speech_dimension_evidence_v1"
CANDIDATE_POLICY_ID = "free_speech_candidate_surface_v1_shadow"
NEUTRAL_ANCHOR = 70.0


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clip01(value: Any) -> Optional[float]:
    number = _finite(value)
    if number is None:
        return None
    return max(0.0, min(1.0, number))


def _summary(values: Sequence[float]) -> Dict[str, Optional[float] | int]:
    clean = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if clean.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(clean.size),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "p10": float(np.quantile(clean, 0.10)),
        "p90": float(np.quantile(clean, 0.90)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def _merged_interval_duration(intervals: Sequence[tuple[float, float]]) -> float:
    usable = sorted((max(0.0, float(a)), max(0.0, float(b))) for a, b in intervals if b > a)
    if not usable:
        return 0.0
    total = 0.0
    start, end = usable[0]
    for next_start, next_end in usable[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += max(0.0, end - start)
            start, end = next_start, next_end
    total += max(0.0, end - start)
    return total


def build_asr_recoverability_evidence(
    asr_info: Mapping[str, Any],
    *,
    speech_duration_sec: float,
) -> Dict[str, Any]:
    """Summarise decoder confidence without calling it human intelligibility."""

    available_asr = bool(asr_info.get("available"))
    words = asr_info.get("words") if isinstance(asr_info.get("words"), list) else []
    segments = asr_info.get("segments") if isinstance(asr_info.get("segments"), list) else []

    word_probs: list[float] = []
    intervals: list[tuple[float, float]] = []
    for raw in words:
        item = raw if isinstance(raw, Mapping) else {}
        prob = _clip01(item.get("probability"))
        if prob is not None:
            word_probs.append(prob)
        start = _finite(item.get("start_sec"))
        end = _finite(item.get("end_sec"))
        if start is not None and end is not None and end > start:
            intervals.append((start, end))

    avg_logprobs: list[float] = []
    no_speech_probs: list[float] = []
    compression_ratios: list[float] = []
    for raw in segments:
        item = raw if isinstance(raw, Mapping) else {}
        value = _finite(item.get("avg_logprob"))
        if value is not None:
            avg_logprobs.append(value)
        value = _clip01(item.get("no_speech_prob"))
        if value is not None:
            no_speech_probs.append(value)
        value = _finite(item.get("compression_ratio"))
        if value is not None:
            compression_ratios.append(value)

    word_summary = _summary(word_probs)
    segment_logprob_summary = _summary(avg_logprobs)
    no_speech_summary = _summary(no_speech_probs)
    compression_summary = _summary(compression_ratios)
    duration = max(0.0, float(speech_duration_sec or 0.0))
    covered = _merged_interval_duration(intervals)
    timing_coverage = min(1.0, covered / duration) if duration > 0 else None

    recoverability_index: Optional[float] = None
    if len(word_probs) >= 2:
        # A deliberately simple machine-confidence summary.  Word probability
        # is affected by the decoder/language model and therefore is *not* a
        # human-comprehensibility probability.
        med = float(np.median(word_probs))
        p10 = float(np.quantile(np.asarray(word_probs, dtype=float), 0.10))
        recoverability_index = max(0.0, min(1.0, 0.65 * med + 0.35 * p10))

    if len(word_probs) >= 5 and (timing_coverage is None or timing_coverage >= 0.35):
        confidence = "medium"
        tier = "word_probability_distribution"
    elif word_probs or avg_logprobs:
        confidence = "low"
        tier = "sparse_decoder_confidence"
    else:
        confidence = "unavailable"
        tier = "no_decoder_confidence"

    return {
        "construct": "machine_recoverability_proxy_not_human_comprehensibility",
        "available": bool(available_asr and (word_probs or avg_logprobs)),
        "confidence": confidence,
        "evidence_tier": tier,
        "provider": str(asr_info.get("provider") or ""),
        "model": str(asr_info.get("model") or ""),
        "word_probability": word_summary,
        "segment_avg_logprob": segment_logprob_summary,
        "segment_no_speech_probability": no_speech_summary,
        "segment_compression_ratio": compression_summary,
        "word_timestamp_coverage": timing_coverage,
        "asr_recoverability_index_0to1": recoverability_index,
        "language_probability": _clip01(asr_info.get("language_probability")),
        "language_probability_role": "routing_only_not_clarity_score",
        "score_mapped": False,
        "product_calibrated": False,
        "user_facing": False,
        "caveat": (
            "decoder confidence can reflect lexical predictability, ASR bias, and model choice; "
            "validate against human clarity/comprehensibility before product promotion"
        ),
    }


def build_word_timing_rhythm_evidence(asr_info: Mapping[str, Any]) -> Dict[str, Any]:
    """Estimate weak local-tempo structure from ASR word timestamps.

    Durations are normalised by the number of Japanese morae in each recognised
    word.  This is a *word-level local-tempo proxy*; it does not assert that
    Japanese morae are perfectly isochronous, and ASR tokenisation/timing errors
    can dominate short utterances.
    """

    words = asr_info.get("words") if isinstance(asr_info.get("words"), list) else []
    sec_per_mora: list[float] = []
    accepted_words: list[Dict[str, Any]] = []
    rejected = 0
    for raw in words:
        item = raw if isinstance(raw, Mapping) else {}
        text = str(item.get("text") or "").strip()
        start = _finite(item.get("start_sec"))
        end = _finite(item.get("end_sec"))
        if not text or start is None or end is None or end <= start:
            rejected += 1
            continue
        duration = end - start
        if duration < 0.04 or duration > 3.0:
            rejected += 1
            continue
        try:
            info = build_text_info(text)
            mora_count = len(info.moras)
        except Exception:
            mora_count = 0
        if mora_count <= 0:
            rejected += 1
            continue
        value = duration / mora_count
        if not math.isfinite(value) or value <= 0:
            rejected += 1
            continue
        sec_per_mora.append(value)
        accepted_words.append({
            "text": text,
            "duration_sec": round(duration, 6),
            "mora_count": int(mora_count),
            "sec_per_mora": round(value, 6),
        })

    available = len(sec_per_mora) >= 3
    if available:
        log_values = np.log(np.asarray(sec_per_mora, dtype=float))
        center = float(np.median(log_values))
        mad = float(np.median(np.abs(log_values - center)))
        spread = float(np.quantile(log_values, 0.90) - np.quantile(log_values, 0.10))
        median_sec = float(np.median(sec_per_mora))
        confidence = "low" if len(sec_per_mora) < 6 else "medium"
    else:
        mad = None
        spread = None
        median_sec = float(np.median(sec_per_mora)) if sec_per_mora else None
        confidence = "unavailable" if not sec_per_mora else "low"

    return {
        "construct": "asr_word_level_local_tempo_structure_not_mora_isochrony",
        "available": available,
        "confidence": confidence,
        "evidence_tier": "asr_word_timestamp_mora_normalized" if available else "insufficient_word_timing",
        "usable_word_count": len(sec_per_mora),
        "rejected_word_count": rejected,
        "median_sec_per_mora": median_sec,
        "local_tempo_irregularity_mad_log_sec_per_mora": mad,
        "local_tempo_spread_p90_p10_log_sec_per_mora": spread,
        "word_examples": accepted_words[:8],
        "score_mapped": False,
        "product_calibrated": False,
        "user_facing": False,
        "caveat": (
            "ASR word segmentation/timestamps are weak boundaries; expressive local slowing is not automatically an error"
        ),
    }


def build_target_independent_intonation_evidence(
    times: Sequence[float] | np.ndarray,
    f0_hz: Sequence[float] | np.ndarray,
) -> Dict[str, Any]:
    """Return robust F0 movement descriptors without an intonation correctness claim."""

    t = np.asarray(times, dtype=float).reshape(-1)
    f0 = np.asarray(f0_hz, dtype=float).reshape(-1)
    n = min(t.size, f0.size)
    t = t[:n]
    f0 = f0[:n]
    valid = np.isfinite(t) & np.isfinite(f0) & (f0 > 0)
    t = t[valid]
    f0 = f0[valid]
    if f0.size < 5:
        return {
            "construct": "target_independent_f0_movement_not_contextual_intonation_correctness",
            "available": False,
            "confidence": "unavailable",
            "evidence_tier": "insufficient_f0",
            "valid_frame_count": int(f0.size),
            "robust_range_semitones_p90_p10": None,
            "median_absolute_step_semitones": None,
            "terminal_slope_semitones_per_sec": None,
            "score_mapped": False,
            "product_calibrated": False,
            "user_facing": False,
            "caveat": "F0 extraction failure must not be interpreted as flat intonation",
        }

    st = 12.0 * np.log2(f0 / float(np.median(f0)))
    robust_range = float(np.quantile(st, 0.90) - np.quantile(st, 0.10))

    diffs: list[float] = []
    for idx in range(1, len(st)):
        dt = float(t[idx] - t[idx - 1])
        if 0.0 < dt <= 0.08:
            diffs.append(abs(float(st[idx] - st[idx - 1])))
    median_step = float(np.median(diffs)) if diffs else None

    terminal_slope: Optional[float] = None
    if len(st) >= 8 and float(t[-1] - t[0]) > 0.25:
        cutoff = float(t[0] + 0.70 * (t[-1] - t[0]))
        mask = t >= cutoff
        if int(np.sum(mask)) >= 4 and float(np.ptp(t[mask])) > 0.08:
            slope = np.polyfit(t[mask], st[mask], 1)[0]
            if math.isfinite(float(slope)):
                terminal_slope = float(slope)

    return {
        "construct": "target_independent_f0_movement_not_contextual_intonation_correctness",
        "available": True,
        "confidence": "low",
        "evidence_tier": "robust_global_f0_movement",
        "valid_frame_count": int(f0.size),
        "robust_range_semitones_p90_p10": robust_range,
        "median_absolute_step_semitones": median_step,
        "terminal_slope_semitones_per_sec": terminal_slope,
        "score_mapped": False,
        "product_calibrated": False,
        "user_facing": False,
        "caveat": (
            "global F0 movement does not establish context-appropriate intonation or lexical pitch-accent correctness"
        ),
    }


def build_free_speech_dimension_evidence(
    *,
    asr_info: Mapping[str, Any],
    speech_duration_sec: float,
    f0_times: Sequence[float] | np.ndarray,
    f0_hz: Sequence[float] | np.ndarray,
    spontaneous_fluency: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "general_japanese_target_independent",
        "clarity": build_asr_recoverability_evidence(
            asr_info,
            speech_duration_sec=speech_duration_sec,
        ),
        "rhythm": build_word_timing_rhythm_evidence(asr_info),
        "fluency": {
            "construct": "spontaneous_utterance_fluency_speed_breakdown_repair",
            "available": bool(spontaneous_fluency),
            "source_schema_version": spontaneous_fluency.get("schema_version"),
            "speed": spontaneous_fluency.get("speed"),
            "breakdown": spontaneous_fluency.get("breakdown"),
            "repair": spontaneous_fluency.get("repair"),
            "score_mapped": False,
            "product_calibrated": False,
            "user_facing": False,
        },
        "intonation": build_target_independent_intonation_evidence(f0_times, f0_hz),
        "score_mapped": False,
        "product_calibrated": False,
        "user_facing": False,
        "interpretation": "construct-separated target-independent evidence for criterion validation",
    }


def _shrunk(raw_score: float, strength: float) -> float:
    value = NEUTRAL_ANCHOR + float(strength) * (float(raw_score) - NEUTRAL_ANCHOR)
    return max(0.0, min(100.0, value))


def build_shadow_candidate_surface(
    evidence: Mapping[str, Any],
    *,
    current_fluency_score: Any,
) -> Dict[str, Any]:
    """Build a deliberately conservative *shadow* score surface.

    These formulas are not claimed to be criterion-valid.  They exist so a
    held acceptance/human panel can test whether the evidence produces useful
    ordering and dispersion before any ProductScore migration.
    """

    clarity = evidence.get("clarity") if isinstance(evidence.get("clarity"), Mapping) else {}
    rhythm = evidence.get("rhythm") if isinstance(evidence.get("rhythm"), Mapping) else {}
    intonation = evidence.get("intonation") if isinstance(evidence.get("intonation"), Mapping) else {}

    clarity_index = _finite(clarity.get("asr_recoverability_index_0to1"))
    clarity_score = NEUTRAL_ANCHOR
    clarity_available = clarity_index is not None
    if clarity_available:
        clarity_score = _shrunk(100.0 * max(0.0, min(1.0, clarity_index)), 0.35)

    rhythm_mad = _finite(rhythm.get("local_tempo_irregularity_mad_log_sec_per_mora"))
    rhythm_score = NEUTRAL_ANCHOR
    rhythm_available = rhythm_mad is not None
    if rhythm_available:
        raw_rhythm = 100.0 * math.exp(-1.6 * max(0.0, rhythm_mad))
        rhythm_score = _shrunk(raw_rhythm, 0.30)

    range_st = _finite(intonation.get("robust_range_semitones_p90_p10"))
    intonation_score = NEUTRAL_ANCHOR
    intonation_available = range_st is not None
    if intonation_available:
        # Broad, intentionally forgiving plateau.  This is a dispersion probe,
        # not a normative claim about the one correct Japanese F0 range.
        if 3.0 <= range_st <= 12.0:
            raw_intonation = 90.0
        elif 1.5 <= range_st < 3.0:
            raw_intonation = 60.0 + 20.0 * (range_st - 1.5) / 1.5
        elif 12.0 < range_st <= 18.0:
            raw_intonation = 90.0 - 25.0 * (range_st - 12.0) / 6.0
        else:
            raw_intonation = 55.0
        intonation_score = _shrunk(raw_intonation, 0.35)

    fluency = _finite(current_fluency_score)
    fluency_score = NEUTRAL_ANCHOR if fluency is None else max(0.0, min(100.0, fluency))

    components = {
        "clarity": round(float(clarity_score), 4),
        "mora_timing": round(float(rhythm_score), 4),
        "delivery_fluency": round(float(fluency_score), 4),
        "intonation": round(float(intonation_score), 4),
    }
    weighted = (
        0.30 * components["clarity"]
        + 0.25 * components["mora_timing"]
        + 0.25 * components["delivery_fluency"]
        + 0.20 * components["intonation"]
    )
    return {
        "policy_id": CANDIDATE_POLICY_ID,
        "available_evidence": {
            "clarity": clarity_available,
            "mora_timing": rhythm_available,
            "delivery_fluency": fluency is not None,
            "intonation": intonation_available,
        },
        "component_candidates": components,
        "weighted_candidate": round(float(weighted), 4),
        "neutral_anchor": NEUTRAL_ANCHOR,
        "score_mapped": True,
        "mapping_status": "heuristic_shadow_only_not_product_calibrated",
        "product_calibrated": False,
        "user_facing": False,
        "product_score_changed": False,
        "promotion_gate": "construct_matched_human_validation_and_held_c_end_acceptance_required",
    }
