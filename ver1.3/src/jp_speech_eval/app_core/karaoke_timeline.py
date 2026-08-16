from __future__ import annotations

"""Evidence-safe timeline payload for C-end karaoke-style playback.

This module is presentation infrastructure. It does not change ProductScore.
All timestamps exposed here use the original recording playback timebase.
ASR word timestamps remain ASR alignment evidence, never phone correctness.
Relative F0 is a visual description of voice movement, never lexical
pitch-accent correctness or contextual intonation correctness.
"""

import math
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCHEMA_VERSION = "consumer_karaoke_timeline_v1"
F0_VISUALIZATION_SCHEMA = "relative_f0_visualization_v1"
CANONICAL_DIMENSION_ORDER = ("delivery_fluency", "clarity", "mora_timing", "intonation")
CANONICAL_LABELS = {
    "delivery_fluency": "流暢さ",
    "clarity": "明瞭さ",
    "mora_timing": "リズム",
    "intonation": "抑揚",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _confidence_band(probability: Any) -> str:
    value = _finite(probability)
    if value is None:
        return "unknown"
    if value >= 0.85:
        return "high"
    if value >= 0.60:
        return "medium"
    return "low"


def build_relative_f0_visualization(
    times: Sequence[float] | Iterable[float],
    f0_hz: Sequence[float] | Iterable[float],
    *,
    max_points: int = 160,
) -> Dict[str, Any]:
    """Compress an already-computed F0 track for UI rendering.

    The vertical unit is semitones relative to the speaker's median F0, which
    makes the visual useful across voice ranges without implying that a higher
    absolute F0 is better. Unvoiced bins are retained as ``value=None`` so a UI
    can draw gaps instead of inventing a continuous contour.
    """

    t_values = list(times)
    f_values = list(f0_hz)
    n = min(len(t_values), len(f_values))
    pairs: List[tuple[float, Optional[float]]] = []
    voiced_values: List[float] = []
    for index in range(n):
        t = _finite(t_values[index])
        f0 = _finite(f_values[index])
        if t is None or t < 0:
            continue
        voiced = f0 if f0 is not None and f0 > 0 else None
        pairs.append((t, voiced))
        if voiced is not None:
            voiced_values.append(voiced)

    if len(voiced_values) < 5 or not pairs:
        return {
            "schema_version": F0_VISUALIZATION_SCHEMA,
            "available": False,
            "source_timebase": "speech_trim_relative",
            "vertical_unit": "semitone_relative_to_speaker_median",
            "points": [],
            "voiced_frame_count": len(voiced_values),
            "interpretation": "visualization_only_not_pitch_accent_correctness",
            "failure_reason": "insufficient_f0",
        }

    center_hz = float(median(voiced_values))
    duration = max(t for t, _ in pairs)
    bin_count = max(1, min(int(max_points), len(pairs)))
    if duration <= 0:
        bin_count = 1
    bins: List[List[tuple[float, float]]] = [[] for _ in range(bin_count)]
    for t, f0 in pairs:
        if f0 is None:
            continue
        if bin_count == 1 or duration <= 0:
            bin_index = 0
        else:
            bin_index = min(bin_count - 1, int((t / duration) * bin_count))
        bins[bin_index].append((t, f0))

    points: List[Dict[str, Any]] = []
    for index, bucket in enumerate(bins):
        if duration > 0:
            t_center = duration * ((index + 0.5) / bin_count)
        else:
            t_center = 0.0
        if not bucket:
            points.append({"t_sec": round(t_center, 4), "relative_semitone": None})
            continue
        bucket_times = [item[0] for item in bucket]
        bucket_f0 = [item[1] for item in bucket]
        t_center = float(median(bucket_times))
        f0_center = float(median(bucket_f0))
        semitone = 12.0 * math.log2(max(f0_center, 1e-6) / max(center_hz, 1e-6))
        # UI robustness only: a single octave error must not blow up the canvas.
        # The score path never reads this clipped visualization value.
        semitone = _clip(semitone, -12.0, 12.0)
        points.append({"t_sec": round(t_center, 4), "relative_semitone": round(semitone, 4)})

    return {
        "schema_version": F0_VISUALIZATION_SCHEMA,
        "available": True,
        "source_timebase": "speech_trim_relative",
        "vertical_unit": "semitone_relative_to_speaker_median",
        "points": points,
        "voiced_frame_count": len(voiced_values),
        "speaker_median_f0_hz": round(center_hz, 3),
        "interpretation": "visualization_only_not_pitch_accent_correctness",
        "failure_reason": "",
    }


def _original_playback_time(relative_time: Any, *, speech_start: float, raw_duration: float) -> Optional[float]:
    value = _finite(relative_time)
    if value is None:
        return None
    return round(_clip(speech_start + value, 0.0, max(raw_duration, 0.0)), 4)


def _word_rows(
    asr: Mapping[str, Any],
    *,
    speech_start: float,
    raw_duration: float,
) -> List[Dict[str, Any]]:
    raw_words = asr.get("words") if isinstance(asr.get("words"), list) else []
    rows: List[Dict[str, Any]] = []
    for raw in raw_words:
        item = _mapping(raw)
        text = str(item.get("text") or "").strip()
        start = _original_playback_time(item.get("start_sec"), speech_start=speech_start, raw_duration=raw_duration)
        end = _original_playback_time(item.get("end_sec"), speech_start=speech_start, raw_duration=raw_duration)
        if not text or start is None or end is None or end <= start:
            continue
        probability = _finite(item.get("probability"))
        rows.append(
            {
                "text": text,
                "start_sec": start,
                "end_sec": end,
                "asr_probability": None if probability is None else round(_clip(probability, 0.0, 1.0), 4),
                "asr_confidence": _confidence_band(probability),
                "source": "asr_word_timestamp",
                "interpretation": "playback_alignment_not_phone_correctness",
            }
        )
    rows.sort(key=lambda row: (row["start_sec"], row["end_sec"]))
    return rows


def _pause_rows(
    result: Mapping[str, Any],
    *,
    speech_start: float,
    raw_duration: float,
) -> List[Dict[str, Any]]:
    pause_info = _mapping(result.get("pause_info"))
    raw_segments = pause_info.get("pause_segments") if isinstance(pause_info.get("pause_segments"), list) else []
    rows: List[Dict[str, Any]] = []
    for raw in raw_segments:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        start = _original_playback_time(raw[0], speech_start=speech_start, raw_duration=raw_duration)
        end = _original_playback_time(raw[1], speech_start=speech_start, raw_duration=raw_duration)
        if start is None or end is None or end <= start:
            continue
        rows.append(
            {
                "start_sec": start,
                "end_sec": end,
                "duration_sec": round(end - start, 4),
                "kind": "long_silence",
                "interpretation": "acoustic_pause_not_automatically_a_fluency_error",
            }
        )
    return rows


def _pitch_payload(
    details: Mapping[str, Any],
    *,
    speech_start: float,
    raw_duration: float,
) -> Dict[str, Any]:
    source = _mapping(details.get("visualization_source"))
    relative_f0 = _mapping(source.get("relative_f0"))
    points: List[Dict[str, Any]] = []
    if bool(relative_f0.get("available")):
        raw_points = relative_f0.get("points") if isinstance(relative_f0.get("points"), list) else []
        for raw in raw_points:
            item = _mapping(raw)
            t = _original_playback_time(item.get("t_sec"), speech_start=speech_start, raw_duration=raw_duration)
            if t is None:
                continue
            value = _finite(item.get("relative_semitone"))
            points.append(
                {
                    "t_sec": t,
                    "relative_semitone": None if value is None else round(value, 4),
                }
            )
    return {
        "available": bool(points and any(point["relative_semitone"] is not None for point in points)),
        "points": points,
        "vertical_unit": "semitone_relative_to_speaker_median",
        "source": "same_pass_f0_visualization" if points else "unavailable",
        "interpretation": "voice_movement_visualization_not_lexical_pitch_accent_correctness",
        "contextual_intonation_claim": False,
    }


def _dimension_rows(user_facing: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw_dimensions = user_facing.get("score_dimensions") if isinstance(user_facing.get("score_dimensions"), list) else []
    by_key = {
        str(_mapping(item).get("key") or ""): _mapping(item)
        for item in raw_dimensions
        if str(_mapping(item).get("key") or "")
    }
    output: List[Dict[str, Any]] = []
    for key in CANONICAL_DIMENSION_ORDER:
        item = by_key.get(key, {})
        value = _finite(item.get("value"))
        state = str(item.get("evidence_state") or "unavailable")
        output.append(
            {
                "key": key,
                "label": CANONICAL_LABELS[key],
                "score": None if value is None else int(round(_clip(value, 0.0, 100.0))),
                "available": value is not None and bool(item.get("available", True)),
                "confidence": str(item.get("confidence") or "unknown"),
                "evidence_state": state,
                "precision_hint": str(item.get("precision_hint") or "unknown"),
                "construct": str(item.get("construct") or ""),
                "note": str(item.get("note") or ""),
                "numeric_semantics": str(item.get("numeric_semantics") or "practice_proxy_not_formal_measurement"),
            }
        )
    return output


def build_consumer_karaoke_timeline(
    result: Mapping[str, Any],
    user_facing: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the stable consumer visualization payload from one evaluation."""

    details = _mapping(result.get("details"))
    endpointing = _mapping(result.get("endpointing")) or _mapping(details.get("endpointing"))
    asr = _mapping(details.get("asr"))
    raw_duration = _finite(endpointing.get("raw_duration"))
    if raw_duration is None:
        raw_duration = _finite(result.get("duration_sec")) or 0.0
    speech_start = _finite(endpointing.get("speech_start")) or 0.0
    speech_end = _finite(endpointing.get("speech_end"))
    if speech_end is None:
        speech_end = raw_duration
    speech_start = _clip(speech_start, 0.0, max(raw_duration, 0.0))
    speech_end = _clip(speech_end, speech_start, max(raw_duration, speech_start))

    words = _word_rows(asr, speech_start=speech_start, raw_duration=raw_duration)
    transcript = str(asr.get("text") or result.get("target_text") or "").strip()
    if words:
        sync_mode = "word_timestamps"
        sync_note = "ASRの実時間スタンプに合わせて字幕を表示します。単語の正誤判定ではありません。"
    elif transcript:
        sync_mode = "sentence_progress_only"
        sync_note = "単語ごとの実時間スタンプがないため、字幕は同期せず全文表示します。"
    else:
        sync_mode = "unavailable"
        sync_note = "同期表示に使える文字情報がありません。"

    pauses = _pause_rows(result, speech_start=speech_start, raw_duration=raw_duration)
    pitch = _pitch_payload(details, speech_start=speech_start, raw_duration=raw_duration)

    return {
        "schema_version": SCHEMA_VERSION,
        "score_role": "visualization_only",
        "product_score_changed": False,
        "timebase": "original_recording_playback_seconds",
        "duration_sec": round(raw_duration, 4),
        "speech_region": {
            "start_sec": round(speech_start, 4),
            "end_sec": round(speech_end, 4),
            "detected": bool(endpointing.get("detected", bool(raw_duration))),
        },
        "transcript": transcript,
        "sync_mode": sync_mode,
        "sync_note": sync_note,
        "words": words,
        "pauses": pauses,
        "pitch": pitch,
        "dimensions": _dimension_rows(user_facing),
        "recording": {
            "analyzability": _finite(_mapping(details.get("recording_quality")).get("score")),
            "is_product_dimension": False,
            "interpretation": "recording_analyzability_not_pronunciation_accuracy",
        },
        "guardrails": {
            "word_timestamps_are_phone_correctness": False,
            "pitch_curve_is_lexical_pitch_accent_correctness": False,
            "pause_is_automatically_an_error": False,
            "recording_quality_is_a_speaking_dimension": False,
            "missing_evidence_should_be_rendered_as_bad_performance": False,
        },
    }
