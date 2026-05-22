from __future__ import annotations

import csv
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from jp_speech_eval.evaluator import EvaluationResult, evaluate_utterance

from .user_profile import CalibrationSample, UserVoiceProfile, utc_now_iso


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return round(float(statistics.fmean(clean)), 4) if clean else None


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return round(float(statistics.median(clean)), 4) if clean else None


def _scores_from_result(result: EvaluationResult) -> Dict[str, float]:
    return {
        "total": float(result.total_score),
        "pronunciation": float(result.pronunciation_score),
        "prosody": float(result.prosody_score),
        "fluency": float(result.fluency_score),
        "expression": float(result.tone_score),
    }


def _extract_energy(details: Dict[str, Any]) -> Optional[float]:
    tone = details.get("tone") or {}
    energy = tone.get("energy")
    if isinstance(energy, dict):
        return _finite_float(energy.get("mean") or energy.get("avg") or energy.get("rms"))
    return _finite_float(energy)


def analysis_to_calibration_sample(
    *,
    text: str,
    audio_path: str | Path,
    result: EvaluationResult,
) -> CalibrationSample:
    """Convert one existing evaluator result into calibration evidence."""

    details = result.details or {}
    reliability = dict(details.get("reliability") or {})
    fluency = details.get("fluency") or {}
    tone = details.get("tone") or {}
    f0_values = [_finite_float(row.f0_hz) for row in result.mora_table]
    features: Dict[str, Optional[float]] = {
        "f0_median_hz": _median(f0_values),
        "f0_range_log": _finite_float(tone.get("pitch_range_log")),
        "mora_rate": _finite_float(fluency.get("speech_rate_mora_per_sec")),
        "avg_mora_duration_sec": _finite_float(fluency.get("avg_mora_duration_sec")),
        "pause_ratio": _finite_float((result.pause_info or {}).get("pause_ratio")),
        "intensity_avg": _extract_energy(details),
        "reliability_overall": _finite_float(reliability.get("overall")),
    }
    return CalibrationSample(
        text=text,
        audio_path=str(audio_path),
        kana=result.kana,
        mora_count=len(result.moras),
        scores=_scores_from_result(result),
        features=features,
        reliability=reliability,
        feedback=list(result.feedback),
    )


def build_voice_profile(user_id: str, samples: Iterable[CalibrationSample]) -> UserVoiceProfile:
    """Build a lightweight profile from calibration samples.

    The profile is used for normalization and progress feedback. It should not
    relax fixed Japanese pronunciation targets; it only describes the user's
    current voice range, pace, and recurring practice hints.
    """

    sample_list = list(samples)
    if not sample_list:
        raise ValueError("At least one calibration sample is required.")

    baseline_scores: Dict[str, float] = {}
    for key in ["total", "pronunciation", "prosody", "fluency", "expression"]:
        value = _mean(sample.scores.get(key) for sample in sample_list)
        if value is not None:
            baseline_scores[key] = value

    feedback_counter = Counter()
    for sample in sample_list:
        for item in sample.feedback:
            text = str(item).strip()
            if text:
                feedback_counter[text] += 1
    common_issues = [text for text, count in feedback_counter.most_common(3) if count >= 2]

    now = utc_now_iso()
    return UserVoiceProfile(
        user_id=user_id,
        calibration_samples=sample_list,
        f0_median_hz=_median(sample.features.get("f0_median_hz") for sample in sample_list),
        f0_range_log=_mean(sample.features.get("f0_range_log") for sample in sample_list),
        mora_rate_avg=_mean(sample.features.get("mora_rate") for sample in sample_list),
        avg_mora_duration_sec=_mean(sample.features.get("avg_mora_duration_sec") for sample in sample_list),
        pause_ratio_avg=_mean(sample.features.get("pause_ratio") for sample in sample_list),
        intensity_avg=_mean(sample.features.get("intensity_avg") for sample in sample_list),
        baseline_scores=baseline_scores,
        common_issues=common_issues,
        created_at=now,
        updated_at=now,
    )


def read_calibration_manifest(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f) if row.get("audio_path") and row.get("text")]


def calibrate_from_manifest(
    *,
    user_id: str,
    rows: Iterable[Dict[str, str]],
    scoring_config_path: str | Path | None = None,
    sample_rate: int = 16000,
) -> UserVoiceProfile:
    samples: List[CalibrationSample] = []
    for row in rows:
        text = row["text"]
        audio_path = row["audio_path"]
        cache_path = row.get("cache_path") or None
        result = evaluate_utterance(
            text=None if cache_path else text,
            wav_path=audio_path,
            cache_path=cache_path,
            alignment_mode="cached_dtw" if cache_path else "equal",
            scoring_config_path=scoring_config_path,
            sample_rate=sample_rate,
            use_content_match=bool(cache_path),
        )
        samples.append(analysis_to_calibration_sample(text=text, audio_path=audio_path, result=result))
    return build_voice_profile(user_id, samples)
