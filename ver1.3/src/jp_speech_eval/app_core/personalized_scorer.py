from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from jp_speech_eval.evaluator import EvaluationResult

from .user_profile import UserVoiceProfile


@dataclass
class PersonalizedComparison:
    """Product-facing progress comparison built from existing evaluator scores."""

    standard_scores: Dict[str, float]
    personal_delta: Dict[str, Optional[float]]
    progress_delta: Dict[str, Optional[float]]
    feedback: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _scores_from_result(result: EvaluationResult) -> Dict[str, float]:
    return {
        "total": float(result.total_score),
        "pronunciation": float(result.pronunciation_score),
        "prosody": float(result.prosody_score),
        "fluency": float(result.fluency_score),
        "expression": float(result.tone_score),
    }


def _feature_from_result(result: EvaluationResult, key: str) -> Optional[float]:
    details = result.details or {}
    fluency = details.get("fluency") or {}
    tone = details.get("tone") or {}
    if key == "mora_rate":
        return _to_float(fluency.get("speech_rate_mora_per_sec"))
    if key == "avg_mora_duration_sec":
        return _to_float(fluency.get("avg_mora_duration_sec"))
    if key == "f0_range_log":
        return _to_float(tone.get("pitch_range_log"))
    if key == "pause_ratio":
        return _to_float((result.pause_info or {}).get("pause_ratio"))
    return None


def _to_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _pct_delta(current: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if current is None or baseline is None or abs(baseline) < 1e-8:
        return None
    return round((current - baseline) / baseline * 100.0, 2)


def _record_score(previous_record: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not previous_record:
        return None
    scores = previous_record.get("scores") or {}
    return _to_float(scores.get(key))


def _record_feature(previous_record: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not previous_record:
        return None
    features = previous_record.get("features") or {}
    return _to_float(features.get(key))


def compare_to_profile(
    result: EvaluationResult,
    profile: Optional[UserVoiceProfile] = None,
    previous_record: Optional[Dict[str, Any]] = None,
) -> PersonalizedComparison:
    """Create at most three simple progress hints without changing standard scores."""

    scores = _scores_from_result(result)
    current_rate = _feature_from_result(result, "mora_rate")
    current_duration = _feature_from_result(result, "avg_mora_duration_sec")
    current_pitch_range = _feature_from_result(result, "f0_range_log")

    personal_delta = {
        "total_vs_calibration": None,
        "mora_rate_pct_vs_calibration": None,
        "avg_mora_duration_pct_vs_calibration": None,
        "f0_range_log_vs_calibration": None,
    }
    feedback: List[str] = []

    if profile is not None:
        base_total = profile.baseline_scores.get("total")
        if base_total is not None:
            personal_delta["total_vs_calibration"] = round(scores["total"] - float(base_total), 2)
        personal_delta["mora_rate_pct_vs_calibration"] = _pct_delta(current_rate, profile.mora_rate_avg)
        personal_delta["avg_mora_duration_pct_vs_calibration"] = _pct_delta(
            current_duration,
            profile.avg_mora_duration_sec,
        )
        if current_pitch_range is not None and profile.f0_range_log is not None:
            personal_delta["f0_range_log_vs_calibration"] = round(current_pitch_range - profile.f0_range_log, 4)

    previous_total = _record_score(previous_record, "total")
    previous_rate = _record_feature(previous_record, "mora_rate")
    progress_delta = {
        "total_vs_previous": round(scores["total"] - previous_total, 2) if previous_total is not None else None,
        "mora_rate_pct_vs_previous": _pct_delta(current_rate, previous_rate),
    }

    reliability = (result.details or {}).get("reliability") or {}
    if str(reliability.get("level")) == "low":
        feedback.append("这次录音不够稳定，建议先重录一次再看进步。")
    elif progress_delta["total_vs_previous"] is not None and progress_delta["total_vs_previous"] >= 3:
        feedback.append("这次比上次更稳定，可以保留这个读法。")
    elif progress_delta["total_vs_previous"] is not None and progress_delta["total_vs_previous"] <= -8:
        feedback.append("这次比上次不稳定一点，先回到慢速清楚地读。")

    rate_vs_profile = personal_delta["mora_rate_pct_vs_calibration"]
    if rate_vs_profile is not None and abs(rate_vs_profile) >= 25:
        if rate_vs_profile > 0:
            feedback.append("这次比你平时快很多，先放慢一点会更清楚。")
        else:
            feedback.append("这次比你平时慢很多，注意不要把句子切得太碎。")
    elif progress_delta["mora_rate_pct_vs_previous"] is not None:
        rate_delta = progress_delta["mora_rate_pct_vs_previous"]
        if -20 <= rate_delta <= -5:
            feedback.append("语速比上次慢了一点，听起来更容易跟上。")

    if profile and profile.common_issues:
        first_issue = profile.common_issues[0]
        if first_issue in result.feedback:
            feedback.append("你常出现的节奏或音高问题这次还在，建议用同一句再练一遍。")

    if not feedback:
        feedback.append("这次可以作为新的练习记录，下一次重点看语速和句末语调是否更稳定。")

    feedback.append("个性化反馈只看进步趋势，不会把错误发音当成正确标准。")

    return PersonalizedComparison(
        standard_scores=scores,
        personal_delta=personal_delta,
        progress_delta=progress_delta,
        feedback=feedback[:4],
        debug={
            "current_features": {
                "mora_rate": current_rate,
                "avg_mora_duration_sec": current_duration,
                "f0_range_log": current_pitch_range,
            },
            "profile_version": profile.version if profile else None,
            "previous_record_available": previous_record is not None,
        },
    )
