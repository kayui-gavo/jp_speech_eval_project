from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .scoring_policy import ScoringPolicy


@dataclass(frozen=True)
class FeedbackCandidate:
    dimension: str
    severity: str
    confidence: str
    evidence_type: str
    location: Optional[Dict[str, Any]]
    user_message: str
    practice_tip: str
    caveat: Optional[str] = None
    category: str = "practice"
    message: Optional[str] = None
    type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data["message"] is None:
            data["message"] = self.user_message
        return data


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _location(result: Mapping[str, Any], index: int, mora: str) -> Dict[str, Any]:
    rows = result.get("mora_table") if isinstance(result.get("mora_table"), list) else []
    row = _mapping(rows[index]) if 0 <= index < len(rows) else {}
    moras = [str(item) for item in (result.get("moras") or [])]
    nearby = "".join(moras[max(0, index - 1): min(len(moras), index + 2)])
    start = row.get("start_sec")
    end = row.get("end_sec")
    return {
        "mora_index": index + 1,
        "mora_text": mora,
        "nearby_kana": nearby or mora,
        "time_range_sec": [start, end] if start is not None and end is not None else None,
    }


def _content_or_evidence_blocker(result: Mapping[str, Any], gate: Any) -> Optional[FeedbackCandidate]:
    reasons = set(str(reason) for reason in (gate.reasons or []))
    details = _mapping(result.get("details"))
    weak = _mapping(details.get("weak_reference_native_likeness"))
    guardrail = _mapping(weak.get("weak_overall_guardrail"))
    guardrail_reasons = set(str(reason) for reason in (guardrail.get("reasons") or []))

    if "content_mismatch" in reasons or "content_mismatch_veto" in reasons:
        latin = bool({"latin_dominant_transcript", "non_japanese_asr_language"}.intersection(reasons))
        return FeedbackCandidate(
            dimension="content_match",
            severity="high",
            confidence="high",
            evidence_type="content_mismatch",
            location=None,
            user_message="这次没有确认到可评分的日语内容。" if latin else "这次读到的内容和练习句不一致。",
            practice_tip="请确认一整句日语文本，再清楚地读一遍。",
            caveat="内容未确认时，不显示发音或音高细节。",
            category="content_match",
        )
    if {"latin_dominant_confirmed_text", "low_japanese_likeness", "kana_or_mora_unavailable"}.intersection(guardrail_reasons):
        return FeedbackCandidate(
            dimension="content_match",
            severity="high",
            confidence="high",
            evidence_type="content_mismatch",
            location=None,
            user_message="系统还没有确认到一整句日语。",
            practice_tip="请手动确认或输入刚才说的日语句子，再重新评价。",
            caveat="内容未确认时，不显示发音或音高细节。",
            category="content_match",
        )
    if "short_utterance" in reasons or "short_utterance_insufficient_evidence" in guardrail_reasons:
        return FeedbackCandidate(
            dimension="recording_quality",
            severity="medium",
            confidence="high",
            evidence_type="too_short",
            location=None,
            user_message="这段太短，系统还不能稳定判断。",
            practice_tip="请说一个完整短句，尽量连续读完。",
            caveat="证据不足，因此不判断具体发音或音高。",
            category="insufficient_evidence",
        )
    if "recording_quality_bad" in reasons:
        return FeedbackCandidate(
            dimension="recording_quality",
            severity="high",
            confidence="high",
            evidence_type="alignment_uncertain",
            location=None,
            user_message="这次录音太小或不够清楚，暂时无法稳定判断。",
            practice_tip="靠近麦克风一点，在安静环境里重录完整短句。",
            caveat="录音证据不足，因此不判断具体发音。",
            category="recording_quality",
        )
    if "fallback_alignment" in reasons or "alignment_confidence_low" in reasons:
        return FeedbackCandidate(
            dimension="pronunciation_clarity",
            severity="medium",
            confidence="medium",
            evidence_type="alignment_uncertain",
            location=None,
            user_message="这次每一拍的位置没有稳定对齐，细节先不下结论。",
            practice_tip="先按一拍一拍慢读，再把整句连起来重读。",
            caveat="对齐不稳定，因此不提示具体拍或音高错误。",
            category="pronunciation",
        )
    return None


def _low_f0_candidate(result: Mapping[str, Any], gate: Any) -> Optional[FeedbackCandidate]:
    reasons = set(str(reason) for reason in (gate.reasons or []))
    details = _mapping(result.get("details"))
    weak = _mapping(details.get("weak_reference_native_likeness"))
    if "low_f0_coverage" not in reasons and weak.get("available") is not False:
        return None
    if weak.get("coarse_fallback_used"):
        return FeedbackCandidate(
            dimension="pitch_naturalness",
            severity="info",
            confidence="low",
            evidence_type="low_f0_coverage",
            location=None,
            user_message="这次逐拍音高证据较少，只显示整体音高变化的低置信度参考。",
            practice_tip="保持自然音量，把完整短句连续读完，音高轨迹会更稳定。",
            caveat="这个数字不是逐拍高低重音判定，也不代表具体重音有错。",
            category="pitch_naturalness",
        )
    return FeedbackCandidate(
        dimension="pitch_naturalness",
        severity="info",
        confidence="high",
        evidence_type="low_f0_coverage",
        location=None,
        user_message="这次能读到的音高信息太少，暂时不显示音高变化判断。",
        practice_tip="保持自然音量，再连续读一遍完整短句。",
        caveat="这里只说明证据不足，不代表音高有问题。",
        category="pitch_naturalness",
    )


def _fluency_candidates(result: Mapping[str, Any]) -> List[FeedbackCandidate]:
    details = _mapping(result.get("details"))
    fluency = _mapping(details.get("fluency"))
    pause_info = _mapping(result.get("pause_info"))
    components = _mapping(fluency.get("delivery_fluency_components"))
    pause_count = int(components.get("long_pause_count") or pause_info.get("pause_count") or 0)
    pause_ratio = _float(components.get("pause_ratio", pause_info.get("pause_ratio")))
    pause_allowance = _float(components.get("natural_pause_ratio_allowance"), 0.18)
    pause_count_excess = int(components.get("pause_count_excess") or 0)
    speech_rate = _float(fluency.get("speech_rate_mora_per_sec"))
    candidates: List[FeedbackCandidate] = []
    if pause_count_excess > 2 or pause_ratio > pause_allowance + 0.08:
        candidates.append(FeedbackCandidate(
            dimension="fluency",
            severity="medium",
            confidence="high",
            evidence_type="long_pause",
            location=None,
            user_message=f"这次有 {pause_count} 次较长停顿，整句听起来不够连续。",
            practice_tip="先找好短语边界，再一口气读完整个短句。",
            category="fluency",
        ))
    if speech_rate >= 8.5:
        candidates.append(FeedbackCandidate(
            dimension="fluency",
            severity="medium",
            confidence="high",
            evidence_type="fast_rate",
            location=None,
            user_message="这次语速偏快，部分拍可能挤在一起。",
            practice_tip="下一次先用大约 80% 的速度读，不要急着连读。",
            category="fluency",
        ))
    elif 0 < speech_rate <= 3.0:
        candidates.append(FeedbackCandidate(
            dimension="fluency",
            severity="mild",
            confidence="medium",
            evidence_type="slow_rate",
            location=None,
            user_message="这次语速偏慢，整句连接感比较弱。",
            practice_tip="先分成两个短语练习，再保持节奏连起来。",
            category="fluency",
        ))
    return candidates


def _special_mora_candidates(
    result: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    *,
    weak_reference: bool,
) -> List[FeedbackCandidate]:
    out: List[FeedbackCandidate] = []
    for item in decisions:
        if not item.get("user_feedback_allowed"):
            continue
        special_type = str(item.get("type") or "")
        mora = str(item.get("surface_mora") or "")
        index = max(0, int(item.get("mora_index") or 1) - 1)
        if special_type == "long_vowel":
            user_message = f"第 {index + 1} 拍「{mora}」可能偏短，长音听起来不够完整。"
            tip = f"把「{_location(result, index, mora)['nearby_kana']}」拆成拍练：前一个音一拍，「{mora}」再留一拍。"
        elif special_type == "moraic_nasal":
            user_message = f"第 {index + 1} 拍「{mora}」可能偏短，拨音不够清楚。"
            tip = f"把「{mora}」单独留出一拍，不要直接滑到后面的音。"
        else:
            continue
        caveat = "这是弱参考下的轻提示，不作严格错误判定。" if weak_reference else None
        out.append(FeedbackCandidate(
            dimension="rhythm_special_mora",
            severity="mild" if weak_reference else "medium",
            confidence=str(item.get("confidence") or "medium"),
            evidence_type="special_mora_duration_issue",
            location=_location(result, index, mora),
            user_message=user_message,
            practice_tip=tip,
            caveat=caveat,
            category="special_mora",
            message=str(item.get("feedback_candidate_text") or user_message),
            type=special_type,
        ))
    return out


def _weak_pitch_candidates(result: Mapping[str, Any], policy: ScoringPolicy, gate: Any) -> List[FeedbackCandidate]:
    if not policy.weak_reference or not gate.allow_pitch_feedback:
        return []
    details = _mapping(result.get("details"))
    weak = _mapping(details.get("weak_reference_native_likeness"))
    if weak.get("available") is False:
        return []
    flatness = _float(weak.get("flatness_penalty"))
    instability = _float(weak.get("instability_penalty"))
    smoothness = _float(weak.get("transition_smoothness"), 1.0)
    components = _mapping(weak.get("component_scores"))
    range_score = _float(components.get("range"), 1.0)
    movement_score = _float(components.get("movement"), 1.0)
    caveat = "音高变化仅供自然度练习参考，不是严格高低重音判定。"
    if flatness >= 0.12 or range_score < 0.55 or movement_score < 0.55:
        return [FeedbackCandidate(
            dimension="pitch_naturalness",
            severity="mild",
            confidence="medium",
            evidence_type="flat_pitch",
            location=None,
            user_message="音高变化仅供参考：这次整体可能有点平。",
            practice_tip="先稍微夸张地做出高低变化，再回到自然程度。",
            caveat=caveat,
            category="pitch_naturalness",
        )]
    if instability >= 0.10 or smoothness < 0.62:
        return [FeedbackCandidate(
            dimension="pitch_naturalness",
            severity="mild",
            confidence="medium",
            evidence_type="unstable_pitch",
            location=None,
            user_message="音高变化仅供参考：这次起伏可能有些不稳定。",
            practice_tip="先听一遍示范，只模仿整句起伏，不要逐拍硬贴。",
            caveat=caveat,
            category="pitch_naturalness",
        )]
    return []


def build_feedback_candidates(
    result: Mapping[str, Any],
    policy: ScoringPolicy,
    gate: Any,
    *,
    special_mora_decisions: Sequence[Mapping[str, Any]],
    max_candidates: int = 2,
) -> List[FeedbackCandidate]:
    blocker = _content_or_evidence_blocker(result, gate)
    if blocker is not None:
        return [blocker]
    low_f0 = _low_f0_candidate(result, gate)
    if low_f0 is not None:
        return [low_f0]

    candidates: List[FeedbackCandidate] = []
    candidates.extend(_fluency_candidates(result))
    candidates.extend(_special_mora_candidates(
        result,
        special_mora_decisions,
        weak_reference=policy.weak_reference,
    ))
    candidates.extend(_weak_pitch_candidates(result, policy, gate))
    if candidates:
        return candidates[:max(1, max_candidates)]
    if policy.weak_reference:
        return [FeedbackCandidate(
            dimension="pronunciation_clarity",
            severity="info",
            confidence="medium",
            evidence_type="clear_recording",
            location=None,
            user_message="这次整体比较清楚，可以继续练习。",
            practice_tip="下一次保持同样速度，把整句连续读完。",
            caveat="这是确认文本后的练习参考，不是教师级评分。",
            category="weak_reference",
            message="確認した文をもとにした練習用フィードバックです。厳密な発音採点ではありません。",
        )]
    return []
