from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class FeedbackDecision:
    """Final learner-facing feedback after evidence-aware prioritization."""

    feedback: List[str]
    suppressed: List[str] = field(default_factory=list)
    policy: str = "evidence_aware_v1"

    def to_dict(self) -> Dict[str, object]:
        return {
            "policy": self.policy,
            "feedback": list(self.feedback),
            "suppressed": list(self.suppressed),
        }


RELIABILITY_FEEDBACK = {
    "这次录音里有些地方不够清楚，重录一次会更准。",
    "这次结果只能作参考，建议再录一次确认。",
    "这次录音里的音高信息不够清楚，重录一次会更准。",
    "我不太确定这次是否读对了目标句，建议再试一次。",
}

FLUENCY_FEEDBACK_PREFIXES = (
    "语速",
    "你说得太快",
    "检测到",
)

PRONUNCIATION_FEEDBACK_MARKERS = (
    "这个音可能太短",
    "节奏不太稳定",
)

PROSODY_ACTIONABLE_MARKERS = (
    "附近，音高",
    "拍「",
    "句末语调和示范音不太一致",
    "音高起伏的走向",
    "有些音高变化出现得稍早或稍晚",
)

PROSODY_SUMMARY_MARKERS = (
    "整体音高",
    "前半句的音高",
    "句末语调和示范音比较接近",
    "这次音高细节还不能稳定判断",
)

TONE_MARKERS = (
    "听起来可能偏平",
    "表达可能偏平",
    "语气可能偏紧张",
    "表达可能偏紧张",
    "音量偏小",
    "能量变化较大",
    "语气可能显得犹豫",
)


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _first_matching(items: Sequence[str], predicate) -> str | None:
    for item in items:
        if predicate(item):
            return item
    return None


def _contains_any(text: str, markers: Sequence[str]) -> bool:
    return any(marker in text for marker in markers)


def choose_feedback(
    *,
    raw_feedback: Sequence[str],
    reliability: Dict,
    mora_evidence_summary: Dict,
    max_items: int = 4,
) -> FeedbackDecision:
    """Choose a small, user-facing subset from raw diagnostic messages.

    The evaluator deliberately computes many diagnostic proxies. This policy is
    the product-facing reduction layer: it avoids presenting every proxy as if it
    were an independent error. Strong pronunciation/prosody feedback is allowed
    only when reliability and mora evidence are adequate; otherwise the learner
    receives a concise retry or recording-quality message.
    """

    raw = _dedupe(raw_feedback)
    if not raw:
        return FeedbackDecision(feedback=["整体听起来比较自然。"])

    level = str(reliability.get("level") or "unknown")
    overall = float(reliability.get("overall", 0.0) or 0.0)
    f0_coverage = float(reliability.get("f0_coverage", 0.0) or 0.0)
    judgement_count = int(mora_evidence_summary.get("judgement_available_count", 0) or 0)
    mora_count = int(mora_evidence_summary.get("mora_count", 0) or 0)
    judgement_needed = max(3, int(mora_count * 0.55)) if mora_count else 3
    enough_mora_evidence = judgement_count >= judgement_needed
    enough_pitch_evidence = f0_coverage >= 0.50

    reliability_items = [item for item in raw if item in RELIABILITY_FEEDBACK]
    severe_fluency = _first_matching(
        raw,
        lambda item: item.startswith("你说得太快")
        or "明显偏快" in item
        or item.startswith("语速和自然"),
    )
    ordinary_fluency = severe_fluency or _first_matching(
        raw,
        lambda item: item.startswith(FLUENCY_FEEDBACK_PREFIXES),
    )

    selected: List[str] = []
    if level == "low" or overall < 0.45:
        if reliability_items:
            selected.append(reliability_items[0])
        if severe_fluency:
            selected.append(severe_fluency)
        if not selected:
            selected.append("这次结果只能作参考，建议再录一次确认。")
        selected = _dedupe(selected)[: max(1, max_items)]
        return FeedbackDecision(
            feedback=selected,
            suppressed=[item for item in raw if item not in selected],
        )

    if reliability_items and overall < 0.75:
        selected.append(reliability_items[0])

    if ordinary_fluency:
        selected.append(ordinary_fluency)

    pronunciation_issue = _first_matching(
        raw,
        lambda item: _contains_any(item, PRONUNCIATION_FEEDBACK_MARKERS),
    )
    if pronunciation_issue and enough_mora_evidence:
        selected.append(pronunciation_issue)

    prosody_action = _first_matching(
        raw,
        lambda item: _contains_any(item, PROSODY_ACTIONABLE_MARKERS),
    )
    prosody_summary = _first_matching(
        raw,
        lambda item: _contains_any(item, PROSODY_SUMMARY_MARKERS),
    )
    if enough_pitch_evidence:
        if prosody_action:
            selected.append(prosody_action)
        elif prosody_summary and not selected:
            selected.append(prosody_summary)
    elif reliability_items:
        selected.append(reliability_items[0])

    tone_issue = _first_matching(raw, lambda item: _contains_any(item, TONE_MARKERS))
    # Tone/expression is useful, but it should not crowd out pronunciation or
    # prosody in a pronunciation product.
    if tone_issue and len(selected) < 2 and not pronunciation_issue and not prosody_action:
        selected.append(tone_issue)

    selected = _dedupe(selected)
    if not selected:
        positive = _first_matching(raw, lambda item: "自然" in item or "比较接近" in item)
        selected = [positive or "整体听起来比较自然。"]

    selected = selected[:max_items]
    return FeedbackDecision(
        feedback=selected,
        suppressed=[item for item in raw if item not in selected],
    )
