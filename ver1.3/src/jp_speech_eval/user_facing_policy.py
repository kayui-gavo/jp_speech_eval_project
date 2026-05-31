from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PracticeScore:
    """Product-facing practice guidance, not a validated pronunciation score."""

    value: Optional[int]
    label: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UserFacingResult:
    """Safe response contract for consumer UI.

    Raw scores and acoustic diagnostics may exist in `debug`, but UI should
    prefer these fields to avoid presenting proxy metrics as scientific truth.
    """

    mode: str
    status: str
    reliability: str
    confidence: str
    practice_check_result: str
    practice_score: PracticeScore
    summary_text: str
    primary_suggestion_text: Optional[str]
    suggestion_type: str
    mode_notice: str
    debug_available: bool
    suppressed_reasons: List[str]
    display_score: Optional[int]
    user_messages: List[str]
    focus_feedback: Optional[Dict[str, Any]]
    display_total_score: bool
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["practice_score"] = self.practice_score.to_dict()
        return data


def practice_score_label(value: Optional[int], status: str) -> str:
    if status == "retry":
        return "録音を確認"
    if value is None:
        return "判定できません"
    if value >= 85:
        return "良好"
    if value >= 70:
        return "もう少し"
    return "録音を確認"


def practice_score_explanation(mode_notice: str) -> str:
    return (
        "このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした"
        "練習用の目安です。発音能力そのものを厳密に評価するものではありません。"
        f" {mode_notice}".strip()
    )
