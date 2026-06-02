from __future__ import annotations

import json
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_USER_FACING_MESSAGES: Dict[str, str] = {
    "status.pass": "全体としてよくできています。",
    "status.practice_suggestion": "全体としては問題ありません。より自然にするための練習ポイントがあります。",
    "status.retry": "録音が短すぎるか，音声がはっきり取れていません。もう一度録音してください。",
    "status.debug_only": "このモードでは，厳密な発音判定は行わず，練習用の参考として表示しています。",
    "notice.fixed_verified": "fixed-reference mode: verified target に基づく練習確認です。",
    "notice.fixed_limited": "fixed-reference mode: 信頼できる内容・リズム・流暢さを中心に確認します。",
    "notice.weak_reference": "認識された文をもとにした参考判定です。厳密な発音評価ではありません。",
    "notice.kanade": "これはあなたの声に近い参考音です。声の似ている度合いは採点していません。",
    "special_mora.mild_long": "より自然にするなら，「{mora}」を少し長めに意識するとよいです。",
    "score_policy.alignment_limited_score_cap": "今回は音声の細かい位置合わせが不安定なため，表示スコアは控えめにしています。",
    "score_policy.clear_recording_but_pronunciation_needs_practice": "録音ははっきりしています。ただし，発音の明瞭さにはまだ改善の余地があります。まずは参考音声をゆっくり聞きながら練習してみましょう。",
    "score_policy.content_match_failed_no_pronunciation_score": "目標文との一致が不足しているため，今回は発音スコアを表示しません。",
    "score_policy.demo_only_no_pronunciation_score": "このモードは参考音声のデモです。発音の正しさは採点していません。",
    "score_policy.short_sentence_overall_only": "文が短いため，今回は全体的な練習コメントを中心に表示します。細かい発音の断定は控えます。",
    "score_policy.weak_reference_practice_feedback": "確認した文をもとにした練習用フィードバックです。厳密な発音採点ではありません。",
}


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
    pronunciation_clarity_score: Optional[int]
    rhythm_fluency_score: Optional[int]
    practice_completion_score: Optional[int]
    confidence_label: str
    score_policy_warnings: List[str]
    score_caps: Dict[str, Any]
    detail_feedback_allowed: bool
    user_messages: List[str]
    focus_feedback: Optional[Dict[str, Any]]
    display_total_score: bool
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["practice_score"] = self.practice_score.to_dict()
        return data


def default_user_facing_messages_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "user_facing_messages_ja.json"


@lru_cache(maxsize=4)
def load_user_facing_messages(path: str | Path | None = None) -> Dict[str, str]:
    """Load learner-facing copy.

    Missing config files fall back to conservative defaults so API callers do
    not fail when the demo is embedded in another pipeline.
    """

    messages = dict(DEFAULT_USER_FACING_MESSAGES)
    message_path = Path(path) if path is not None else default_user_facing_messages_path()
    if not message_path.exists():
        return messages
    try:
        raw = json.loads(message_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return messages
    if not isinstance(raw, dict):
        return messages
    for key, value in raw.items():
        if isinstance(value, str) and value.strip():
            messages[str(key)] = value.strip()
    return messages


def user_message(key: str, **kwargs: Any) -> str:
    template = load_user_facing_messages().get(key, DEFAULT_USER_FACING_MESSAGES.get(key, ""))
    try:
        return template.format(**kwargs)
    except (KeyError, ValueError):
        return template


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
