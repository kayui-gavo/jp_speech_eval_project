from __future__ import annotations

import json
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_USER_FACING_MESSAGES: Dict[str, str] = {
    "status.pass": "全体としてよくできています。",
    "status.practice_suggestion": "今回の結果をもとに、次に意識するとよいポイントがあります。",
    "status.retry": "録音をうまく確認できませんでした。もう一度録音してください。",
    "status.debug_only": "このモードは参考表示です。",
    "notice.fixed_verified": "目標文と参考音声をもとに、発音・リズム・流暢さ・抑揚を確認します。",
    "notice.fixed_limited": "今回は信頼できる項目を中心に評価しています。",
    "notice.general_japanese": "目標文との細かい比較ではなく、日本語としての全体的な話し方を評価しています。",
    "notice.general_japanese_fallback": "目標文とは違う内容でしたが、日本語としての全体的な話し方を評価しています。",
    "notice.weak_reference": "確認した文をもとにした練習用の目安です。細かいアクセント判定は行いません。",
    "notice.kanade": "これはあなたの声に近い参考音です。声の似ている度合いは採点していません。",
    "special_mora.mild_long": "より自然にするなら，「{mora}」を少し長めに意識するとよいです。",
    "score_policy.clear_recording_but_pronunciation_needs_practice": "録音ははっきりしています。発音をもう少し整えると、さらに聞き取りやすくなります。",
    "score_policy.content_mismatch_general_score": "目標文とは違う内容でしたが、日本語としての全体的な話し方は評価しています。",
    "score_policy.broad_score_only": "今回は細かい判定よりも、全体的な話し方を中心に評価しています。",
    "score_policy.invalid_or_non_japanese": "今回は日本語として安定して確認できませんでした。もう一度話してみてください。",
    "score_policy.recording_unusable_no_score": "録音をうまく確認できませんでした。マイクに少し近づいて、もう一度録音してください。",
    "score_policy.demo_only_no_pronunciation_score": "このモードは参考音声のデモです。発音の正しさは採点していません。",
    "score_policy.weak_reference_practice_feedback": "確認した文をもとにした練習用フィードバックです。細かいアクセント判定は控えています。",
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
    """Consumer UI response contract.

    Raw acoustic/debug values remain available under `debug`, while the main
    fields are designed for a stable practice experience.
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
    score_dimensions: List[Dict[str, Any]]
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
    """Load learner-facing copy with local defaults as a safe fallback."""

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
    if value >= 90:
        return "とても良い"
    if value >= 80:
        return "良好"
    if value >= 70:
        return "もう少し"
    if value >= 55:
        return "練習中"
    return "要練習"


def practice_score_explanation(mode_notice: str) -> str:
    return (
        "このスコアは、今回の録音について、発音・リズム・流暢さなどをもとにした"
        "練習用の目安です。正式な試験スコアではありません。"
        f" {mode_notice}".strip()
    )
