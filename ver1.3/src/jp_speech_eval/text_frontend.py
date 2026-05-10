from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

SMALL_KANA = set("ァィゥェォャュョヮぁぃぅぇぉゃゅょゎ")
PUNCT = set("、。！？!?,.・「」『』（）()[]【】 　\n\t'’")


def hira_to_kata(s: str) -> str:
    out = []
    for ch in s:
        code = ord(ch)
        if 0x3041 <= code <= 0x3096:
            out.append(chr(code + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def kata_normalize(s: str) -> str:
    s = hira_to_kata(s)
    return "".join(ch for ch in s if ch not in PUNCT)


def split_mora(kana: str) -> List[str]:
    """
    Split katakana pronunciation into mora units.

    Examples:
      ラーメン -> ラ・ー・メ・ン
      キョウ -> キョ・ウ
      ガッコウ -> ガ・ッ・コ・ウ
    """
    kana = kata_normalize(kana)
    moras: List[str] = []
    for ch in kana:
        if ch in SMALL_KANA and moras:
            moras[-1] += ch
        else:
            moras.append(ch)
    return moras


def text_to_kana(text: str) -> str:
    """Convert Japanese text to katakana pronunciation using pyopenjtalk."""
    try:
        import pyopenjtalk
    except ImportError as exc:
        raise RuntimeError("pyopenjtalk is not installed. Run: pip install -r requirements.txt") from exc

    try:
        kana = pyopenjtalk.g2p(text, kana=True, join=True)
    except TypeError:
        # Older versions may not accept join=True.
        kana = "".join(pyopenjtalk.g2p(text, kana=True))
    return kata_normalize(kana)


def run_frontend(text: str) -> List[Dict[str, Any]]:
    try:
        import pyopenjtalk
    except ImportError as exc:
        raise RuntimeError("pyopenjtalk is not installed. Run: pip install -r requirements.txt") from exc
    return pyopenjtalk.run_frontend(text)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def rough_target_pitch_pattern(text: str, moras: List[str]) -> List[str]:
    """
    Approximate target H/L pitch pattern using OpenJTalk frontend accent info.

    This is intentionally simple and product-prototype oriented.
    It should later be replaced/validated by OJAD/OpenJTalk accent labels,
    a native-speaker reference recording, or manually curated lesson data.
    """
    if not moras:
        return []

    try:
        feats = run_frontend(text)
    except Exception:
        return ["L"] + ["H"] * (len(moras) - 1)

    pattern: List[str] = []
    for f in feats:
        pron = f.get("pron") or f.get("pronunciation") or ""
        pron = kata_normalize(str(pron))
        if not pron:
            continue
        word_moras = split_mora(pron)
        n = len(word_moras)
        if n == 0:
            continue

        acc = _safe_int(f.get("acc", 0), default=0)
        if n == 1:
            word_pattern = ["H" if acc == 1 else "L"]
        elif acc == 1:
            word_pattern = ["H"] + ["L"] * (n - 1)
        else:
            word_pattern = ["L"] + ["H"] * (n - 1)
            if acc > 1:
                # acc is treated as 1-indexed accent nucleus. After it, pitch drops.
                for i in range(acc, n):
                    word_pattern[i] = "L"
        pattern.extend(word_pattern)

    if not pattern:
        pattern = ["L"] + ["H"] * (len(moras) - 1)

    if len(pattern) < len(moras):
        pattern.extend([pattern[-1]] * (len(moras) - len(pattern)))
    return pattern[: len(moras)]


def is_question_sentence(text: str, kana: str | None = None) -> bool:
    if "?" in text or "？" in text:
        return True
    kana = kana or text_to_kana(text)
    # Very rough heuristic: many polite questions end with カ.
    return kata_normalize(kana).endswith("カ")


@dataclass(frozen=True)
class TextInfo:
    text: str
    kana: str
    moras: List[str]
    target_pitch: List[str]
    pitch_target_source: str
    is_question: bool


def build_text_info(text: str) -> TextInfo:
    kana = text_to_kana(text)
    moras = split_mora(kana)
    target_pitch = rough_target_pitch_pattern(text, moras)
    return TextInfo(
        text=text,
        kana=kana,
        moras=moras,
        target_pitch=target_pitch,
        pitch_target_source="heuristic",
        is_question=is_question_sentence(text, kana),
    )
