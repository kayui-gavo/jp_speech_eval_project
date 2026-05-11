from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

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


def _pattern_from_accent_phrase(mora_count: int, accent_position: int) -> List[str]:
    if mora_count <= 0:
        return []
    acc = max(0, int(accent_position))
    if mora_count == 1:
        return ["H" if acc == 1 else "L"]
    if acc == 1:
        return ["H"] + ["L"] * (mora_count - 1)
    pattern = ["L"] + ["H"] * (mora_count - 1)
    if acc > 1:
        for i in range(acc, mora_count):
            pattern[i] = "L"
    return pattern


def _frontend_accent_phrases(feats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group OpenJTalk frontend words into accent phrases.

    `chain_flag == 1` marks a word that chains to the previous word, which is
    important for Japanese particles and auxiliaries. Treating every frontend
    word as a fresh accent phrase would incorrectly reset pitch at boundaries
    like noun + particle or verb + ます.
    """
    phrases: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    for feat in feats:
        pron = kata_normalize(str(feat.get("pron") or feat.get("pronunciation") or ""))
        if not pron:
            continue
        if current and _safe_int(feat.get("chain_flag", 0), default=0) != 1:
            phrases.append({"features": current})
            current = []
        current.append(feat)
    if current:
        phrases.append({"features": current})

    out: List[Dict[str, Any]] = []
    for phrase in phrases:
        phrase_feats = phrase["features"]
        phrase_moras: List[str] = []
        words: List[str] = []
        accent_position = 0
        offset = 0
        for idx, feat in enumerate(phrase_feats):
            pron = kata_normalize(str(feat.get("pron") or feat.get("pronunciation") or ""))
            word_moras = split_mora(pron)
            words.append(str(feat.get("string") or feat.get("orig") or pron))
            if idx == 0:
                accent_position = _safe_int(feat.get("acc", 0), default=0)
            elif accent_position == 0:
                # If the phrase head is heiban, keep the phrase heiban. Otherwise
                # use a later non-zero accent only as a weak fallback for unusual
                # frontend output.
                later_acc = _safe_int(feat.get("acc", 0), default=0)
                if later_acc > len(word_moras):
                    accent_position = offset + later_acc
            phrase_moras.extend(word_moras)
            offset += len(word_moras)
        if accent_position > len(phrase_moras):
            accent_position = len(phrase_moras)
        out.append({
            "words": words,
            "moras": phrase_moras,
            "accent_position": accent_position,
            "chain_flags": [_safe_int(f.get("chain_flag", 0), default=0) for f in phrase_feats],
            "chain_rules": [str(f.get("chain_rule", "")) for f in phrase_feats],
        })
    return out


def phrase_aware_target_pitch_pattern(text: str, moras: List[str]) -> Tuple[List[str], str, List[Dict[str, Any]]]:
    """
    Approximate sentence-level H/L pitch using OpenJTalk accent-phrase chaining.

    This is still a tool-generated target, not ground truth. It is better than
    word-by-word labels because it keeps particles/auxiliaries inside the same
    accent phrase when OpenJTalk marks them with `chain_flag == 1`.
    """
    if not moras:
        return [], "empty", []

    try:
        feats = run_frontend(text)
    except Exception:
        return ["L"] + ["H"] * (len(moras) - 1), "heuristic_fallback", []

    pattern: List[str] = []
    phrases = _frontend_accent_phrases(feats)
    for phrase in phrases:
        pattern.extend(_pattern_from_accent_phrase(len(phrase["moras"]), int(phrase["accent_position"])))

    if not pattern:
        pattern = ["L"] + ["H"] * (len(moras) - 1)
        source = "heuristic_fallback"
    else:
        source = "openjtalk_accent_phrase_chain"

    if len(pattern) < len(moras):
        pattern.extend([pattern[-1]] * (len(moras) - len(pattern)))
    return pattern[: len(moras)], source, phrases


def rough_target_pitch_pattern(text: str, moras: List[str]) -> List[str]:
    pattern, _source, _phrases = phrase_aware_target_pitch_pattern(text, moras)
    return pattern


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
    accent_phrases: List[Dict[str, Any]]


def build_text_info(text: str) -> TextInfo:
    kana = text_to_kana(text)
    moras = split_mora(kana)
    target_pitch, pitch_target_source, accent_phrases = phrase_aware_target_pitch_pattern(text, moras)
    return TextInfo(
        text=text,
        kana=kana,
        moras=moras,
        target_pitch=target_pitch,
        pitch_target_source=pitch_target_source,
        is_question=is_question_sentence(text, kana),
        accent_phrases=accent_phrases,
    )
