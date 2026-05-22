from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


VOWEL_BY_KANA: Dict[str, str] = {}
for _chars, _vowel in [
    ("アカサタナハマヤラワガザダバパァャヮ", "a"),
    ("イキシチニヒミリギジヂビピィ", "i"),
    ("ウクスツヌフムユルグズヅブプゥュヴ", "u"),
    ("エケセテネヘメレゲゼデベペェ", "e"),
    ("オコソトノホモヨロヲゴゾドボポォョ", "o"),
]:
    for _ch in _chars:
        VOWEL_BY_KANA[_ch] = _vowel

PURE_VOWEL_MORA = {
    "ア": "a",
    "イ": "i",
    "ウ": "u",
    "エ": "e",
    "オ": "o",
}


@dataclass(frozen=True)
class MoraPhonology:
    index: int
    mora: str
    vowel: Optional[str]
    mora_type: str
    duration_role: str
    strength: str
    note: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def mora_vowel(mora: str) -> Optional[str]:
    """Return the vowel class of a katakana mora when it is inferable."""

    if not mora:
        return None
    for ch in reversed(mora):
        if ch == "ー":
            continue
        vowel = VOWEL_BY_KANA.get(ch)
        if vowel:
            return vowel
    return None


def _is_vowel_lengthening_candidate(prev_vowel: Optional[str], mora: str) -> bool:
    current = PURE_VOWEL_MORA.get(mora)
    if prev_vowel is None or current is None:
        return False
    if current == prev_vowel:
        return True
    # Common kana spellings for long /o:/ and /e:/ in Japanese reading.
    if prev_vowel == "o" and current == "u":
        return True
    if prev_vowel == "e" and current == "i":
        return True
    return False


def classify_mora_sequence(moras: List[str]) -> List[MoraPhonology]:
    """Annotate morae with duration-sensitive phonological roles.

    The labels are deliberately conservative:
    - explicit_long_vowel, sokuon, and nasal are strong special mora evidence;
    - vowel_lengthening_candidate is weak evidence for spellings such as おう,
      えい, and ああ. It is useful for diagnostics but should not be treated as
      hard phoneme correctness without lexical or forced-alignment support.
    """

    rows: List[MoraPhonology] = []
    prev_vowel: Optional[str] = None
    for i, mora in enumerate(moras):
        vowel = mora_vowel(mora)
        if mora == "ー":
            mora_type = "explicit_long_vowel"
            role = "lengthened_vowel"
            strength = "strong"
            note = "katakana_long_vowel_mark"
        elif mora == "ッ":
            mora_type = "sokuon"
            role = "geminate_closure"
            strength = "strong"
            note = "sokuon_requires_timing_hold"
        elif mora == "ン":
            mora_type = "nasal"
            role = "moraic_nasal"
            strength = "strong"
            note = "moraic_nasal_requires_duration"
        elif _is_vowel_lengthening_candidate(prev_vowel, mora):
            mora_type = "vowel_lengthening_candidate"
            role = "possible_lengthened_vowel"
            strength = "weak"
            note = "kana_sequence_often_realized_as_long_vowel_but_needs_lexical_confirmation"
        else:
            mora_type = "normal"
            role = "plain_mora"
            strength = "none"
            note = ""
        rows.append(
            MoraPhonology(
                index=i + 1,
                mora=mora,
                vowel=vowel,
                mora_type=mora_type,
                duration_role=role,
                strength=strength,
                note=note,
            )
        )
        if mora not in {"ー", "ッ", "ン"}:
            prev_vowel = vowel
    return rows
