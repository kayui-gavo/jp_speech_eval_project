"""Infer construct roles for Japanese target-phone positions.

OpenJTalk's dictionary pronunciation commonly normalizes lexical long vowels to
``ー`` (for example ``ガッコー`` -> ``g a cl k o o``). This gives a useful
construct boundary that a phone-level clarity model should not ignore:

* ordinary mora onset/vowel phones -> ``ordinary_segmental_clarity``;
* ``ン`` / ``ッ`` -> ``special_mora_timing``;
* the phone realizing ``ー`` -> ``long_vowel_timing``.

The classifier is intentionally conservative. It consumes only simple Japanese
mora structure (vowel-only or onset+vowel) and fails closed if the kana/phone
sequence cannot be aligned exactly. It does **not** guess that an orthographic
``ウ`` or ``イ`` is a long-vowel extension when the canonical reading did not
normalize it to ``ー``; reviewed anchors can provide explicit roles instead.

This is target-side construct metadata, not learner acoustic evidence and not a
pronunciation score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

from .japanese_phoneme_gop import logical_phone
from .text_frontend import split_mora


ORDINARY_ROLE = "ordinary_segmental_clarity"
SPECIAL_MORA_ROLE = "special_mora_timing"
LONG_VOWEL_ROLE = "long_vowel_timing"
VOWELS = frozenset({"a", "i", "u", "e", "o"})


@dataclass(frozen=True)
class PhoneConstructRoleResult:
    available: bool
    reading_kana: str
    moras: list[str]
    phones: list[str]
    roles: list[str]
    phone_to_mora_index: list[int]
    summary: Dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _unavailable(
    reading_kana: str,
    phones: Sequence[str],
    moras: Sequence[str],
    reason: str,
) -> PhoneConstructRoleResult:
    return PhoneConstructRoleResult(
        available=False,
        reading_kana=str(reading_kana),
        moras=list(moras),
        phones=[str(phone) for phone in phones],
        roles=[],
        phone_to_mora_index=[],
        summary={
            "reason": str(reason),
            "target_side_only": True,
            "score_mapped": False,
            "product_score_changed": False,
        },
        warnings=[str(reason)],
    )


def infer_phone_construct_roles(
    reading_kana: str,
    phones: Sequence[str],
) -> PhoneConstructRoleResult:
    """Align canonical kana moras to a sanitized Japanese phone sequence.

    ``phones`` should exclude pause/silence/control tokens but may contain
    model-specific high-vowel labels ``I``/``U``; those are treated as their
    logical vowel classes for structural alignment.
    """
    moras = split_mora(str(reading_kana))
    raw_phones = [str(phone) for phone in phones if str(phone)]
    logical = [logical_phone(phone) for phone in raw_phones]
    if not moras:
        return _unavailable(reading_kana, raw_phones, moras, "empty_mora_sequence")
    if not logical:
        return _unavailable(reading_kana, raw_phones, moras, "empty_phone_sequence")

    roles: list[str] = []
    phone_to_mora: list[int] = []
    cursor = 0
    for mora_index, mora in enumerate(moras):
        if cursor >= len(logical):
            return _unavailable(reading_kana, raw_phones, moras, "mora_sequence_outlasts_phone_sequence")

        if mora == "ン":
            if logical[cursor] != "N":
                return _unavailable(reading_kana, raw_phones, moras, f"expected_N_at_mora_{mora_index}")
            roles.append(SPECIAL_MORA_ROLE)
            phone_to_mora.append(mora_index)
            cursor += 1
            continue

        if mora == "ッ":
            if logical[cursor] != "cl":
                return _unavailable(reading_kana, raw_phones, moras, f"expected_cl_at_mora_{mora_index}")
            roles.append(SPECIAL_MORA_ROLE)
            phone_to_mora.append(mora_index)
            cursor += 1
            continue

        if mora == "ー":
            if logical[cursor] not in VOWELS:
                return _unavailable(reading_kana, raw_phones, moras, f"expected_vowel_for_long_mark_at_mora_{mora_index}")
            roles.append(LONG_VOWEL_ROLE)
            phone_to_mora.append(mora_index)
            cursor += 1
            continue

        # Ordinary Japanese mora: vowel-only (one token) or onset+vowel (two
        # tokens). Palatalized onsets such as ky/by/my are already one logical
        # phone token in the pinned phone inventories.
        if logical[cursor] in VOWELS:
            roles.append(ORDINARY_ROLE)
            phone_to_mora.append(mora_index)
            cursor += 1
            continue

        if cursor + 1 >= len(logical) or logical[cursor + 1] not in VOWELS:
            return _unavailable(reading_kana, raw_phones, moras, f"ordinary_mora_not_onset_plus_vowel_at_{mora_index}")
        roles.extend([ORDINARY_ROLE, ORDINARY_ROLE])
        phone_to_mora.extend([mora_index, mora_index])
        cursor += 2

    if cursor != len(logical):
        return _unavailable(reading_kana, raw_phones, moras, "phone_sequence_outlasts_mora_sequence")

    return PhoneConstructRoleResult(
        available=True,
        reading_kana=str(reading_kana),
        moras=moras,
        phones=raw_phones,
        roles=roles,
        phone_to_mora_index=phone_to_mora,
        summary={
            "method": "canonical_kana_mora_to_phone_structure_v1",
            "ordinary_segmental_phone_count": sum(role == ORDINARY_ROLE for role in roles),
            "special_mora_phone_count": sum(role == SPECIAL_MORA_ROLE for role in roles),
            "long_vowel_timing_phone_count": sum(role == LONG_VOWEL_ROLE for role in roles),
            "orthographic_u_i_long_vowel_guessing_used": False,
            "requires_exact_kana_phone_alignment": True,
            "target_side_only": True,
            "score_mapped": False,
            "product_score_changed": False,
        },
        warnings=[],
    )
