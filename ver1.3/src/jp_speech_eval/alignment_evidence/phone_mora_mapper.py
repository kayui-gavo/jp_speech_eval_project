from __future__ import annotations

from typing import Dict, List, Sequence

from jp_speech_eval.phonology import classify_mora_sequence

from .base import MoraSegment, PhoneSegment


def canonical_special_type(mora_type: str, mora: str) -> str | None:
    if mora_type == "explicit_long_vowel":
        return "long_vowel"
    if mora_type == "sokuon":
        return "sokuon"
    if mora_type == "nasal":
        return "moraic_nasal"
    if any(ch in mora for ch in "ャュョゃゅょ"):
        return "yoon"
    return None


def _expected_phone_count(mora: str, special_type: str | None) -> int:
    if special_type in {"long_vowel", "sokuon", "moraic_nasal"}:
        return 1
    if special_type == "yoon":
        return 2
    if len(mora) == 1 and mora in "アイウエオンーッ":
        return 1
    return 2


def map_phones_to_moras(phone_segments: Sequence[PhoneSegment], moras: Sequence[str]) -> tuple[List[MoraSegment], Dict[str, object]]:
    """Map phone segments to moras with a conservative sequential heuristic."""

    warnings: List[str] = []
    if not phone_segments:
        return [], {
            "mapping_success": False,
            "unmapped_phones": [],
            "unmapped_mora": list(moras),
            "special_mora_mapping_success": False,
            "mapping_warning_flags": ["missing_phone_segments"],
        }
    if not moras:
        return [], {
            "mapping_success": False,
            "unmapped_phones": [p.phone for p in phone_segments],
            "unmapped_mora": [],
            "special_mora_mapping_success": False,
            "mapping_warning_flags": ["missing_moras"],
        }
    segments: List[MoraSegment] = []
    ph_rows = classify_mora_sequence([str(m) for m in moras])
    expected = [_expected_phone_count(str(mora), canonical_special_type(ph_rows[idx].mora_type, ph_rows[idx].mora)) for idx, mora in enumerate(moras)]
    original_expected = list(expected)
    diff = len(phone_segments) - sum(expected)
    idx = 0
    while diff > 0 and expected:
        if ph_rows[idx % len(expected)].mora_type not in {"explicit_long_vowel", "sokuon", "nasal"}:
            expected[idx % len(expected)] += 1
            diff -= 1
        idx += 1
        if idx > len(expected) * 4:
            break
    idx = len(expected) - 1
    while diff < 0 and idx >= 0:
        if expected[idx] > 1:
            expected[idx] -= 1
            diff += 1
        idx -= 1
    cursor = 0
    for idx, mora in enumerate(moras):
        take = expected[idx] if idx < len(expected) else 1
        if idx == len(moras) - 1:
            phones = list(phone_segments[cursor:])
        else:
            phones = list(phone_segments[cursor: cursor + take])
        cursor += len(phones)
        if not phones:
            warnings.append(f"unmapped_mora:{mora}")
            continue
        special_type = canonical_special_type(ph_rows[idx].mora_type, ph_rows[idx].mora)
        segments.append(
            MoraSegment(
                mora=str(mora),
                start=min(p.start for p in phones),
                end=max(p.end for p in phones),
                phones=[p.phone for p in phones],
                special_mora_type=special_type,
                confidence=min(float(p.confidence if p.confidence is not None else 1.0) for p in phones),
            )
        )
    unmapped_phones = [p.phone for p in phone_segments[cursor:]]
    if unmapped_phones:
        warnings.append("unmapped_phones")
    mapped_special = [seg for seg in segments if seg.special_mora_type]
    expected_special = [row for row in ph_rows if canonical_special_type(row.mora_type, row.mora)]
    return segments, {
        "mapping_success": len(segments) == len(moras),
        "unmapped_phones": unmapped_phones,
        "unmapped_mora": [str(m) for m in moras[len(segments):]],
        "special_mora_mapping_success": len(mapped_special) == len(expected_special),
        "mapping_warning_flags": warnings,
        "expected_mora_sequence": [str(m) for m in moras],
        "expected_phone_count_by_mora_initial": original_expected,
        "expected_phone_count_by_mora_adjusted": expected,
        "observed_phone_sequence": [p.phone for p in phone_segments],
        "mora_grouping_rationale": "sequential phone grouping using mora-type expected counts, with global length-difference adjustment",
        "special_mora_mapping_rationale": [
            {
                "mora": seg.mora,
                "special_type": seg.special_mora_type,
                "phones": seg.phones,
                "reason": "special mora inferred from pyopenjtalk mora classification; phones grouped sequentially",
            }
            for seg in mapped_special
        ],
    }
