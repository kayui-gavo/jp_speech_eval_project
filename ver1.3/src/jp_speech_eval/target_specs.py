from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .phonology import classify_mora_sequence
from .text_frontend import split_mora, text_to_kana


TARGET_SPEC_VERSION = "target_pronunciation_spec_v1"


@dataclass(frozen=True)
class TargetSpecValidation:
    ok: bool
    errors: List[str]
    warnings: List[str]

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(self.errors))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_pitch_labels(raw: str) -> List[str]:
    """Parse H/L pitch labels from comma, space, or slash separated input."""

    normalized = raw.replace(",", " ").replace("，", " ").replace("/", " ").replace("・", " ")
    labels = [part.strip().upper() for part in normalized.split() if part.strip()]
    if len(labels) == 1 and len(labels[0]) > 1:
        labels = list(labels[0])
    return labels


def parse_ints(raw: str | None) -> List[int]:
    if not raw:
        return []
    return [int(part.strip()) for part in raw.replace("，", ",").split(",") if part.strip()]


def build_accent_phrases(
    moras: List[str],
    phrase_lengths: List[int],
    accent_positions: List[int],
) -> List[Dict[str, Any]]:
    if not phrase_lengths:
        return []
    if any(length <= 0 for length in phrase_lengths):
        raise ValueError("phrase lengths must be positive")
    if sum(phrase_lengths) != len(moras):
        raise ValueError("phrase lengths must sum to the mora count")
    if accent_positions and len(accent_positions) != len(phrase_lengths):
        raise ValueError("accent positions must match the number of phrase lengths")
    if not accent_positions:
        accent_positions = [0 for _ in phrase_lengths]

    out: List[Dict[str, Any]] = []
    offset = 0
    for length, accent_position in zip(phrase_lengths, accent_positions):
        phrase_moras = moras[offset: offset + length]
        if accent_position < 0 or accent_position > len(phrase_moras):
            raise ValueError("accent position must be within the phrase mora count")
        out.append({
            "words": [],
            "moras": phrase_moras,
            "accent_position": int(accent_position),
            "start_mora_index": offset + 1,
            "end_mora_index": offset + length,
            "chain_flags": [],
            "chain_rules": [],
        })
        offset += length
    return out


def special_mora_metadata(moras: List[str]) -> Dict[str, Any]:
    rows = [row.to_dict() for row in classify_mora_sequence(moras)]
    by_type: Dict[str, List[int]] = {}
    for row in rows:
        mora_type = str(row["mora_type"])
        if mora_type == "normal":
            continue
        by_type.setdefault(mora_type, []).append(int(row["index"]))
    return {
        "items": rows,
        "by_type": by_type,
        "note": "weak vowel_lengthening_candidate labels are diagnostic until manually verified",
    }


def validate_target_spec(entry: Dict[str, Any]) -> TargetSpecValidation:
    errors: List[str] = []
    warnings: List[str] = []
    text = str(entry.get("text") or "")
    kana = str(entry.get("kana") or "")
    moras = list(entry.get("moras") or [])
    target_pitch = [str(label).upper() for label in (entry.get("target_pitch") or [])]
    accent_phrases = list(entry.get("accent_phrases") or [])

    if not text:
        errors.append("text is required")
    if not kana:
        errors.append("kana is required")
    if not moras:
        errors.append("moras are required")
    if len(target_pitch) != len(moras):
        errors.append(f"target_pitch count ({len(target_pitch)}) must match mora count ({len(moras)})")
    if any(label not in {"H", "L"} for label in target_pitch):
        errors.append("target_pitch must contain only H/L labels")

    if accent_phrases:
        phrase_total = 0
        for idx, phrase in enumerate(accent_phrases):
            phrase_moras = list(phrase.get("moras") or [])
            phrase_total += len(phrase_moras)
            accent_position = int(phrase.get("accent_position", 0) or 0)
            if accent_position < 0 or accent_position > len(phrase_moras):
                errors.append(f"accent phrase {idx + 1} has invalid accent_position")
        if phrase_total != len(moras):
            errors.append("accent phrase mora counts must sum to the full mora count")
    else:
        warnings.append("accent_phrases missing; scorer can use pitch labels but phrase-level drop roles are weaker")

    if str(entry.get("pitch_target_source", "")).startswith("ojad") and not entry.get("verification"):
        warnings.append("OJAD source has no verification metadata")

    return TargetSpecValidation(ok=not errors, errors=errors, warnings=warnings)


def build_verified_target_entry(
    *,
    text: str,
    kana: Optional[str],
    target_pitch: List[str],
    source: str,
    phrase_lengths: List[int],
    accent_positions: List[int],
    note: Optional[str] = None,
    source_url: Optional[str] = None,
    verified_by: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_kana = kana or text_to_kana(text)
    moras = split_mora(resolved_kana)
    accent_phrases = build_accent_phrases(moras, phrase_lengths, accent_positions)
    entry: Dict[str, Any] = {
        "schema_version": TARGET_SPEC_VERSION,
        "text": text,
        "kana": resolved_kana,
        "moras": moras,
        "target_pitch": target_pitch,
        "pitch_target_source": source,
        "accent_phrases": accent_phrases,
        "special_mora": special_mora_metadata(moras),
        "verification": {
            "source": source,
            "note": note,
            "source_url": source_url,
            "verified_by": verified_by,
            "verified_at": utc_now_iso(),
        },
    }
    validation = validate_target_spec(entry)
    if validation.warnings:
        entry["warnings"] = validation.warnings
    validation.raise_if_invalid()
    return entry
