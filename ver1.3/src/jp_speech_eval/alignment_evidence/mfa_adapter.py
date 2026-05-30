from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from .base import AlignmentEvidence
from .phone_mora_mapper import map_phones_to_moras
from .textgrid_parser import parse_lab_phone_segments, parse_textgrid_phone_segments


def mfa_available() -> bool:
    return shutil.which("mfa") is not None


def evidence_from_textgrid(
    *,
    utterance_id: str,
    target_text: str,
    moras: Sequence[str],
    textgrid_path: str | Path,
    method: str = "mfa_japanese",
) -> AlignmentEvidence:
    phones = parse_textgrid_phone_segments(textgrid_path)
    mora_segments, mapping = map_phones_to_moras(phones, moras)
    warnings = list(mapping.get("mapping_warning_flags") or [])
    usable = bool(phones and mora_segments and mapping.get("mapping_success") and mapping.get("special_mora_mapping_success"))
    return AlignmentEvidence(
        utterance_id=utterance_id,
        target_text=target_text,
        method=method,
        phone_segments=list(phones),
        mora_segments=mora_segments,
        alignment_confidence=0.9 if usable else 0.45,
        fallback_used=False,
        usable_for_mora_feedback=bool(mora_segments),
        usable_for_special_mora_feedback=usable,
        usable_for_pitch_feedback=usable,
        warning_flags=warnings,
        failure_reason=None if usable else "phone_to_mora_mapping_incomplete",
    )


def evidence_from_lab(
    *,
    utterance_id: str,
    target_text: str,
    moras: Sequence[str],
    lab_path: str | Path,
    method: str = "existing_label",
) -> AlignmentEvidence:
    phones = parse_lab_phone_segments(lab_path)
    mora_segments, mapping = map_phones_to_moras(phones, moras)
    warnings = list(mapping.get("mapping_warning_flags") or [])
    usable = bool(phones and mora_segments and mapping.get("mapping_success") and mapping.get("special_mora_mapping_success"))
    return AlignmentEvidence(
        utterance_id=utterance_id,
        target_text=target_text,
        method=method,
        phone_segments=list(phones),
        mora_segments=mora_segments,
        alignment_confidence=0.85 if usable else 0.45,
        fallback_used=False,
        usable_for_mora_feedback=bool(mora_segments),
        usable_for_special_mora_feedback=usable,
        usable_for_pitch_feedback=usable,
        warning_flags=warnings,
        failure_reason=None if usable else "lab_phone_to_mora_mapping_incomplete",
    )


def run_mfa_align(
    *,
    corpus_dir: str | Path,
    output_dir: str | Path,
    dictionary: str = "japanese_mfa",
    acoustic_model: str = "japanese_mfa",
    timeout_sec: int = 600,
) -> tuple[bool, str]:
    if not mfa_available():
        return False, "mfa command not found"
    cmd = ["mfa", "align", str(corpus_dir), dictionary, acoustic_model, str(output_dir), "--clean"]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout_sec)
    except Exception as exc:
        return False, str(exc)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "mfa align failed").strip()
    return True, proc.stdout.strip()
