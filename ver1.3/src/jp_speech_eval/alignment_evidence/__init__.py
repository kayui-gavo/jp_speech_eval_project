from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .alignment_quality import backend_summary
from .base import AlignmentEvidence, MoraSegment, PhoneSegment
from .equal_fallback_adapter import evidence_from_equal_boundaries
from .mfcc_dtw_adapter import evidence_from_mfcc_dtw_boundaries
from .mfa_adapter import evidence_from_lab, evidence_from_textgrid, mfa_available


def _with_failure(ev: AlignmentEvidence, *, method: str, warning: str, failure_reason: str) -> AlignmentEvidence:
    return AlignmentEvidence(
        utterance_id=ev.utterance_id,
        target_text=ev.target_text,
        method=method,
        phone_segments=ev.phone_segments,
        mora_segments=ev.mora_segments,
        word_or_phrase_segments=ev.word_or_phrase_segments,
        alignment_confidence=ev.alignment_confidence,
        fallback_used=ev.fallback_used,
        usable_for_mora_feedback=False,
        usable_for_special_mora_feedback=False,
        usable_for_pitch_feedback=False,
        warning_flags=[warning],
        failure_reason=failure_reason,
    )


def build_alignment_evidence(
    *,
    backend: str,
    utterance_id: str,
    target_text: str,
    moras: Sequence[str],
    mora_table: Sequence[Mapping[str, Any]],
    alignment_cache_dir: str | Path | None = None,
    existing_label_path: str | Path | None = None,
    skip_mfa_if_unavailable: bool = True,
) -> AlignmentEvidence:
    backend = (backend or "auto").strip()
    if backend in {"equal", "equal_fallback"}:
        return evidence_from_equal_boundaries(utterance_id=utterance_id, target_text=target_text, moras=moras, mora_table=mora_table)
    if backend == "mfcc_dtw":
        return evidence_from_mfcc_dtw_boundaries(utterance_id=utterance_id, target_text=target_text, moras=moras, mora_table=mora_table)
    if backend in {"mfa", "mfa_japanese"}:
        if not mfa_available():
            if not skip_mfa_if_unavailable:
                raise RuntimeError("mfa command not found")
            ev = evidence_from_equal_boundaries(utterance_id=utterance_id, target_text=target_text, moras=moras, mora_table=mora_table, method="mfa_japanese_skipped")
            return _with_failure(ev, method="mfa_japanese_skipped", warning="mfa_unavailable", failure_reason="mfa command not found")
        # The actual batch MFA run is handled offline. At item level we only
        # consume cached TextGrid if present.
        if alignment_cache_dir:
            candidate = Path(alignment_cache_dir) / f"{utterance_id}.TextGrid"
            if candidate.exists():
                return evidence_from_textgrid(utterance_id=utterance_id, target_text=target_text, moras=moras, textgrid_path=candidate)
        ev = evidence_from_equal_boundaries(utterance_id=utterance_id, target_text=target_text, moras=moras, mora_table=mora_table, method="mfa_japanese_skipped")
        return _with_failure(ev, method="mfa_japanese_skipped", warning="mfa_textgrid_missing", failure_reason="cached MFA TextGrid not found")
    if backend == "existing_label":
        if existing_label_path and Path(existing_label_path).exists():
            return evidence_from_lab(utterance_id=utterance_id, target_text=target_text, moras=moras, lab_path=existing_label_path)
        if alignment_cache_dir:
            candidate = Path(alignment_cache_dir) / f"{utterance_id}.TextGrid"
            if candidate.exists():
                return evidence_from_textgrid(utterance_id=utterance_id, target_text=target_text, moras=moras, textgrid_path=candidate, method="existing_label")
        ev = evidence_from_equal_boundaries(utterance_id=utterance_id, target_text=target_text, moras=moras, mora_table=mora_table, method="existing_label_missing")
        return _with_failure(ev, method="existing_label_missing", warning="existing_label_missing", failure_reason="existing label not found")
    if backend == "auto":
        if existing_label_path and Path(existing_label_path).exists():
            return evidence_from_lab(utterance_id=utterance_id, target_text=target_text, moras=moras, lab_path=existing_label_path)
        if alignment_cache_dir:
            for method in ("existing_label", "mfa_japanese"):
                candidate = Path(alignment_cache_dir) / f"{utterance_id}.TextGrid"
                if candidate.exists():
                    return evidence_from_textgrid(utterance_id=utterance_id, target_text=target_text, moras=moras, textgrid_path=candidate, method=method)
        if mfa_available():
            ev = evidence_from_equal_boundaries(utterance_id=utterance_id, target_text=target_text, moras=moras, mora_table=mora_table, method="mfa_japanese_skipped")
            return _with_failure(ev, method="mfa_japanese_skipped", warning="mfa_available_but_no_cached_textgrid", failure_reason="cached MFA TextGrid not found")
        return evidence_from_mfcc_dtw_boundaries(utterance_id=utterance_id, target_text=target_text, moras=moras, mora_table=mora_table)
    return evidence_from_equal_boundaries(utterance_id=utterance_id, target_text=target_text, moras=moras, mora_table=mora_table)


__all__ = [
    "AlignmentEvidence",
    "MoraSegment",
    "PhoneSegment",
    "backend_summary",
    "build_alignment_evidence",
    "evidence_from_lab",
    "evidence_from_textgrid",
    "mfa_available",
]
