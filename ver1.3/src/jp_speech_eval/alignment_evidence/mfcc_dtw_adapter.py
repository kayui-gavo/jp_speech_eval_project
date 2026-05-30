from __future__ import annotations

from typing import Any, Mapping, Sequence

from jp_speech_eval.phonology import classify_mora_sequence

from .base import AlignmentEvidence, MoraSegment
from .phone_mora_mapper import canonical_special_type


def evidence_from_mfcc_dtw_boundaries(
    *,
    utterance_id: str,
    target_text: str,
    moras: Sequence[str],
    mora_table: Sequence[Mapping[str, Any]],
) -> AlignmentEvidence:
    segments = []
    phonology = classify_mora_sequence([str(m) for m in moras])
    for idx, mora in enumerate(moras):
        row = mora_table[idx] if idx < len(mora_table) else {}
        start = float(row.get("start_sec", 0.0) or 0.0)
        end = float(row.get("end_sec", start) or start)
        special_type = canonical_special_type(phonology[idx].mora_type, phonology[idx].mora) if idx < len(phonology) else None
        segments.append(MoraSegment(mora=str(mora), start=start, end=end, phones=[], special_mora_type=special_type, confidence=0.55))
    return AlignmentEvidence(
        utterance_id=utterance_id,
        target_text=target_text,
        method="mfcc_dtw",
        mora_segments=segments,
        alignment_confidence=0.55,
        fallback_used=False,
        usable_for_mora_feedback=True,
        usable_for_special_mora_feedback=False,
        usable_for_pitch_feedback=False,
        warning_flags=["mfcc_dtw_has_no_phone_level_special_mora_evidence"],
        failure_reason="not reliable enough for special mora threshold calibration",
    )
