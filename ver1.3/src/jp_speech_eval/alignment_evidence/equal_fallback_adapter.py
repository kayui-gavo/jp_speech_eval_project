from __future__ import annotations

from typing import Any, Mapping, Sequence

from jp_speech_eval.phonology import classify_mora_sequence

from .base import AlignmentEvidence, MoraSegment
from .phone_mora_mapper import canonical_special_type


def evidence_from_equal_boundaries(
    *,
    utterance_id: str,
    target_text: str,
    moras: Sequence[str],
    mora_table: Sequence[Mapping[str, Any]],
    method: str = "equal_fallback",
) -> AlignmentEvidence:
    segments = []
    phonology = classify_mora_sequence([str(m) for m in moras])
    for idx, mora in enumerate(moras):
        row = mora_table[idx] if idx < len(mora_table) else {}
        start = float(row.get("start_sec", 0.0) or 0.0)
        end = float(row.get("end_sec", start) or start)
        special_type = canonical_special_type(phonology[idx].mora_type, phonology[idx].mora) if idx < len(phonology) else None
        segments.append(MoraSegment(mora=str(mora), start=start, end=end, phones=[], special_mora_type=special_type, confidence=0.0))
    return AlignmentEvidence(
        utterance_id=utterance_id,
        target_text=target_text,
        method=method,
        mora_segments=segments,
        alignment_confidence=0.35,
        fallback_used=True,
        usable_for_mora_feedback=False,
        usable_for_special_mora_feedback=False,
        usable_for_pitch_feedback=False,
        warning_flags=["equal_fallback_not_usable_for_special_mora_or_pitch"],
        failure_reason="equal fallback has no phone-level timing evidence",
    )
