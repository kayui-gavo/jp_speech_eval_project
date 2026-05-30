from __future__ import annotations

from .base import AlignmentEvidence


def backend_summary(evidence: AlignmentEvidence) -> dict:
    return {
        "method": evidence.method,
        "success": evidence.failure_reason is None,
        "fallback_used": evidence.fallback_used,
        "alignment_confidence": evidence.alignment_confidence,
        "usable_for_special_mora_feedback": evidence.usable_for_special_mora_feedback,
        "warning_flags": "|".join(evidence.warning_flags),
        "failure_reason": evidence.failure_reason or "",
    }
