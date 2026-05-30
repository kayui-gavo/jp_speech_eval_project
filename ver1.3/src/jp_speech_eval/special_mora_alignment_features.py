from __future__ import annotations

from typing import Any, Dict, List

from .alignment_evidence import AlignmentEvidence


def extract_special_mora_alignment_features(evidence: AlignmentEvidence) -> List[Dict[str, Any]]:
    """Extract special-mora timing features only from usable alignment evidence."""

    rows: List[Dict[str, Any]] = []
    durations = [max(0.0, seg.end - seg.start) for seg in evidence.mora_segments]
    avg = sum(durations) / len(durations) if durations else 0.0
    for idx, seg in enumerate(evidence.mora_segments):
        if not seg.special_mora_type:
            continue
        duration = durations[idx] if idx < len(durations) else 0.0
        neighbors = [durations[j] for j in (idx - 1, idx + 1) if 0 <= j < len(durations)]
        neighbor_avg = sum(neighbors) / len(neighbors) if neighbors else avg
        reliable = bool(evidence.usable_for_special_mora_feedback)
        ratio_avg = duration / max(avg, 1e-8) if avg > 0 else None
        ratio_neighbor = duration / max(neighbor_avg, 1e-8) if neighbor_avg > 0 else None
        row: Dict[str, Any] = {
            "special_type": seg.special_mora_type,
            "mora_index": idx + 1,
            "mora": seg.mora,
            "duration_sec": round(duration, 4),
            "ratio_to_avg_mora": None if ratio_avg is None else round(float(ratio_avg), 4),
            "ratio_to_neighbor_mora": None if ratio_neighbor is None else round(float(ratio_neighbor), 4),
            "evidence_confidence": float(seg.confidence if seg.confidence is not None else evidence.alignment_confidence),
            "judgement_available": reliable,
            "alignment_method": evidence.method,
            "fallback_used": evidence.fallback_used,
            "alignment_unsafe_for_threshold": not reliable,
            "uncertain": not reliable,
            "warning_flags": "|".join(evidence.warning_flags),
        }
        if seg.special_mora_type == "long_vowel":
            row.update({
                "long_vowel_duration": row["duration_sec"],
                "long_vowel_ratio_to_neighbor_mora": row["ratio_to_neighbor_mora"],
                "long_vowel_ratio_to_avg_mora": row["ratio_to_avg_mora"],
            })
        elif seg.special_mora_type == "sokuon":
            row.update({
                "closure_duration": row["duration_sec"],
                "closure_ratio_to_neighbor_mora": row["ratio_to_neighbor_mora"],
                "following_consonant_duration": durations[idx + 1] if idx + 1 < len(durations) else None,
            })
        elif seg.special_mora_type == "moraic_nasal":
            row.update({
                "nasal_mora_duration": row["duration_sec"],
                "nasal_ratio_to_avg_mora": row["ratio_to_avg_mora"],
                "nasal_phone_duration": row["duration_sec"] if seg.phones else None,
            })
        elif seg.special_mora_type == "yoon":
            row.update({
                "yoon_mora_duration": row["duration_sec"],
                "yoon_mora_count_consistency": 1,
                "phone_group_duration": row["duration_sec"] if seg.phones else None,
            })
        rows.append(row)
    return rows
