from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


ANNOTATION_FIELDS = [
    "teacher_pronunciation_score",
    "teacher_prosody_score",
    "teacher_fluency_score",
    "native_naturalness_score",
    "listener_intelligibility_score",
    "error_long_vowel",
    "error_sokuon",
    "error_nasal",
    "error_pitch_accent",
]


@dataclass
class UnifiedEvaluationResult:
    mode: str
    input_info: Dict[str, Any]
    features: Dict[str, Any]
    scores: Dict[str, Any]
    reliability: Dict[str, Any]
    warnings: List[str]
    feedback: List[str]
    debug: Dict[str, Any]
    latency_ms: float
    raw_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_log_record(self) -> Dict[str, Any]:
        record = self.to_dict()
        record.update({field_name: None for field_name in ANNOTATION_FIELDS})
        return record


def _result_to_dict(result: Any) -> Dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if isinstance(result, Mapping):
        return dict(result)
    raise TypeError(f"Unsupported evaluation result type: {type(result).__name__}")


def _pick(d: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _flatten_selected(result: Dict[str, Any]) -> Dict[str, Any]:
    details = result.get("details", {}) if isinstance(result.get("details"), Mapping) else {}
    prosody = result.get("prosody_metrics") or details.get("prosody_metrics") or details.get("prosody") or {}
    acoustic = details.get("acoustic_features") or {}
    structure = details.get("structure_features") or {}
    recording = details.get("recording_quality") or {}
    content = details.get("content_match") or {}
    alignment = details.get("alignment") or {}
    evidence = details.get("mora_evidence_summary") or {}
    reliability = details.get("reliability") or {}
    fluency = details.get("fluency") or {}
    pronunciation = details.get("pronunciation") or {}
    tone = details.get("tone") or {}
    endpointing = result.get("endpointing") or details.get("endpointing") or {}
    pause = result.get("pause_info") or {}

    features: Dict[str, Any] = {
        "duration_sec": result.get("duration_sec"),
        "raw_duration_sec": endpointing.get("raw_duration"),
        "speech_duration_sec": endpointing.get("speech_duration") or acoustic.get("speech_duration_sec"),
        "speech_ratio": endpointing.get("speech_ratio"),
        "leading_silence_sec": endpointing.get("leading_silence"),
        "trailing_silence_sec": endpointing.get("trailing_silence"),
        "pause_ratio": pause.get("pause_ratio") or acoustic.get("pause_ratio"),
        "pause_count": pause.get("pause_count") or acoustic.get("pause_count"),
        "voiced_ratio": acoustic.get("voiced_ratio"),
        "f0_mean_hz": acoustic.get("f0_mean_hz"),
        "f0_std_hz": acoustic.get("f0_std_hz"),
        "relative_log_f0_range": acoustic.get("relative_log_f0_range") or tone.get("pitch_range_log"),
        "energy_mean": recording.get("energy_mean") or _pick(tone, "energy", "mean"),
        "energy_cv": recording.get("energy_cv") or _pick(tone, "energy", "cv"),
        "recording_quality_score": recording.get("score"),
        "recording_snr_db": recording.get("snr_db"),
        "recording_noise_rms": recording.get("noise_rms"),
        "recording_dynamic_range_db": recording.get("dynamic_range_db"),
        "clipping_ratio": recording.get("clipping_ratio"),
        "mora_evidence_score": reliability.get("mora_evidence"),
        "mora_judgement_available_count": evidence.get("judgement_available_count"),
        "mora_prosody_available_count": evidence.get("prosody_available_count"),
        "mean_mora_boundary_confidence": evidence.get("mean_boundary_confidence"),
        "mean_mora_energy_coverage": evidence.get("mean_energy_coverage"),
        "mean_mora_f0_coverage": evidence.get("mean_f0_coverage"),
        "special_mora_judgement_available_count": evidence.get("special_mora_judgement_available_count"),
        "mora_count": structure.get("mora_count") or len(result.get("moras") or []),
        "long_vowel_count": structure.get("long_vowel_count"),
        "sokuon_count": structure.get("sokuon_count"),
        "nasal_count": structure.get("nasal_count"),
        "special_mora_count": structure.get("special_mora_count"),
        "special_mora_density": structure.get("special_mora_density"),
        "speech_rate_mora_per_sec": fluency.get("speech_rate_mora_per_sec") or structure.get("mora_rate"),
        "avg_mora_duration_sec": fluency.get("avg_mora_duration_sec") or structure.get("avg_mora_duration_sec"),
        "normalized_f0_range": structure.get("normalized_f0_range"),
        "normalized_f0_slope": structure.get("normalized_f0_slope"),
        "f0_direction_change_rate": structure.get("f0_direction_change_rate"),
        "f0_rise_ratio": structure.get("f0_rise_ratio"),
        "f0_fall_ratio": structure.get("f0_fall_ratio"),
        "f0_flat_ratio": structure.get("f0_flat_ratio"),
        "final_f0_movement": structure.get("final_f0_movement"),
        "too_fast_for_special_mora": structure.get("too_fast_for_special_mora"),
        "compressed_mora_risk": structure.get("compressed_mora_risk"),
        "low_voicing_risk": structure.get("low_voicing_risk"),
        "high_pause_risk": structure.get("high_pause_risk"),
        "mora_duration_cv": pronunciation.get("mora_duration_cv"),
        "special_mora_penalty": pronunciation.get("special_mora_penalty"),
        "contour_corr": prosody.get("contour_corr"),
        "contour_rmse": prosody.get("contour_rmse"),
        "transition_agreement": prosody.get("transition_agreement"),
        "final_intonation_match": prosody.get("final_intonation_match"),
        "hl_match_rate": prosody.get("hl_match_rate") or prosody.get("hl_match"),
        "pitch_target_source": prosody.get("pitch_target_source"),
        "pitch_target_consistency": prosody.get("pitch_target_consistency"),
        "alignment_mode": result.get("alignment_mode") or alignment.get("mode"),
        "alignment_boundary_cv": alignment.get("boundary_duration_cv"),
        "dtw_cost": content.get("dtw_cost"),
        "duration_ratio": content.get("duration_ratio"),
        "kana_similarity": content.get("kana_similarity"),
        "content_match_status": content.get("status"),
        "asr_provider": content.get("asr_provider") or _pick(details, "asr", "provider"),
    }
    return features


def unify_evaluation_result(
    result: Any,
    *,
    mode: Optional[str] = None,
    audio_path: str | Path | None = None,
    target_text: str | None = None,
    asr_transcript: str | None = None,
    extra_input: Optional[Dict[str, Any]] = None,
) -> UnifiedEvaluationResult:
    raw = _result_to_dict(result)
    details = raw.get("details", {}) if isinstance(raw.get("details"), Mapping) else {}
    inferred_mode = (
        mode
        or details.get("mode")
        or raw.get("mode")
        or ("reference_based" if raw.get("moras") else "unknown")
    )
    timing = raw.get("timing", {}) if isinstance(raw.get("timing"), Mapping) else {}
    latency_ms = float(timing.get("total", 0.0) or 0.0) * 1000.0

    scores = {
        "total": raw.get("total_score"),
        "pronunciation": raw.get("pronunciation_score"),
        "prosody": raw.get("prosody_score"),
        "fluency": raw.get("fluency_score"),
        "expression": raw.get("tone_score"),
    }
    reliability = details.get("reliability") or {}
    warnings = list(reliability.get("warnings") or [])
    content_note = _pick(details, "content_match", "note")
    if content_note:
        warnings.append(f"content_match: {content_note}")
    if inferred_mode in {"reference_free_acoustic", "acoustic"}:
        warnings.append("reference_free_acoustic outputs acoustic proxies, not kana correctness.")
    if inferred_mode == "asr_pseudo_reference":
        warnings.append("ASR transcript is a pseudo-reference, not ground truth.")

    input_info = {
        "audio_path": None if audio_path is None else str(audio_path),
        "target_text": target_text if target_text is not None else raw.get("target_text"),
        "kana": raw.get("kana"),
        "mora_count": len(raw.get("moras") or []),
        "asr_transcript": asr_transcript or _pick(details, "asr", "text") or _pick(details, "content_match", "transcript"),
        "cache_prefix": raw.get("cache_prefix"),
        "alignment_mode": raw.get("alignment_mode"),
        "reference_source": details.get("reference_source"),
    }
    if extra_input:
        input_info.update(extra_input)

    debug = {
        "endpointing": raw.get("endpointing") or details.get("endpointing") or {},
        "pause_info": raw.get("pause_info") or {},
        "prosody_metrics": raw.get("prosody_metrics") or details.get("prosody_metrics") or {},
        "content_match": details.get("content_match") or {},
        "alignment": details.get("alignment") or {},
        "mora_evidence_summary": details.get("mora_evidence_summary") or {},
        "mora_evidence": details.get("mora_evidence") or [],
        "timing": timing,
        "mora_table": raw.get("mora_table") or [],
    }

    return UnifiedEvaluationResult(
        mode=str(inferred_mode),
        input_info=input_info,
        features=_flatten_selected(raw),
        scores=scores,
        reliability=dict(reliability),
        warnings=warnings,
        feedback=list(raw.get("feedback") or []),
        debug=debug,
        latency_ms=round(float(latency_ms), 3),
        raw_metrics=raw,
    )
