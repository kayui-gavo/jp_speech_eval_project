from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .asr import transcribe_language_aware
from .asr_confirmation import free_speech_language_eligibility
from .audio_features import basic_energy_stats, detect_pauses, extract_f0, load_audio
from .config import load_scoring_config
from .app_core.karaoke_timeline import build_relative_f0_visualization
from .free_speech_evidence import (
    build_free_speech_dimension_evidence,
    build_shadow_candidate_surface,
)
from .recording_quality import assess_recording_quality
from .scoring import clamp_score
from .spontaneous_fluency import build_spontaneous_fluency_evidence
from .structure_features import (
    f0_structure_features,
    light_pronunciation_risk_features,
    mora_structure_features,
)
from .text_frontend import build_text_info
from .transcript_sanity import check_asr_transcript_sanity
from .vad import trim_to_speech


def _none_if_nan(value: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _score_from_range(value: float, good_min: float, good_max: float, bad_min: float, bad_max: float) -> float:
    if good_min <= value <= good_max:
        return 100.0
    if bad_min <= value < good_min:
        return 55.0 + (value - bad_min) / max(good_min - bad_min, 1e-6) * 45.0
    if good_max < value <= bad_max:
        return 100.0 - (value - good_max) / max(bad_max - good_max, 1e-6) * 45.0
    return 45.0


def _language_reject_result(
    *,
    transcript: str,
    asr_info: Dict[str, Any],
    language_reason: str,
    transcript_sanity: Dict[str, Any],
    speech_duration: float,
    endpointing: Dict[str, Any],
    recording_quality: Dict[str, Any],
    timing: Dict[str, float],
) -> Dict[str, Any]:
    """Return an explicit no-score raw result for non-Japanese/unsafe ASR input.

    Raw score fields are ``None`` rather than zero: language eligibility is a
    routing decision, not evidence of poor Japanese ability.  The product score
    policy independently sees ``transcript_sanity.ok = false`` and therefore
    also returns a no-score/retry response.
    """
    quality_factor = float(recording_quality.get("reliability_factor", 1.0) or 1.0)
    reliability_overall = min(0.25, max(0.0, quality_factor * 0.25))
    timing["total"] = timing.get("total", 0.0)
    return {
        "target_text": transcript or "",
        "kana": "",
        "moras": [],
        "target_pitch": [],
        "duration_sec": round(float(speech_duration), 4),
        "f0_method": "not_run_language_reject",
        "alignment_mode": "none",
        "pronunciation_score": None,
        "prosody_score": None,
        "fluency_score": None,
        "tone_score": None,
        "total_score": None,
        "feedback": ["今回は日本語として安定して確認できませんでした。日本語でもう一度話してください。"],
        "pause_info": {},
        "endpointing": endpointing,
        "details": {
            "mode": "transcript_assisted_light",
            "interpretation": "language_eligibility_reject_no_score",
            "score_eligible": False,
            "asr": asr_info,
            "language_gate": {
                "eligible": False,
                "reason": language_reason,
                "policy": "unforced_asr_then_japanese_eligibility_v1",
            },
            "transcript_sanity": transcript_sanity,
            "endpointing": endpointing,
            "recording_quality": recording_quality,
            "reliability": {
                "overall": round(reliability_overall, 4),
                "level": "low",
                "endpointing": 1.0 if endpointing.get("detected") else 0.0,
                "alignment": 0.0,
                "f0_coverage": 0.0,
                "recording_quality": round(float(recording_quality.get("score", 1.0) or 1.0), 4),
                "valid_mora_count": 0,
                "mora_count": 0,
                "score_is_diagnostic": False,
                "warnings": [f"free_speech_language_reject:{language_reason}"],
            },
        },
        "mora_table": [],
        "prosody_metrics": {
            "contour_corr": None,
            "contour_rmse": None,
            "transition_agreement": None,
            "final_intonation_match": None,
            "hl_match_rate": None,
            "pitch_target_source": "unavailable_language_reject",
            "hl_target_source": "unavailable_language_reject",
            "pitch_target_consistency": "not_checked",
        },
        "timing": {k: round(float(v), 6) for k, v in timing.items()},
        "cache_prefix": None,
    }


def evaluate_transcript_assisted_light(
    wav_path: str | Path,
    transcript: Optional[str] = None,
    sample_rate: int = 16000,
    scoring_config_path: str | Path | None = None,
    asr_model: str = "small",
    asr_provider: str = "auto",
) -> Dict:
    """Light free-speaking diagnosis using a transcript, but no TTS reference/DTW.

    If no transcript is supplied, ASR is deliberately *language-aware* and
    unforced.  A confident non-Japanese result is rejected before mora/F0
    scoring.  This prevents English/Chinese/etc. speech from being coerced into
    plausible-looking Japanese by a forced ``language='ja'`` decoder.

    An externally supplied transcript is still checked for Japanese-script
    sanity before it can drive mora-rate scoring.  ``spontaneous_fluency_v2``
    remains shadow evidence only and does not change the current C-end mapping.
    """
    t0 = time.perf_counter()
    timing: Dict[str, float] = {}
    audio = load_audio(str(wav_path), sr=sample_rate)
    y_speech, region = trim_to_speech(audio.y, audio.sr)
    quality = assess_recording_quality(audio.y, audio.sr, region)
    endpointing = {
        k: round(float(v), 4) if isinstance(v, float) else v
        for k, v in region.to_dict().items()
    }
    speech_duration = float(region.speech_duration if region.detected else len(y_speech) / audio.sr)

    external_transcript = bool(transcript and transcript.strip())
    asr_info: Dict[str, Any] = {
        "available": external_transcript,
        "text": transcript or "",
        "provider": "external",
        "model": "",
        "language": "",
        "language_probability": None,
        "note": "external_transcript",
    }
    language_eligible = True
    language_reason = "external_transcript_japanese_script_gate"
    if not external_transcript:
        ts = time.perf_counter()
        config = load_scoring_config(scoring_config_path)
        content_cfg = config.get("content_match", {})
        asr = transcribe_language_aware(
            y_speech,
            audio.sr,
            model_name=str(content_cfg.get("asr_model", asr_model)),
            provider=str(content_cfg.get("asr_provider", asr_provider)),
        )
        timing["asr"] = time.perf_counter() - ts
        asr_info = asr.to_dict()
        language_eligible, language_reason = free_speech_language_eligibility(asr)
        transcript = asr.text if asr.available else ""

    sanity = check_asr_transcript_sanity(transcript or "")
    sanity_payload = sanity.to_dict()
    if external_transcript:
        language_eligible = bool(sanity.ok)
        if not language_eligible:
            language_reason = "external_transcript_not_safely_japanese"
    safe_to_score = bool(language_eligible and sanity.ok)
    if not safe_to_score:
        # Ensure the existing product no-score gate sees the language decision,
        # even when a non-Japanese ASR happened to hallucinate Japanese-looking
        # text that passed the lightweight script sanity check.
        original_reason = sanity_payload.get("reason")
        sanity_payload["ok"] = False
        sanity_payload["reason"] = (
            f"language_gate:{language_reason}"
            if not language_eligible
            else str(original_reason or "transcript_sanity_failed")
        )
        sanity_payload["language_gate_reason"] = language_reason
        sanity_payload["original_sanity_reason"] = original_reason
        timing["total"] = time.perf_counter() - t0
        return _language_reject_result(
            transcript=transcript or "",
            asr_info=asr_info,
            language_reason=language_reason,
            transcript_sanity=sanity_payload,
            speech_duration=speech_duration,
            endpointing=endpointing,
            recording_quality=quality,
            timing=timing,
        )

    text_info = None
    moras: List[str] = []
    kana = ""
    if transcript:
        try:
            text_info = build_text_info(transcript)
            moras = text_info.moras
            kana = text_info.kana
        except Exception as exc:
            asr_info["note"] = f"{asr_info.get('note', '')}; text_frontend_failed: {type(exc).__name__}: {exc}"

    ts = time.perf_counter()
    _times, f0, f0_method = extract_f0(y_speech, audio.sr)
    f0_arr = np.asarray(f0, dtype=float)
    voiced = np.isfinite(f0_arr) & (f0_arr > 0)
    voiced_ratio = float(np.mean(voiced)) if f0_arr.size else 0.0
    f0_valid = f0_arr[voiced]
    if f0_valid.size >= 3:
        log_f0 = np.log(f0_valid)
        f0_mean = float(np.mean(f0_valid))
        f0_std = float(np.std(f0_valid))
        f0_range_log = float(np.max(log_f0) - np.min(log_f0))
    else:
        f0_mean = float("nan")
        f0_std = float("nan")
        f0_range_log = float("nan")
    timing["extract_f0"] = time.perf_counter() - ts

    pause_info = detect_pauses(y_speech, audio.sr)
    energy = basic_energy_stats(y_speech)
    mora_count = len(moras)
    mora_rate = mora_count / max(speech_duration, 1e-6) if mora_count else None
    pause_ratio = float(pause_info.get("pause_ratio", 0.0))
    mora_struct = mora_structure_features(moras, speech_duration)
    f0_struct = f0_structure_features(f0_arr)
    risk_struct = light_pronunciation_risk_features(moras, speech_duration, voiced_ratio, pause_ratio)
    structure_features = {
        **mora_struct,
        **f0_struct,
        **risk_struct,
        "interpretation": "speaker_normalized_structural_proxy",
    }
    transcript_source = str(asr_info.get("provider") or "unknown")
    if asr_info.get("model"):
        transcript_source = f"{transcript_source}:{asr_info.get('model')}"
    spontaneous_fluency_v2 = build_spontaneous_fluency_evidence(
        mora_count=mora_count,
        speech_duration_sec=speech_duration,
        pause_info=pause_info,
        transcript=transcript or "",
        transcript_source=transcript_source,
        silent_pause_threshold_sec=0.30,
        word_timestamps=(
            asr_info.get("words")
            if isinstance(asr_info.get("words"), list)
            else None
        ),
    )
    free_speech_dimension_evidence = build_free_speech_dimension_evidence(
        asr_info=asr_info,
        speech_duration_sec=speech_duration,
        f0_times=_times,
        f0_hz=f0_arr,
        spontaneous_fluency=spontaneous_fluency_v2,
    )
    relative_f0_visualization = build_relative_f0_visualization(_times, f0_arr)

    feedback: List[str] = [
        "当前为 Transcript-assisted light 模式： transcript 只用于估计 mora 数，不生成 TTS reference、不做 DTW，因此不输出具体假名纠错。"
    ]
    warnings: List[str] = []
    reliability_score = 1.0
    if not region.detected:
        reliability_score *= 0.2
        warnings.append("No stable speech region detected.")
    if not transcript:
        reliability_score *= 0.35
        warnings.append("No transcript available; only acoustic proxies are reliable.")
        feedback.append("没有可用 transcript，无法估计 mora rate。")
    if speech_duration < 0.35:
        reliability_score *= 0.5
        warnings.append("Speech is too short for stable diagnosis.")
    if voiced_ratio < 0.25:
        reliability_score *= 0.7
        warnings.append("Low voiced-frame coverage limits F0/prosody diagnosis.")
    reliability_score *= float(quality.get("reliability_factor", 1.0) or 1.0)
    warnings.extend(str(w) for w in quality.get("warnings", []) or [])

    fluency_score = 65.0
    if mora_rate is not None:
        if 4.0 <= mora_rate <= 7.0:
            fluency_score = 92.0
        elif 3.0 <= mora_rate < 4.0 or 7.0 < mora_rate <= 8.5:
            fluency_score = 76.0
            feedback.append("语速略偏离自然范围，但这里仅按 transcript 的 mora 数粗估。")
        else:
            fluency_score = 55.0
            feedback.append("语速可能偏慢或偏快；该判断依赖 ASR/transcript 的 mora 数。")
    fluency_score -= min(35.0, pause_ratio * 120.0)
    if risk_struct["too_fast_for_special_mora"]:
        feedback.append("特殊拍比例较高且语速偏快，长音、促音、拨音可能被压缩；这只是风险提示，不是具体假名判错。")
    elif risk_struct["compressed_mora_risk"]:
        feedback.append("平均 mora 时长偏短，快速发话时切分和特殊拍听感会更不稳定。")

    prosody_score = 65.0
    if np.isfinite(f0_range_log):
        prosody_score = _score_from_range(f0_range_log, 0.18, 0.80, 0.04, 1.25)
        if f0_range_log < 0.18:
            feedback.append("音高变化较小，表达可能偏平。")
        elif f0_range_log > 0.80:
            feedback.append("音高起伏较大，表达可能偏紧张或夸张。")
    else:
        prosody_score = 50.0
        feedback.append("F0 提取不足，音高起伏判断不稳定。")

    recording_score = 100.0
    if energy["mean"] < 0.012:
        recording_score -= 25.0
        warnings.append("Low input energy reduces confidence.")
    recording_score = min(recording_score, 100.0 * float(quality.get("score", 1.0) or 1.0))
    for warning in quality.get("warnings", []) or []:
        feedback.append(f"录音条件提示：{warning}")
    clarity_score = 100.0 - (25.0 if voiced_ratio < 0.25 else 0.0) - (15.0 if energy["cv"] > 1.2 else 0.0)
    pronunciation_risk = 0.35 * clarity_score + 0.25 * prosody_score + 0.25 * fluency_score + 0.15 * recording_score
    total = 0.30 * pronunciation_risk + 0.25 * prosody_score + 0.30 * fluency_score + 0.15 * recording_score
    free_speech_candidate_surface = build_shadow_candidate_surface(
        free_speech_dimension_evidence,
        current_fluency_score=fluency_score,
    )

    reliability = {
        "overall": round(float(reliability_score), 4),
        "level": "high" if reliability_score >= 0.75 else "medium" if reliability_score >= 0.45 else "low",
        "endpointing": 1.0 if region.detected else 0.0,
        "alignment": 0.0,
        "f0_coverage": round(float(voiced_ratio), 4),
        "recording_quality": round(float(quality.get("score", 1.0) or 1.0), 4),
        "valid_mora_count": 0,
        "mora_count": mora_count,
        "score_is_diagnostic": True,
        "warnings": warnings,
    }

    timing["total"] = time.perf_counter() - t0
    return {
        "target_text": transcript or "Transcript-assisted light diagnosis",
        "kana": kana,
        "moras": moras,
        "target_pitch": text_info.target_pitch if text_info else [],
        "duration_sec": round(float(speech_duration), 4),
        "f0_method": f0_method,
        "alignment_mode": "none",
        "pronunciation_score": clamp_score(pronunciation_risk),
        "prosody_score": clamp_score(prosody_score),
        "fluency_score": clamp_score(fluency_score),
        "tone_score": clamp_score(recording_score),
        "total_score": clamp_score(total),
        "feedback": feedback[:10],
        "pause_info": pause_info,
        "endpointing": endpointing,
        "details": {
            "mode": "transcript_assisted_light",
            "interpretation": "transcript_assisted_acoustic_proxy_not_kana_correctness",
            "score_eligible": True,
            "visualization_source": {
                "schema_version": "consumer_visualization_source_v1",
                "timebase": "speech_trim_relative",
                "score_role": "visualization_only",
                "product_score_changed": False,
                "relative_f0": relative_f0_visualization,
            },
            "asr": asr_info,
            "language_gate": {
                "eligible": True,
                "reason": language_reason,
                "policy": "unforced_asr_then_japanese_eligibility_v1",
            },
            "transcript_sanity": sanity_payload,
            "endpointing": endpointing,
            "acoustic_features": {
                "speech_duration_sec": speech_duration,
                "voiced_ratio": voiced_ratio,
                "pause_ratio": pause_ratio,
                "pause_count": int(pause_info.get("pause_count", 0)),
                "f0_mean_hz": _none_if_nan(f0_mean),
                "f0_std_hz": _none_if_nan(f0_std),
                "relative_log_f0_range": _none_if_nan(f0_range_log),
                "f0_method": f0_method,
                "mora_count_from_transcript": mora_count,
                "mora_rate_from_transcript": mora_rate,
            },
            "structure_features": structure_features,
            "fluency": {
                "speech_duration_sec": speech_duration,
                "speech_rate_mora_per_sec": mora_rate,
                "avg_mora_duration_sec": None if mora_rate is None else speech_duration / max(mora_count, 1),
                "spontaneous_v2_shadow": spontaneous_fluency_v2,
                "note": "transcript_assisted_proxy_no_dtw; spontaneous_v2_is_shadow_only",
            },
            "spontaneous_fluency_v2": spontaneous_fluency_v2,
            "shadow": {
                "free_speech_dimension_evidence": free_speech_dimension_evidence,
                "free_speech_candidate_surface": free_speech_candidate_surface,
            },
            "recording_quality": {
                **quality,
                "energy_mean": energy["mean"],
                "energy_cv": energy["cv"],
            },
            "reliability": reliability,
        },
        "mora_table": [],
        "prosody_metrics": {
            "contour_corr": None,
            "contour_rmse": None,
            "transition_agreement": None,
            "final_intonation_match": None,
            "hl_match_rate": None,
            "pitch_target_source": "transcript_heuristic",
            "hl_target_source": "heuristic",
            "pitch_target_consistency": "not_checked",
        },
        "timing": {k: round(float(v), 6) for k, v in timing.items()},
        "cache_prefix": None,
    }