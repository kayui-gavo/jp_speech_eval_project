from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .asr import transcribe_japanese
from .audio_features import basic_energy_stats, detect_pauses, extract_f0, load_audio
from .config import load_scoring_config
from .recording_quality import assess_recording_quality
from .scoring import clamp_score
from .structure_features import (
    f0_structure_features,
    light_pronunciation_risk_features,
    mora_structure_features,
)
from .text_frontend import build_text_info
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


def evaluate_transcript_assisted_light(
    wav_path: str | Path,
    transcript: Optional[str] = None,
    sample_rate: int = 16000,
    scoring_config_path: str | Path | None = None,
    asr_model: str = "small",
    asr_provider: str = "auto",
) -> Dict:
    """Light free-speaking diagnosis using a transcript, but no TTS reference/DTW.

    This mode is intentionally conservative. The transcript can be provided by
    ASR or externally. It is used only to estimate kana/mora count and rough
    speaking-rate context.
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

    asr_info = {"available": False, "text": transcript or "", "provider": "external", "model": "", "note": "external_transcript"}
    if not transcript:
        ts = time.perf_counter()
        config = load_scoring_config(scoring_config_path)
        content_cfg = config.get("content_match", {})
        asr = transcribe_japanese(
            y_speech,
            audio.sr,
            model_name=str(content_cfg.get("asr_model", asr_model)),
            provider=str(content_cfg.get("asr_provider", asr_provider)),
        )
        timing["asr"] = time.perf_counter() - ts
        transcript = asr.text if asr.available else ""
        asr_info = asr.to_dict()

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
            "asr": asr_info,
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
                "note": "transcript_assisted_proxy_no_dtw",
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
