from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .audio_features import basic_energy_stats, detect_pauses, extract_f0, load_audio
from .recording_quality import assess_recording_quality
from .structure_features import f0_structure_features
from .vad import trim_to_speech


def _clamp_score(value: float) -> int:
    return int(max(0, min(100, round(float(value)))))


def _none_if_nan(value: float) -> float | None:
    if not np.isfinite(value):
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


def evaluate_reference_free_acoustic(
    wav_path: str | Path,
    sample_rate: int = 16000,
) -> Dict:
    """Wav-only acoustic diagnosis.

    This deliberately avoids text, ASR, pyopenjtalk, TTS references, and DTW.
    Outputs are acoustic proxies and risk flags, not pronunciation correctness.
    """
    t0 = time.perf_counter()
    audio = load_audio(str(wav_path), sr=sample_rate)
    y_speech, region = trim_to_speech(audio.y, audio.sr)
    quality = assess_recording_quality(audio.y, audio.sr, region)
    endpointing = {
        k: round(float(v), 4) if isinstance(v, float) else v
        for k, v in region.to_dict().items()
    }
    speech_duration = float(region.speech_duration if region.detected else len(y_speech) / audio.sr)

    times, f0, f0_method = extract_f0(y_speech, audio.sr)
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

    pause_info = detect_pauses(y_speech, audio.sr)
    energy = basic_energy_stats(y_speech)
    f0_struct = f0_structure_features(f0_arr)
    peak = float(np.max(np.abs(y_speech))) if y_speech.size else 0.0
    clipping_ratio = float(np.mean(np.abs(y_speech) > 0.98)) if y_speech.size else 0.0

    recording_score = 100.0
    feedback: List[str] = []
    if energy["mean"] < 0.012:
        recording_score -= 25.0
        feedback.append("录音音量偏小，后续发音判断只能作为风险提示。")
    if clipping_ratio > 0.01:
        recording_score -= 25.0
        feedback.append("录音可能有削波或爆音，建议降低输入音量。")
    if speech_duration < 0.35:
        recording_score -= 30.0
        feedback.append("有效语音太短，无法稳定判断口语表现。")
    recording_score = min(recording_score, 100.0 * float(quality.get("score", 1.0) or 1.0))
    for warning in quality.get("warnings", []) or []:
        feedback.append(f"录音条件提示：{warning}")

    pause_ratio = float(pause_info.get("pause_ratio", 0.0))
    fluency_score = 100.0 - min(55.0, pause_ratio * 180.0 + int(pause_info.get("pause_count", 0)) * 8.0)
    if pause_ratio > 0.25:
        feedback.append("停顿比例较高，可能存在犹豫或断续。")

    prosody_score = 75.0
    if np.isfinite(f0_range_log):
        prosody_score = _score_from_range(f0_range_log, 0.18, 0.80, 0.04, 1.25)
        if f0_range_log < 0.18:
            feedback.append("音高变化较小，表达可能偏平。")
        elif f0_range_log > 0.80:
            feedback.append("音高起伏较大，表达可能偏紧张或夸张。")
    else:
        prosody_score = 50.0
        feedback.append("F0 提取不足，无法稳定判断音高起伏。")

    clarity_score = 100.0
    if voiced_ratio < 0.25:
        clarity_score -= 25.0
        feedback.append("有声帧比例偏低，可能是录音弱、清辅音多，或声音不够清晰。")
    if energy["cv"] > 1.20:
        clarity_score -= 15.0
        feedback.append("音量稳定性较低，可能影响听感清楚度。")

    pronunciation_risk_score = 0.35 * clarity_score + 0.25 * prosody_score + 0.20 * fluency_score + 0.20 * recording_score
    total_score = 0.35 * pronunciation_risk_score + 0.30 * fluency_score + 0.20 * prosody_score + 0.15 * recording_score

    reliability_score = 1.0
    warnings: List[str] = []
    if not region.detected:
        reliability_score *= 0.2
        warnings.append("No stable speech region detected.")
    if speech_duration < 0.35:
        reliability_score *= 0.5
        warnings.append("Speech is too short for stable acoustic diagnosis.")
    if voiced_ratio < 0.25:
        reliability_score *= 0.7
        warnings.append("Low voiced-frame coverage limits F0/prosody diagnosis.")
    if energy["mean"] < 0.012:
        reliability_score *= 0.75
        warnings.append("Low input energy reduces confidence.")
    reliability_score *= float(quality.get("reliability_factor", 1.0) or 1.0)
    warnings.extend(str(w) for w in quality.get("warnings", []) or [])

    if not feedback:
        feedback.append("声学表现整体稳定；注意这只是无文本 acoustic proxy，不能证明具体发音完全正确。")
    feedback.insert(0, "当前为 Acoustic-only 实验模式：不使用目标文本、不使用 ASR，因此只能输出发音风险和表达诊断，不能判断具体假名是否读对。")

    details = {
        "mode": "reference_free_acoustic",
        "interpretation": "wav_only_acoustic_proxy_not_pronunciation_correctness",
        "endpointing": endpointing,
        "recording_quality": {
            **quality,
            "peak": peak,
            "clipping_ratio": clipping_ratio,
            "energy_mean": energy["mean"],
            "energy_cv": energy["cv"],
        },
        "acoustic_features": {
            "speech_duration_sec": speech_duration,
            "voiced_ratio": voiced_ratio,
            "pause_ratio": pause_ratio,
            "pause_count": int(pause_info.get("pause_count", 0)),
            "f0_mean_hz": _none_if_nan(f0_mean),
            "f0_std_hz": _none_if_nan(f0_std),
            "relative_log_f0_range": _none_if_nan(f0_range_log),
            "f0_method": f0_method,
        },
        "structure_features": {
            **f0_struct,
            "interpretation": "wav_only_speaker_normalized_f0_structure",
        },
        "reliability": {
            "overall": round(float(reliability_score), 4),
            "level": "high" if reliability_score >= 0.75 else "medium" if reliability_score >= 0.45 else "low",
            "endpointing": 1.0 if region.detected else 0.0,
            "alignment": 0.0,
            "f0_coverage": round(float(voiced_ratio), 4),
            "recording_quality": round(float(quality.get("score", 1.0) or 1.0), 4),
            "valid_mora_count": 0,
            "mora_count": 0,
            "score_is_diagnostic": True,
            "warnings": warnings,
        },
    }

    return {
        "target_text": "Reference-free acoustic diagnosis",
        "kana": "",
        "moras": [],
        "target_pitch": [],
        "duration_sec": round(float(speech_duration), 4),
        "f0_method": f0_method,
        "alignment_mode": "none",
        "pronunciation_score": _clamp_score(pronunciation_risk_score),
        "prosody_score": _clamp_score(prosody_score),
        "fluency_score": _clamp_score(fluency_score),
        "tone_score": _clamp_score(recording_score),
        "total_score": _clamp_score(total_score),
        "feedback": feedback[:10],
        "pause_info": pause_info,
        "endpointing": endpointing,
        "details": details,
        "mora_table": [],
        "prosody_metrics": {
            "contour_corr": None,
            "contour_rmse": None,
            "transition_agreement": None,
            "final_intonation_match": None,
            "hl_match_rate": None,
            "pitch_target_source": "none",
            "hl_target_source": "none",
            "pitch_target_consistency": "not_applicable",
        },
        "timing": {"total": round(float(time.perf_counter() - t0), 6)},
        "cache_prefix": None,
    }
