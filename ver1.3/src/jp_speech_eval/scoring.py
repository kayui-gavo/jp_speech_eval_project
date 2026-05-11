from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .audio_features import basic_energy_stats, log_f0_normalize
from .config import DEFAULT_SCORING_CONFIG


def _cfg(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return config or DEFAULT_SCORING_CONFIG


def clamp_score(x: float) -> int:
    return int(max(0, min(100, round(float(x)))))


def score_fluency(
    mora_count: int,
    duration: float,
    pause_info: Dict,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[int, List[str], Dict]:
    c = _cfg(config)["fluency"]
    feedback: List[str] = []
    speech_duration = max(float(duration), 1e-6)
    speech_rate = mora_count / speech_duration

    target_min = float(c["target_mora_per_sec_min"])
    target_max = float(c["target_mora_per_sec_max"])
    slow = float(c["slow_mora_per_sec"])
    fast = float(c["fast_mora_per_sec"])
    very_fast = fast * 1.25

    if target_min <= speech_rate <= target_max:
        rate_score = 100.0
    elif slow <= speech_rate < target_min:
        rate_score = 75.0 + (speech_rate - slow) / max(target_min - slow, 1e-6) * 25.0
        feedback.append("语速稍慢，但仍然可以理解。")
    elif target_max < speech_rate <= fast:
        rate_score = 100.0 - (speech_rate - target_max) / max(fast - target_max, 1e-6) * 25.0
        feedback.append("语速稍快，可能影响清晰度。")
    elif fast < speech_rate <= very_fast:
        rate_score = 55.0 - (speech_rate - fast) / max(very_fast - fast, 1e-6) * 25.0
        feedback.append("语速明显偏快，当前轻量级对齐可能无法稳定切分每个 mora。")
    else:
        rate_score = 25.0
        feedback.append("语速偏离自然范围较多，结果更适合作为调试诊断。")

    pause_ratio = float(pause_info.get("pause_ratio", 0.0))
    pause_count = int(pause_info.get("pause_count", 0))
    pause_score = 100.0 - pause_ratio * float(c["pause_ratio_weight"]) - pause_count * float(c["pause_count_penalty"])
    if pause_count > 0:
        feedback.append(f"检测到 {pause_count} 次较长停顿，流畅度会下降。")

    score = 0.6 * rate_score + 0.4 * pause_score
    details = {
        "dimension_label": "delivery_style_not_pronunciation_correctness",
        "speech_duration_sec": speech_duration,
        "speech_rate_mora_per_sec": speech_rate,
        "avg_mora_duration_sec": speech_duration / max(mora_count, 1),
        "rate_score": clamp_score(rate_score),
        "pause_score": clamp_score(pause_score),
        "note": "engineering_thresholds_use_endpointed_speech_duration",
    }
    return clamp_score(score), feedback, details


def score_pronunciation_rhythm(
    moras: List[str],
    boundaries: List[Tuple[float, float]],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[int, List[str], Dict]:
    """
    Prototype pronunciation proxy.

    What it covers now:
    - mora rhythm stability
    - long vowel / nasal / sokuon duration proxy

    What it does NOT cover yet:
    - phoneme substitution such as す vs ず
    - real GOP or ASR posterior based pronunciation correctness
    """
    c = _cfg(config)["pronunciation"]
    feedback: List[str] = []
    durations = np.array([e - s for s, e in boundaries], dtype=float)
    if durations.size == 0:
        return 0, ["没有检测到有效 mora。"], {}

    avg = float(np.mean(durations))
    cv = float(np.std(durations) / (avg + 1e-8))
    rhythm_score = 100.0 - cv * float(c["rhythm_cv_weight"])

    special_penalty = 0.0
    special_short_ratio = float(c["special_mora_short_ratio"])
    for m, d in zip(moras, durations):
        if m in ["ー", "ン", "ッ"] and d < special_short_ratio * avg:
            special_penalty += float(c["special_mora_penalty"])
            feedback.append(f"「{m}」这一拍可能太短。")

    if cv > float(c["rhythm_cv_warning"]):
        feedback.append("mora 节奏不太稳定，可能有拖音或卡顿。")

    details = {
        "dimension_label": "core_pronunciation_related_proxy",
        "mora_duration_mean_sec": avg,
        "mora_duration_cv": cv,
        "special_mora_penalty": special_penalty,
        "score_interpretation": "mora_timing_proxy_not_full_segmental_pronunciation",
    }
    return clamp_score(rhythm_score - special_penalty), feedback, details


def _hl_from_norm_f0(z: float) -> str:
    if not np.isfinite(z):
        return "?"
    return "H" if z >= 0 else "L"


def _direction(a: float, b: float, th: float = 0.25) -> str:
    if not np.isfinite(a) or not np.isfinite(b):
        return "?"
    diff = b - a
    if diff > th:
        return "↑"
    if diff < -th:
        return "↓"
    return "→"


def _safe_float(value: float) -> Optional[float]:
    if not np.isfinite(value):
        return None
    return float(value)


def _direction_match(target: str, observed: str) -> bool:
    if target == "?" or observed == "?":
        return False
    if target == "→":
        return True
    return target == observed


def score_prosody(
    moras: List[str],
    target_pattern: List[str],
    f0_by_mora: List[float],
    reference_f0_by_mora: Optional[List[float]] = None,
    pitch_target_source: str = "heuristic",
    is_question: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[int, List[str], Dict]:
    c = _cfg(config)["prosody"]
    feedback: List[str] = []
    f0_arr = np.array(f0_by_mora, dtype=float)
    z = log_f0_normalize(f0_arr)
    observed_pattern = [_hl_from_norm_f0(v) for v in z]
    ref_arr = np.array(reference_f0_by_mora if reference_f0_by_mora is not None else [], dtype=float)
    if ref_arr.size:
        ref_z = log_f0_normalize(ref_arr)
    else:
        ref_z = np.array([
            1.0 if p == "H" else -1.0 if p == "L" else float("nan")
            for p in target_pattern
        ], dtype=float)
    primary_pitch_target_source = "tts_reference" if ref_arr.size else pitch_target_source
    hl_target_source = pitch_target_source
    if ref_z.size < z.size:
        ref_z = np.pad(ref_z, (0, z.size - ref_z.size), constant_values=np.nan)
    elif ref_z.size > z.size:
        ref_z = ref_z[: z.size]
    reference_pattern = [_hl_from_norm_f0(v) for v in ref_z]

    valid_idx = [i for i, obs in enumerate(observed_pattern) if obs != "?" and i < len(target_pattern)]
    contour_valid_idx = [
        i for i in range(min(len(z), len(ref_z)))
        if np.isfinite(z[i]) and np.isfinite(ref_z[i])
    ]
    if not valid_idx:
        return 50, ["F0 提取不稳定，语调评分可靠性较低。"], {
            "dimension_label": "core_pronunciation_related_prosody",
            "observed_pitch": observed_pattern,
            "reference_pitch": reference_pattern,
            "reference_normalized_log_f0": [_safe_float(v) for v in ref_z],
            "normalized_log_f0": [_safe_float(v) for v in z],
            "contour_corr": None,
            "contour_rmse": None,
            "transition_agreement": None,
            "final_intonation_match": None,
            "hl_match_rate": None,
            "pitch_target_source": primary_pitch_target_source,
            "hl_target_source": hl_target_source,
            "pitch_target_consistency": "unknown",
            "valid_mora_count": 0,
            "mora_count": len(moras),
            "note": "no_valid_f0",
            "target_pitch_note": "openjtalk_or_heuristic_labels_are_weak_auxiliary_targets; sentence_level_reference_contour_is_primary_when_available",
        }
    min_reliable = max(3, int(np.ceil(len(moras) * 0.5)))
    if len(valid_idx) < min_reliable or len(contour_valid_idx) < min_reliable:
        return 50, [f"只有 {len(valid_idx)}/{len(moras)} 个 mora 有可靠 F0，语调评分可靠性较低，可能是切分或 F0 提取不足。"], {
            "dimension_label": "core_pronunciation_related_prosody",
            "observed_pitch": observed_pattern,
            "reference_pitch": reference_pattern,
            "reference_normalized_log_f0": [_safe_float(v) for v in ref_z],
            "normalized_log_f0": [_safe_float(v) for v in z],
            "valid_mora_count": len(valid_idx),
            "mora_count": len(moras),
            "contour_corr": None,
            "contour_rmse": None,
            "transition_agreement": None,
            "final_intonation_match": None,
            "hl_match_rate": None,
            "pitch_target_source": primary_pitch_target_source,
            "hl_target_source": hl_target_source,
            "pitch_target_consistency": "unknown",
            "note": "insufficient_valid_mora_f0",
            "target_pitch_note": "openjtalk_or_heuristic_labels_are_weak_auxiliary_targets; sentence_level_reference_contour_is_primary_when_available",
        }

    direction_th = float(c["f0_direction_threshold"])
    heuristic_dir: List[str] = []
    reference_dir: List[str] = []
    observed_dir: List[str] = []
    for i in range(len(moras) - 1):
        if i + 1 >= len(target_pattern):
            break
        if target_pattern[i] == "L" and target_pattern[i + 1] == "H":
            td = "↑"
        elif target_pattern[i] == "H" and target_pattern[i + 1] == "L":
            td = "↓"
        else:
            td = "→"
        heuristic_dir.append(td)
        reference_dir.append(_direction(ref_z[i], ref_z[i + 1], th=direction_th))
        observed_dir.append(_direction(z[i], z[i + 1], th=direction_th))

    ref_dir_valid = [i for i, d in enumerate(observed_dir) if d != "?" and i < len(reference_dir) and reference_dir[i] != "?"]
    transition_agreement = float(np.mean([
        _direction_match(reference_dir[i], observed_dir[i]) for i in ref_dir_valid
    ])) if ref_dir_valid else 0.5

    heuristic_dir_valid = [i for i, d in enumerate(heuristic_dir) if i < len(reference_dir) and reference_dir[i] != "?"]
    heuristic_dir_match = float(np.mean([
        _direction_match(reference_dir[i], heuristic_dir[i]) for i in heuristic_dir_valid
    ])) if heuristic_dir_valid else 0.0
    heuristic_hl_valid = [i for i in range(min(len(reference_pattern), len(target_pattern))) if reference_pattern[i] != "?"]
    heuristic_hl_match = float(np.mean([
        reference_pattern[i] == target_pattern[i] for i in heuristic_hl_valid
    ])) if heuristic_hl_valid else 0.0
    if hl_target_source in {"dictionary", "manual"}:
        pitch_target_consistency = "trusted"
    elif hl_target_source.startswith("openjtalk"):
        if heuristic_hl_match >= 0.70 and heuristic_dir_match >= 0.55:
            pitch_target_consistency = "tool_generated_high"
        elif heuristic_hl_match >= 0.50 or heuristic_dir_match >= 0.40:
            pitch_target_consistency = "tool_generated_medium"
        else:
            pitch_target_consistency = "tool_generated_low"
    elif heuristic_hl_match >= 0.75 and heuristic_dir_match >= 0.60:
        pitch_target_consistency = "high"
    elif heuristic_hl_match >= 0.55 or heuristic_dir_match >= 0.45:
        pitch_target_consistency = "medium"
    else:
        pitch_target_consistency = "low"

    contour_x = z[contour_valid_idx]
    contour_y = ref_z[contour_valid_idx]
    if len(contour_valid_idx) >= 2 and float(np.std(contour_x)) > 1e-8 and float(np.std(contour_y)) > 1e-8:
        contour_corr = float(np.corrcoef(contour_x, contour_y)[0, 1])
    else:
        contour_corr = 0.0
    contour_rmse = float(np.sqrt(np.mean((contour_x - contour_y) ** 2))) if len(contour_valid_idx) else float("nan")
    corr_score = (contour_corr + 1.0) / 2.0
    rmse_score = np.exp(-0.5 * contour_rmse) if np.isfinite(contour_rmse) else 0.5
    contour_score = 0.70 * corr_score + 0.30 * float(rmse_score)

    hl_match = float(np.mean([observed_pattern[i] == target_pattern[i] for i in valid_idx]))
    if hl_target_source in {"dictionary", "manual"}:
        hl_weight = 0.20
    elif pitch_target_consistency in {"high", "medium", "tool_generated_high", "tool_generated_medium"}:
        hl_weight = 0.08
    else:
        hl_weight = 0.02

    final_intonation_match = None
    final_score = 0.75
    if len(z) >= 3 and len(ref_z) >= 3 and np.isfinite(z[-1]) and np.isfinite(z[-3]) and np.isfinite(ref_z[-1]) and np.isfinite(ref_z[-3]):
        final_user = _direction(z[-3], z[-1], th=direction_th)
        final_ref = _direction(ref_z[-3], ref_z[-1], th=direction_th)
        final_intonation_match = _direction_match(final_ref, final_user)
        final_score = 1.0 if final_intonation_match else 0.45
    elif len(z) >= 3 and np.isfinite(z[-1]) and np.isfinite(z[-3]):
        final_slope = float(z[-1] - z[-3])
        if is_question:
            final_intonation_match = final_slope > float(c["question_final_rise_threshold"])
        else:
            final_intonation_match = final_slope <= float(c["statement_final_rise_threshold"])
        final_score = 1.0 if final_intonation_match else 0.55

    transition_weight = 0.30
    final_weight = 0.12
    contour_weight = max(0.0, 1.0 - transition_weight - final_weight - hl_weight)
    score = 100.0 * (
        contour_weight * contour_score
        + transition_weight * transition_agreement
        + final_weight * final_score
        + hl_weight * hl_match
    )

    if contour_corr >= 0.70 and transition_agreement >= 0.60:
        feedback.append("整体音高轮廓接近参考音。")
    elif contour_corr >= 0.45:
        feedback.append("整体音高走向有接近参考的部分，但还有一些起伏差异。")
    else:
        feedback.append("整体音高轮廓和参考音差异较明显。")

    early_valid = [i for i in range(min(3, len(reference_dir), len(observed_dir))) if reference_dir[i] != "?" and observed_dir[i] != "?"]
    if early_valid:
        early_match = float(np.mean([_direction_match(reference_dir[i], observed_dir[i]) for i in early_valid]))
        if early_match >= 0.67:
            feedback.append("前半部分的音高起伏比较自然。")

    lift_indices = [i for i, d in enumerate(reference_dir) if d == "↑" and i < len(observed_dir)]
    weak_lifts = [
        i for i in lift_indices
        if observed_dir[i] != "↑" and i + 1 < len(moras)
    ]
    if weak_lifts:
        i = weak_lifts[0]
        feedback.append(f"参考音在「{moras[i]}〜{moras[i + 1]}」附近有更明显的上扬，你的上扬可以再清楚一点。")

    if final_intonation_match is True:
        feedback.append("句末语调接近参考音。")
    elif final_intonation_match is False:
        feedback.append("句末语调和参考音不太一致。")

    trusted_hl = hl_target_source in {"dictionary", "manual"} or pitch_target_consistency in {"high", "medium"}
    large_dev = []
    for i in valid_idx:
        if i >= len(ref_z) or not np.isfinite(ref_z[i]) or not np.isfinite(z[i]):
            continue
        if observed_pattern[i] != target_pattern[i] and abs(float(z[i] - ref_z[i])) >= 1.0:
            large_dev.append(i)
    if trusted_hl:
        for i in large_dev[:2]:
            feedback.append(
                f"第 {i + 1} 拍「{moras[i]}」的音高和目标差异较大，可以单独确认。"
            )
    elif pitch_target_consistency in {"low", "tool_generated_low"}:
        feedback.append("当前 H/L 标签来自自动前端/启发式规则，且和参考 F0 轮廓不完全一致，因此不按单拍 H/L 强判错。")

    details = {
        "dimension_label": "core_pronunciation_related_prosody",
        "observed_pitch": observed_pattern,
        "reference_pitch": reference_pattern,
        "target_direction": heuristic_dir,
        "reference_direction": reference_dir,
        "observed_direction": observed_dir,
        "normalized_log_f0": [_safe_float(v) for v in z],
        "reference_normalized_log_f0": [_safe_float(v) for v in ref_z],
        "valid_mora_count": len(valid_idx),
        "contour_valid_mora_count": len(contour_valid_idx),
        "mora_count": len(moras),
        "hl_match": hl_match,
        "direction_match": transition_agreement,
        "transition_agreement": transition_agreement,
        "final_score": clamp_score(final_score * 100.0),
        "contour_corr": _safe_float(contour_corr),
        "contour_rmse": _safe_float(contour_rmse),
        "final_intonation_match": final_intonation_match,
        "hl_match_rate": hl_match,
        "pitch_target_source": primary_pitch_target_source,
        "hl_target_source": hl_target_source,
        "pitch_target_consistency": pitch_target_consistency,
        "heuristic_reference_hl_match": heuristic_hl_match,
        "heuristic_reference_direction_match": heuristic_dir_match,
        "prosody_score_components": {
            "contour": _safe_float(contour_score),
            "transition": _safe_float(transition_agreement),
            "final": _safe_float(final_score),
            "hl": _safe_float(hl_match),
            "weights": {
                "contour": contour_weight,
                "transition": transition_weight,
                "final": final_weight,
                "hl": hl_weight,
            },
        },
        "theory_hint": "speaker_normalized_reference_contour_and_adjacent_mora_direction",
        "target_pitch_note": "openjtalk_or_heuristic_labels_are_weak_auxiliary_targets; sentence_level_reference_contour_is_primary_when_available",
    }
    return clamp_score(score), feedback[:5], details


def score_tone_simple(
    f0_by_mora: List[float],
    y: np.ndarray,
    pause_info: Dict,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[int, List[str], Dict]:
    """
    Lightweight tone/emotion proxy.

    This is not a true emotion recognizer. It estimates whether the voice sounds
    too flat, too unstable, too small, or hesitant from F0/energy/pause features.
    """
    c = _cfg(config)["tone"]
    feedback: List[str] = []
    f0 = np.array(f0_by_mora, dtype=float)
    valid = np.isfinite(f0) & (f0 > 0)

    pitch_score = 80.0
    pitch_range = None
    if valid.sum() >= 3:
        logf0 = np.log(f0[valid])
        pitch_range = float(np.max(logf0) - np.min(logf0))
        if pitch_range < float(c["low_pitch_range_log"]):
            pitch_score = 65.0
            feedback.append("音高变化较小，听起来可能偏平。")
        elif pitch_range > float(c["high_pitch_range_log"]):
            pitch_score = 70.0
            feedback.append("音高起伏较大，语气可能偏紧张或夸张。")
        else:
            pitch_score = 90.0

    energy = basic_energy_stats(y)
    energy_score = 85.0
    if energy["mean"] < float(c["low_energy_mean"]):
        energy_score = 65.0
        feedback.append("音量偏小，可能影响对方听清。")
    if energy["cv"] > float(c["high_energy_cv"]):
        energy_score -= 10.0
        feedback.append("能量变化较大，可能有紧张或不稳定感。")

    pause_ratio = float(pause_info.get("pause_ratio", 0.0))
    pause_score = 100.0 - pause_ratio * float(c["pause_ratio_weight"])
    if pause_ratio > float(c["high_pause_ratio"]):
        feedback.append("停顿比例较高，语气可能显得犹豫。")

    score = 0.45 * pitch_score + 0.35 * energy_score + 0.20 * pause_score
    details = {
        "dimension_label": "expression_style_not_pronunciation_correctness",
        "pitch_range_log": pitch_range,
        "pitch_score": clamp_score(pitch_score),
        "energy": energy,
        "energy_score": clamp_score(energy_score),
        "pause_score": clamp_score(pause_score),
        "score_interpretation": "expression_proxy_not_full_emotion_recognition",
    }
    return clamp_score(score), feedback, details
