from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .alignment import estimate_mora_boundaries, estimate_mora_boundaries_equal
from .audio_features import (
    detect_pauses,
    extract_f0,
    load_audio,
    median_f0_by_mora,
)
from .config import load_scoring_config
from .content_match import estimate_content_match
from .feedback_policy import FeedbackDecision, choose_feedback
from .mora_evidence import build_mora_evidence
from .recording_quality import assess_recording_quality
from .scoring import (
    score_fluency,
    score_pronunciation_rhythm,
    score_prosody,
    score_tone_simple,
)
from .sentence_cache import SentenceCache, load_sentence_cache
from .text_frontend import TextInfo, build_text_info
from .vad import trim_to_speech


@dataclass
class MoraRow:
    index: int
    mora: str
    start_sec: float
    end_sec: float
    f0_hz: Optional[float]
    target_pitch: str
    observed_pitch: str


@dataclass
class EvaluationResult:
    target_text: str
    kana: str
    moras: List[str]
    target_pitch: List[str]
    duration_sec: float
    f0_method: str
    alignment_mode: str
    pronunciation_score: int
    prosody_score: int
    fluency_score: int
    tone_score: int
    total_score: int
    feedback: List[str]
    pause_info: Dict
    endpointing: Dict
    details: Dict
    mora_table: List[MoraRow]
    prosody_metrics: Dict
    timing: Dict[str, float]
    cache_prefix: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


def _none_if_nan(x: float) -> Optional[float]:
    if x is None or not np.isfinite(x):
        return None
    return float(x)


def _text_info_from_cache(cache: SentenceCache) -> TextInfo:
    from .text_frontend import TextInfo
    return TextInfo(
        text=cache.meta.text,
        kana=cache.meta.kana,
        moras=cache.meta.moras,
        target_pitch=cache.meta.target_pitch,
        pitch_target_source=cache.meta.pitch_target_source,
        is_question=cache.meta.is_question,
        accent_phrases=cache.meta.accent_phrases,
    )


def _boundary_duration_cv(boundaries: List[tuple[float, float]]) -> float:
    durations = np.array([max(0.0, e - s) for s, e in boundaries], dtype=float)
    if durations.size == 0:
        return 0.0
    avg = float(np.mean(durations))
    return float(np.std(durations) / (avg + 1e-8))


def _boundary_health(boundaries: List[tuple[float, float]]) -> Dict[str, float | bool]:
    durations = np.array([max(0.0, e - s) for s, e in boundaries], dtype=float)
    if durations.size == 0:
        return {"cv": 0.0, "min_duration": 0.0, "max_duration": 0.0, "first_duration": 0.0, "is_unstable": True}
    avg = float(np.mean(durations))
    cv = float(np.std(durations) / (avg + 1e-8))
    min_duration = float(np.min(durations))
    max_duration = float(np.max(durations))
    first_duration = float(durations[0])
    is_unstable = bool(
        cv > 0.75
        or min_duration < 0.07
        or first_duration < 0.10
        or max_duration > max(0.55, 3.2 * avg)
    )
    return {
        "cv": cv,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "first_duration": first_duration,
        "is_unstable": is_unstable,
    }


def _score_to_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _build_learner_feedback(
    *,
    pronunciation_feedback: List[str],
    prosody_feedback: List[str],
    fluency_feedback: List[str],
    tone_feedback: List[str],
    reliability: Dict,
    mora_evidence_summary: Dict,
    content_match,
    mora_count: int,
) -> FeedbackDecision:
    """Create short user-facing feedback; keep diagnostics in details instead."""
    if content_match and content_match.status == "fail":
        return FeedbackDecision(
            feedback=["我没能确认你读的是目标句。请看着句子再读一次。"],
            policy="content_gate_fail",
        )

    feedback: List[str] = []
    if content_match and content_match.status == "uncertain":
        feedback.append("我不太确定这次是否读对了目标句，建议再试一次。")

    avg_mora_duration = float(reliability.get("avg_mora_duration_sec", 1.0) or 1.0)
    judgement_count = int(mora_evidence_summary.get("judgement_available_count", 0) or 0)
    judgement_needed = max(3, int(mora_count * 0.55))
    low_f0_coverage = float(reliability.get("f0_coverage", 0.0) or 0.0) < 0.50
    if avg_mora_duration < 0.09:
        feedback.append("你说得太快了。先放慢一点，把每个音说清楚。")
    elif judgement_count < judgement_needed or str(reliability.get("level")) == "low":
        feedback.append("这次录音里有些地方不够清楚，重录一次会更准。")
    elif bool(reliability.get("score_is_diagnostic")):
        feedback.append("这次结果只能作参考，建议再录一次确认。")
    if low_f0_coverage:
        feedback.append("这次录音里的音高信息不够清楚，重录一次会更准。")

    groups = [fluency_feedback]
    if judgement_count >= judgement_needed and avg_mora_duration >= 0.09:
        groups.append(pronunciation_feedback)
    if not low_f0_coverage:
        groups.append(prosody_feedback)
    groups.append(tone_feedback)
    for group in groups:
        feedback.extend(group)

    clean = _dedupe(feedback)
    return choose_feedback(
        raw_feedback=clean,
        reliability=reliability,
        mora_evidence_summary=mora_evidence_summary,
        max_items=4,
    )


def _build_reliability(
    endpointing: Dict,
    alignment_mode: str,
    boundary_health: Dict,
    prosody_details: Dict,
    fluency_details: Dict,
    recording_quality: Optional[Dict] = None,
    mora_evidence_summary: Optional[Dict] = None,
    reference_duration_sec: Optional[float] = None,
) -> Dict:
    warnings: List[str] = []

    endpoint_score = 1.0 if endpointing.get("detected") else 0.0
    if float(endpointing.get("speech_duration", 0.0)) < 0.4:
        endpoint_score *= 0.4
        warnings.append("Detected speech is very short.")

    alignment_score = 1.0
    if alignment_mode.endswith("fallback_equal"):
        alignment_score = 0.55
        warnings.append("Cached DTW mora alignment looked unstable; equal-time fallback was used.")
    elif float(boundary_health.get("cv", 0.0)) > 0.55:
        alignment_score = 0.65
        warnings.append("Mora boundary durations are uneven; alignment may be unreliable.")

    mora_count = int(prosody_details.get("mora_count", 0) or 0)
    valid_mora_count = int(prosody_details.get("valid_mora_count", 0) or 0)
    f0_coverage = valid_mora_count / max(mora_count, 1)
    f0_score = min(1.0, f0_coverage / 0.75)
    if f0_coverage < 0.5:
        warnings.append("Less than half of morae have reliable F0; pitch feedback is limited.")

    avg_mora_duration = float(fluency_details.get("avg_mora_duration_sec", 0.0) or 0.0)
    speech_rate = float(fluency_details.get("speech_rate_mora_per_sec", 0.0) or 0.0)
    duration_ratio = None
    if reference_duration_sec and reference_duration_sec > 0:
        duration_ratio = float(endpointing.get("speech_duration", 0.0)) / float(reference_duration_sec)

    if avg_mora_duration and avg_mora_duration < 0.09:
        alignment_score = min(alignment_score, 0.35)
        f0_score = min(f0_score, 0.55)
        warnings.append("Speech is too compressed for stable mora-level alignment/F0 analysis.")
    elif avg_mora_duration and avg_mora_duration < 0.11:
        alignment_score = min(alignment_score, 0.55)
        warnings.append("Speech is very fast; mora boundaries may be approximate.")

    if duration_ratio is not None and duration_ratio < 0.70:
        endpoint_score = min(endpoint_score, 0.65)
        alignment_score = min(alignment_score, 0.45)
        warnings.append("Detected speech is much shorter than the model reference; recording may be truncated or too fast.")

    recording_score = 1.0
    if recording_quality:
        recording_score = float(recording_quality.get("score", 1.0) or 1.0)
        warnings.extend(str(w) for w in recording_quality.get("warnings", []) or [])

    evidence_score = 1.0
    if mora_evidence_summary:
        mora_count_e = int(mora_evidence_summary.get("mora_count", 0) or 0)
        judgement_count = int(mora_evidence_summary.get("judgement_available_count", 0) or 0)
        evidence_ratio = judgement_count / max(mora_count_e, 1)
        boundary_conf = float(mora_evidence_summary.get("mean_boundary_confidence", 0.0) or 0.0)
        evidence_score = 0.55 * evidence_ratio + 0.45 * boundary_conf
        if evidence_ratio < 0.55:
            warnings.append("Less than 55% of morae have enough acoustic evidence for strong mora-level judgement.")
        if int(mora_evidence_summary.get("special_mora_count", 0) or 0) > 0:
            special_total = int(mora_evidence_summary.get("special_mora_count", 0) or 0)
            special_ok = int(mora_evidence_summary.get("special_mora_judgement_available_count", 0) or 0)
            if special_ok < special_total:
                warnings.append("Some special morae lack enough evidence for strong long/sokuon/nasal judgement.")
        alignment_score = min(alignment_score, max(0.35, evidence_score))

    overall = 0.25 * endpoint_score + 0.25 * alignment_score + 0.35 * f0_score + 0.15 * recording_score
    return {
        "overall": round(float(overall), 4),
        "level": _score_to_level(overall),
        "endpointing": round(float(endpoint_score), 4),
        "alignment": round(float(alignment_score), 4),
        "f0_coverage": round(float(f0_coverage), 4),
        "recording_quality": round(float(recording_score), 4),
        "mora_evidence": round(float(evidence_score), 4),
        "mora_judgement_available_count": None if not mora_evidence_summary else int(mora_evidence_summary.get("judgement_available_count", 0) or 0),
        "mora_prosody_available_count": None if not mora_evidence_summary else int(mora_evidence_summary.get("prosody_available_count", 0) or 0),
        "speech_rate_mora_per_sec": round(float(speech_rate), 4),
        "avg_mora_duration_sec": round(float(avg_mora_duration), 4),
        "duration_ratio_to_reference": None if duration_ratio is None else round(float(duration_ratio), 4),
        "valid_mora_count": valid_mora_count,
        "mora_count": mora_count,
        "score_is_diagnostic": overall < 0.75,
        "warnings": warnings,
    }


def evaluate_utterance(
    text: Optional[str] = None,
    wav_path: str | Path = "",
    alignment_mode: str = "dtw",
    sample_rate: int = 16000,
    cache_path: Optional[str | Path] = None,
    scoring_config_path: Optional[str | Path] = None,
    profile: bool = False,
    use_content_match: Optional[bool] = None,
) -> EvaluationResult:
    """
    Sentence-final detailed evaluation.

    Product-oriented usage:
      - prepare cache once with scripts/prepare_cache.py
      - call this with cache_path and alignment_mode='cached_dtw'

    Debug usage:
      - call this with text and alignment_mode='dtw' or 'equal'
    """
    t0 = time.perf_counter()
    timing: Dict[str, float] = {}

    cache: Optional[SentenceCache] = None
    if cache_path is not None:
        ts = time.perf_counter()
        cache = load_sentence_cache(cache_path)
        text_info = _text_info_from_cache(cache)
        sample_rate = cache.meta.sr
        timing["load_cache"] = time.perf_counter() - ts
        if alignment_mode == "dtw":
            alignment_mode = "cached_dtw"
    else:
        if text is None:
            raise ValueError("Either --text or --cache must be provided.")
        ts = time.perf_counter()
        text_info = build_text_info(text)
        timing["text_frontend"] = time.perf_counter() - ts

    config = load_scoring_config(scoring_config_path)

    ts = time.perf_counter()
    audio = load_audio(str(wav_path), sr=sample_rate)
    timing["load_audio"] = time.perf_counter() - ts

    ts = time.perf_counter()
    y_speech, speech_region = trim_to_speech(audio.y, audio.sr)
    active_duration = speech_region.speech_duration if speech_region.detected else len(y_speech) / audio.sr
    endpointing = speech_region.to_dict()
    endpointing = {
        k: round(float(v), 4) if isinstance(v, float) else v
        for k, v in endpointing.items()
    }
    recording_quality = assess_recording_quality(audio.y, audio.sr, speech_region)
    timing["endpointing_vad"] = time.perf_counter() - ts

    ts = time.perf_counter()
    times, f0, f0_method = extract_f0(y_speech, audio.sr)
    timing["extract_f0"] = time.perf_counter() - ts

    ts = time.perf_counter()
    content_cfg = config.get("content_match", {})
    should_use_content_match = bool(content_cfg.get("enabled", True)) if use_content_match is None else bool(use_content_match)
    content_match = estimate_content_match(
        cache,
        y_speech,
        audio.sr,
        use_asr=bool(content_cfg.get("use_asr", True)),
        asr_policy=str(content_cfg.get("asr_policy", "if_acoustic_uncertain")),
        asr_model=str(content_cfg.get("asr_model", "small")),
        asr_provider=str(content_cfg.get("asr_provider", "auto")),
    ) if cache and should_use_content_match else None
    timing["content_match"] = time.perf_counter() - ts

    ts = time.perf_counter()
    boundaries = estimate_mora_boundaries(
        text=text_info.text,
        y_trim=y_speech,
        sr=audio.sr,
        mora_count=len(text_info.moras),
        mode=alignment_mode,
        cache=cache,
    )
    boundary_health = _boundary_health(boundaries)
    if alignment_mode == "cached_dtw" and boundary_health["is_unstable"]:
        boundaries = estimate_mora_boundaries_equal(active_duration, len(text_info.moras))
        alignment_mode = "cached_dtw_fallback_equal"
        boundary_health = _boundary_health(boundaries)
    boundary_cv = float(boundary_health["cv"])
    timing["alignment"] = time.perf_counter() - ts

    ts = time.perf_counter()
    f0_mora = median_f0_by_mora(times, f0, boundaries)
    mora_evidence, mora_evidence_summary = build_mora_evidence(
        moras=text_info.moras,
        boundaries=boundaries,
        f0_times=times,
        f0_hz=f0,
        y_speech=y_speech,
        sr=audio.sr,
    )
    ref_f0_mora = (
        median_f0_by_mora(cache.ref_f0_times, cache.ref_f0, cache.meta.ref_mora_boundaries)
        if cache
        else None
    )
    pause_info = detect_pauses(
        y_speech,
        audio.sr,
        min_pause_sec=float(config["pause"]["long_pause_sec"]),
    )
    pronunciation_score, pron_fb, pron_details = score_pronunciation_rhythm(
        text_info.moras, boundaries, config=config
    )
    prosody_score, prosody_fb, prosody_details = score_prosody(
        moras=text_info.moras,
        target_pattern=text_info.target_pitch,
        f0_by_mora=f0_mora,
        reference_f0_by_mora=ref_f0_mora,
        pitch_target_source=text_info.pitch_target_source,
        is_question=text_info.is_question,
        accent_phrases=text_info.accent_phrases,
        config=config,
    )
    fluency_score, fluency_fb, fluency_details = score_fluency(
        mora_count=len(text_info.moras),
        duration=active_duration,
        pause_info=pause_info,
        config=config,
    )
    tone_score, tone_fb, tone_details = score_tone_simple(f0_mora, y_speech, pause_info, config=config)
    timing["scoring"] = time.perf_counter() - ts

    reliability = _build_reliability(
        endpointing=endpointing,
        alignment_mode=alignment_mode,
        boundary_health=boundary_health,
        prosody_details=prosody_details,
        fluency_details=fluency_details,
        recording_quality=recording_quality,
        mora_evidence_summary=mora_evidence_summary,
        reference_duration_sec=cache.meta.ref_duration_sec if cache else None,
    )

    score_adjustments: List[str] = []
    if alignment_mode.endswith("fallback_equal"):
        pronunciation_score = min(pronunciation_score, 65)
        score_adjustments.append(
            "mora 边界回退到等分切分，发音代理分已封顶。"
        )
    judgement_count = int(mora_evidence_summary.get("judgement_available_count", 0) or 0)
    judgement_needed = max(3, int(len(text_info.moras) * 0.55))
    if judgement_count < judgement_needed:
        pronunciation_score = min(pronunciation_score, 60)
        score_adjustments.append(
            "可判定的 mora 证据不足，发音代理分已封顶。"
        )
    if float(reliability.get("f0_coverage", 0.0) or 0.0) < 0.50:
        prosody_score = min(prosody_score, 55)
        score_adjustments.append(
            "F0 覆盖不足 50%，韵律分已封顶。"
        )

    aggregate_cfg = config.get("aggregate", {})
    aggregate_weights = {
        "pronunciation": float(aggregate_cfg.get("pronunciation_weight", 0.35)),
        "prosody": float(aggregate_cfg.get("prosody_weight", 0.40)),
        "fluency": float(aggregate_cfg.get("fluency_weight", 0.25)),
        "tone": float(aggregate_cfg.get("tone_weight", 0.0)),
    }
    aggregate_denominator = sum(max(0.0, value) for value in aggregate_weights.values())
    if aggregate_denominator <= 0:
        raise ValueError("aggregate score weights must sum to a positive value")
    total_score = round(
        (
            aggregate_weights["pronunciation"] * pronunciation_score
            + aggregate_weights["prosody"] * prosody_score
            + aggregate_weights["fluency"] * fluency_score
            + aggregate_weights["tone"] * tone_score
        )
        / aggregate_denominator
    )
    if float(reliability.get("overall", 0.0) or 0.0) < 0.75:
        total_score = min(total_score, 70)
        score_adjustments.append(
            "整体可靠性不足，本次总分作为诊断结果封顶。"
        )
    if content_match and content_match.status == "fail":
        pronunciation_score = 0
        prosody_score = 0
        fluency_score = 0
        tone_score = 0
        total_score = 0
        reliability["score_is_diagnostic"] = True
        reliability["level"] = "low"
        reliability["overall"] = min(float(reliability.get("overall", 0.0)), 0.25)
    elif content_match and content_match.status == "uncertain":
        total_score = min(int(total_score), 50)
        reliability["score_is_diagnostic"] = True
        reliability["level"] = "low"

    observed_pitch = prosody_details.get("observed_pitch", ["?"] * len(text_info.moras))
    mora_table: List[MoraRow] = []
    for i, mora in enumerate(text_info.moras):
        if i < len(boundaries):
            start, end = boundaries[i]
        else:
            start, end = 0.0, 0.0
        f0_val = f0_mora[i] if i < len(f0_mora) else float("nan")
        mora_table.append(
            MoraRow(
                index=i + 1,
                mora=mora,
                start_sec=round(float(start), 4),
                end_sec=round(float(end), 4),
                f0_hz=_none_if_nan(f0_val),
                target_pitch=text_info.target_pitch[i] if i < len(text_info.target_pitch) else "?",
                observed_pitch=observed_pitch[i] if i < len(observed_pitch) else "?",
            )
        )

    technical_feedback = pron_fb + prosody_fb + fluency_fb + tone_fb
    feedback_decision = _build_learner_feedback(
        pronunciation_feedback=pron_fb,
        prosody_feedback=prosody_fb,
        fluency_feedback=fluency_fb,
        tone_feedback=tone_fb,
        reliability=reliability,
        mora_evidence_summary=mora_evidence_summary,
        content_match=content_match,
        mora_count=len(text_info.moras),
    )
    feedback = feedback_decision.feedback

    prosody_metrics = {
        "contour_corr": prosody_details.get("contour_corr"),
        "contour_rmse": prosody_details.get("contour_rmse"),
        "transition_agreement": prosody_details.get("transition_agreement"),
        "final_intonation_match": prosody_details.get("final_intonation_match"),
        "hl_match_rate": prosody_details.get("hl_match_rate", prosody_details.get("hl_match")),
        "pitch_target_source": prosody_details.get("pitch_target_source", text_info.pitch_target_source),
        "hl_target_source": prosody_details.get("hl_target_source", text_info.pitch_target_source),
        "pitch_target_consistency": prosody_details.get("pitch_target_consistency", "unknown"),
    }

    timing["total"] = time.perf_counter() - t0
    timing = {k: round(float(v), 6) for k, v in timing.items()}

    result = EvaluationResult(
        target_text=text_info.text,
        kana=text_info.kana,
        moras=text_info.moras,
        target_pitch=text_info.target_pitch,
        duration_sec=round(float(active_duration), 4),
        f0_method=f0_method,
        alignment_mode=alignment_mode,
        pronunciation_score=pronunciation_score,
        prosody_score=prosody_score,
        fluency_score=fluency_score,
        tone_score=tone_score,
        total_score=int(total_score),
        feedback=feedback[:10],
        pause_info=pause_info,
        endpointing=endpointing,
        details={
            "endpointing": endpointing,
            "content_match": content_match.to_dict() if content_match else {
                "status": "unknown",
                "content_verified": False,
                "note": "no_sentence_cache_available_for_content_match",
            },
            "reliability": reliability,
            "technical_feedback": {
                "score_adjustments": score_adjustments,
                "raw_feedback": technical_feedback,
                "feedback_policy": feedback_decision.to_dict(),
            },
            "recording_quality": recording_quality,
            "mora_evidence": mora_evidence,
            "mora_evidence_summary": mora_evidence_summary,
            "alignment": {
                "boundary_duration_cv": boundary_cv,
                "min_mora_duration_sec": round(float(boundary_health["min_duration"]), 4),
                "first_mora_duration_sec": round(float(boundary_health["first_duration"]), 4),
                "max_mora_duration_sec": round(float(boundary_health["max_duration"]), 4),
                "mode": alignment_mode,
                "note": "fallback_equal_used_when_cached_dtw_boundary_cv_is_too_high"
                if alignment_mode == "cached_dtw_fallback_equal"
                else "cached_alignment_used",
            },
            "pronunciation": pron_details,
            "prosody": prosody_details,
            "prosody_metrics": prosody_metrics,
            "aggregate": {
                "weights": aggregate_weights,
                "score_interpretation": "pronunciation_oriented_total_excludes_expression_style_when_tone_weight_is_zero",
            },
            "score_adjustments": score_adjustments,
            "accent_phrases": text_info.accent_phrases,
            "reference_source": cache.meta.reference_source if cache else None,
            "reference_id": cache.meta.reference_id if cache else None,
            "reference_provider": cache.meta.reference_provider if cache else None,
            "reference_model": cache.meta.reference_model if cache else None,
            "reference_voice": cache.meta.reference_voice if cache else None,
            "reference_config_hash": cache.meta.reference_config_hash if cache else None,
            "fluency": fluency_details,
            "tone": tone_details,
        },
        mora_table=mora_table,
        prosody_metrics=prosody_metrics,
        timing=timing,
        cache_prefix=str(cache.prefix) if cache else None,
    )
    if profile:
        print_timing(result)
    return result


def plot_evaluation(
    result: EvaluationResult,
    wav_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
) -> None:
    audio = load_audio(str(wav_path), sr=sample_rate)
    y_speech, _region = trim_to_speech(audio.y, audio.sr)
    times, f0, _method = extract_f0(y_speech, audio.sr)

    f0_plot = np.array(f0, dtype=float)
    f0_plot[f0_plot <= 0] = np.nan
    if np.any(np.isfinite(f0_plot)):
        max_f0 = float(np.nanmax(f0_plot))
    else:
        max_f0 = 300.0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(times, f0_plot, linewidth=1.4, label="F0")

    for row in result.mora_table:
        s = row.start_sec
        e = row.end_sec
        mid = (s + e) / 2
        plt.axvline(s, linestyle="--", linewidth=0.5)
        label = f"{row.mora}\nT:{row.target_pitch}/O:{row.observed_pitch}"
        plt.text(mid, max_f0 * 0.96, label, ha="center", va="top", fontsize=9)
    if result.mora_table:
        plt.axvline(result.mora_table[-1].end_sec, linestyle="--", linewidth=0.5)

    plt.title(f"Mora-level F0 analysis: {result.target_text}")
    plt.xlabel("Time [s]")
    plt.ylabel("F0 [Hz]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def print_timing(result: EvaluationResult) -> None:
    print("\nTiming")
    for k, v in result.timing.items():
        print(f"  {k:16s}: {v * 1000:.2f} ms")


def print_result(result: EvaluationResult) -> None:
    print("\n========== Japanese Speech Evaluation ==========")
    print(f"Target text      : {result.target_text}")
    print(f"Kana             : {result.kana}")
    print(f"Mora             : {'・'.join(result.moras)}")
    print(f"Target pitch     : {' '.join(result.target_pitch)}")
    observed_pitch = [row.observed_pitch for row in result.mora_table]
    print(f"Observed pitch   : {' '.join(observed_pitch)}")
    print(f"Speech duration  : {result.duration_sec:.2f} sec")
    if result.endpointing:
        print(
            f"Raw duration     : {float(result.endpointing.get('raw_duration', 0.0)):.2f} sec "
            f"(lead {float(result.endpointing.get('leading_silence', 0.0)):.2f}, "
            f"trail {float(result.endpointing.get('trailing_silence', 0.0)):.2f})"
        )
    print(f"F0 method        : {result.f0_method}")
    print(f"Alignment        : {result.alignment_mode}")
    reliability = result.details.get("reliability", {})
    if reliability:
        print(
            f"Reliability      : {reliability.get('level', '?')} "
            f"({float(reliability.get('overall', 0.0)):.2f})"
        )
    if result.cache_prefix:
        print(f"Cache            : {result.cache_prefix}")
    print("\nScores")
    print(f"  Pronunciation  : {result.pronunciation_score}  # current proxy = mora timing / special mora duration")
    print(f"  Prosody        : {result.prosody_score}  # mora-level F0 / pitch direction")
    if result.prosody_metrics:
        print(
            "    contour      : "
            f"corr={result.prosody_metrics.get('contour_corr')} "
            f"rmse={result.prosody_metrics.get('contour_rmse')} "
            f"transition={result.prosody_metrics.get('transition_agreement')} "
            f"target={result.prosody_metrics.get('pitch_target_source')} "
            f"consistency={result.prosody_metrics.get('pitch_target_consistency')}"
        )
    print(f"  Fluency        : {result.fluency_score}  # delivery/style: speech rate / in-speech pauses")
    print(f"  Tone           : {result.tone_score}  # expression/style proxy, not pronunciation correctness")
    print(f"  Total          : {result.total_score}")
    print("\nFeedback")
    for fb in result.feedback:
        print(f"  - {fb}")
    print("\nMora table")
    print("idx\tmora\tstart\tend\tf0\ttarget\tobs")
    for row in result.mora_table:
        f0_str = "nan" if row.f0_hz is None else f"{row.f0_hz:.1f}"
        print(
            f"{row.index}\t{row.mora}\t{row.start_sec:.2f}\t{row.end_sec:.2f}\t"
            f"{f0_str}\t{row.target_pitch}\t{row.observed_pitch}"
        )
    if result.timing:
        print_timing(result)
