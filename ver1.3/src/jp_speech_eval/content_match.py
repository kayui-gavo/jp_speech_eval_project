from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import librosa
import numpy as np

from .alignment import _user_mfcc
from .asr import transcribe_japanese
from .sentence_cache import SentenceCache
from .text_frontend import kata_normalize, text_to_kana


@dataclass(frozen=True)
class ContentMatch:
    status: str
    score: float
    dtw_cost: float
    duration_ratio: float
    kana_similarity: float
    transcript: str
    transcript_kana: str
    target_kana: str
    asr_provider: str
    content_verified: bool
    method: str
    note: str

    def to_dict(self) -> Dict:
        return asdict(self)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = cur
    return prev[-1]


def _kana_similarity(a: str, b: str) -> float:
    a = kata_normalize(a)
    b = kata_normalize(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return _clip01(1.0 - _edit_distance(a, b) / max(len(a), len(b), 1))


def _empty(
    *,
    status: str,
    score: float,
    dtw_cost: float,
    duration_ratio: float,
    target_kana: str,
    method: str,
    note: str,
) -> ContentMatch:
    return ContentMatch(
        status=status,
        score=score,
        dtw_cost=dtw_cost,
        duration_ratio=duration_ratio,
        kana_similarity=0.0,
        transcript="",
        transcript_kana="",
        target_kana=target_kana,
        asr_provider="none",
        content_verified=False,
        method=method,
        note=note,
    )


def _asr_gate(
    cache: SentenceCache,
    y_speech: np.ndarray,
    sr: int,
    acoustic_status: str,
    acoustic_score: float,
    dtw_cost: float,
    duration_ratio: float,
    model_name: str,
    provider: str,
) -> ContentMatch:
    target_kana = cache.meta.kana
    transcript = transcribe_japanese(y_speech, sr, model_name=model_name, provider=provider)
    if not transcript.available:
        return ContentMatch(
            status=acoustic_status,
            score=round(float(acoustic_score), 4),
            dtw_cost=round(float(dtw_cost), 4),
            duration_ratio=round(float(duration_ratio), 4),
            kana_similarity=0.0,
            transcript="",
            transcript_kana="",
            target_kana=target_kana,
            asr_provider=transcript.provider,
            content_verified=acoustic_status == "pass",
            method="mfcc_dtw_reference_gate",
            note=f"asr_unavailable_fallback_to_acoustic_gate: {transcript.note}",
        )

    try:
        transcript_kana = text_to_kana(transcript.text)
    except Exception:
        transcript_kana = kata_normalize(transcript.text)
    similarity = _kana_similarity(target_kana, transcript_kana)
    combined = 0.75 * similarity + 0.25 * acoustic_score

    if similarity < 0.45 or combined < 0.45:
        status = "fail"
    elif similarity < 0.75 or combined < 0.70:
        status = "uncertain"
    else:
        status = "pass"

    return ContentMatch(
        status=status,
        score=round(float(combined), 4),
        dtw_cost=round(float(dtw_cost), 4),
        duration_ratio=round(float(duration_ratio), 4),
        kana_similarity=round(float(similarity), 4),
        transcript=transcript.text,
        transcript_kana=transcript_kana,
        target_kana=target_kana,
        asr_provider=transcript.provider,
        content_verified=status == "pass",
        method="asr_kana_match+mfcc_dtw_reference_gate",
        note="asr_transcript_compared_as_kana",
    )


def estimate_content_match(
    cache: SentenceCache,
    y_speech: np.ndarray,
    sr: int,
    use_asr: bool = True,
    asr_policy: str = "if_acoustic_uncertain",
    asr_model: str = "small",
    asr_provider: str = "auto",
) -> ContentMatch:
    """Utterance-level content gate before pronunciation scoring.

    Prefer ASR kana matching when an optional backend is installed. Fall back to
    MFCC-DTW reference similarity when ASR is unavailable.
    """
    ref_duration = max(float(cache.meta.ref_duration_sec), 1e-6)
    user_duration = float(len(y_speech) / max(sr, 1))
    duration_ratio = user_duration / ref_duration
    target_kana = cache.meta.kana

    if y_speech.size < int(sr * 0.20):
        return _empty(
            status="fail",
            score=0.0,
            dtw_cost=float("inf"),
            duration_ratio=duration_ratio,
            target_kana=target_kana,
            method="mfcc_dtw_reference_gate",
            note="speech_too_short_for_content_match",
        )

    try:
        user_mfcc = _user_mfcc(y_speech, sr=sr)
        ref_mfcc = cache.ref_mfcc
        if ref_mfcc.ndim != 2 or user_mfcc.ndim != 2 or ref_mfcc.shape[1] < 2 or user_mfcc.shape[1] < 2:
            raise ValueError("insufficient_mfcc_frames")
        D, wp = librosa.sequence.dtw(X=ref_mfcc, Y=user_mfcc, metric="euclidean")
        dtw_cost = float(D[-1, -1] / max(len(wp), 1))
    except Exception:
        return _empty(
            status="unknown",
            score=0.0,
            dtw_cost=float("nan"),
            duration_ratio=duration_ratio,
            target_kana=target_kana,
            method="mfcc_dtw_reference_gate",
            note="content_match_failed_to_compute",
        )

    acoustic_score = _clip01(1.0 - (dtw_cost - 3.4) / 1.6)
    # Duration is mainly a fluency cue, not content correctness. Keep it as an
    # extreme sanity bound below, but do not blend moderate speaking-rate
    # differences into the content score itself.
    score = acoustic_score

    if duration_ratio < 0.45 or duration_ratio > 2.60 or dtw_cost > 5.20 or score < 0.25:
        status = "fail"
    elif dtw_cost > 4.70 or score < 0.55:
        status = "uncertain"
    else:
        status = "pass"

    asr_policy = str(asr_policy or "if_acoustic_uncertain").strip().lower()
    if asr_policy not in {"always", "if_acoustic_uncertain", "never"}:
        raise ValueError(f"Unknown ASR content-match policy: {asr_policy}")

    should_run_asr = bool(use_asr) and (
        asr_policy == "always"
        or (asr_policy == "if_acoustic_uncertain" and status != "pass")
    )
    if should_run_asr:
        return _asr_gate(
            cache=cache,
            y_speech=y_speech,
            sr=sr,
            acoustic_status=status,
            acoustic_score=score,
            dtw_cost=dtw_cost,
            duration_ratio=duration_ratio,
            model_name=asr_model,
            provider=asr_provider,
        )

    if use_asr and asr_policy == "if_acoustic_uncertain" and status == "pass":
        note = "acoustic_pass_asr_skipped_by_policy"
    elif not use_asr or asr_policy == "never":
        note = "coarse_acoustic_gate_without_asr"
    else:
        note = "coarse_acoustic_gate_not_asr_transcript"
    return ContentMatch(
        status=status,
        score=round(float(score), 4),
        dtw_cost=round(float(dtw_cost), 4),
        duration_ratio=round(float(duration_ratio), 4),
        kana_similarity=0.0,
        transcript="",
        transcript_kana="",
        target_kana=target_kana,
        asr_provider="none",
        content_verified=status == "pass",
        method="mfcc_dtw_reference_gate",
        note=note,
    )
