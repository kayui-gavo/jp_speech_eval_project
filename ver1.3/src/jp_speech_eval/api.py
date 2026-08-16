from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .asr_confirmation import build_asr_confirmation_prompt
from .app_core.karaoke_timeline import build_consumer_karaoke_timeline
from .asr import AsrTranscript, detect_spoken_language
from .audio_features import load_audio
from .vad import trim_to_speech
from .eval_modes import evaluate_mode
from .feedback_renderer import render_user_facing_result
from .shadow_assessment import run_assessment_shadows
from .transcript_sanity import check_asr_transcript_sanity


FIXED_REFERENCE_REQUEST_MODES = {
    "reference",
    "reference_based",
    "reference_fixed_sentence",
    "fixed_reference",
}


def _has_japanese_script(text: str) -> bool:
    """Weak compatibility signal only; never an eligibility rule by itself."""
    return any(
        ("\u3040" <= char <= "\u309f")
        or ("\u30a0" <= char <= "\u30ff")
        or ("\u3400" <= char <= "\u9fff")
        for char in str(text or "")
    )


def _japanese_transcript_morphology(text: str) -> Dict[str, Any]:
    """Return lexical evidence for broad-fallback eligibility.

    This is deliberately evidence, not a grammaticality veto.  Japanese
    fixed-reading prompts can be valid noun phrases (``東京大学`` or
    ``大学院入学試験``), so an absence of a particle or verb cannot make the
    utterance ineligible by itself.  The only negative signal retained here is
    a very specific fragmented-token pattern observed in forced-Japanese ASR
    hallucinations; it must still be combined with voice and ASR evidence by
    :func:`_fallback_language_eligibility`.
    """
    try:
        import pyopenjtalk

        tokens = pyopenjtalk.run_frontend(str(text or ""))
    except Exception as exc:
        # Preserve the existing route if the optional text frontend is not
        # available; ASR language evidence still participates in eligibility.
        return {
            "available": False,
            "lexically_coherent": None,
            "reason": f"frontend_unavailable:{type(exc).__name__}",
            "token_count": 0,
            "pos": [],
        }

    content_tokens = [token for token in tokens if str(token.get("pos") or "") != "記号"]
    pos = [str(token.get("pos") or "") for token in content_tokens]
    # Do not reject ordinary multi-token noun compounds.  This narrow pattern
    # catches the known forced-Japanese Mandarin hallucination (a one-character
    # prefix followed by fragmented content words) without treating ordinary
    # nominal phrases as ungrammatical.
    first = content_tokens[0] if content_tokens else {}
    suspicious_fragment_sequence = (
        len(content_tokens) >= 4
        and str(first.get("pos") or "") == "接頭詞"
        and len(str(first.get("string") or "")) <= 1
    )
    return {
        "available": True,
        "lexically_coherent": not suspicious_fragment_sequence,
        "reason": "lexically_coherent" if not suspicious_fragment_sequence else "suspicious_fragment_sequence",
        "token_count": len(content_tokens),
        "pos": pos,
        "tokens": [str(token.get("string") or "") for token in content_tokens],
    }


def _fallback_language_eligibility(
    transcript: str,
    *,
    speech_detected: bool,
    f0_coverage: float,
    evidence: AsrTranscript,
) -> Dict[str, Any]:
    """Separate broad-mode language eligibility from target text verification.

    Unforced ASR language evidence can reject a confident non-Japanese
    observation, but short-utterance language ID is not assumed infallible.
    If it is inconclusive, this returns ``uncertain`` rather than treating
    morphology as a final veto.  Callers may retain a broad score with reduced
    confidence for that state; only clear non-speech/non-Japanese or a joint
    hallucination signal is ineligible.
    """
    language = str(evidence.language or "").lower()
    probability = evidence.language_probability
    probability = float(probability) if probability is not None else None
    # VAD can miss a very short but clearly periodic Japanese response (e.g.
    # いいえ).  Treat reliable periodic voice evidence as a backup to VAD;
    # transcript sanity and lexical coherence still protect the broad route.
    voiced = bool(speech_detected) or float(f0_coverage) >= 0.10
    script_compatible = _has_japanese_script(transcript)
    morphology = _japanese_transcript_morphology(transcript)
    payload = {
        "language": language,
        "language_probability": probability,
        "speech_present": bool(speech_detected),
        "voice_evidence": float(f0_coverage) >= 0.10,
        "script_compatible": script_compatible,
        "morphology": morphology,
    }
    if not voiced:
        return {"ok": False, "eligibility": "ineligible", "reason": "insufficient_voiced_speech_evidence", **payload}
    # Short utterance language labels below 0.70 are empirically unstable on
    # this backend.  They are evidence, not a veto against short Japanese
    # words; a confident non-Japanese label remains a safety rejection.
    if evidence.available and language and language != "ja" and (probability is None or probability >= 0.70):
        return {"ok": False, "eligibility": "ineligible", "reason": "detected_non_japanese", **payload}
    if (
        evidence.available
        and language == "ja"
        and (probability is None or probability >= 0.35)
        and morphology.get("lexically_coherent") is False
    ):
        return {"ok": False, "eligibility": "ineligible", "reason": "joint_asr_hallucination_evidence", **payload}
    if evidence.available and language == "ja" and (probability is None or probability >= 0.35):
        return {"ok": True, "eligibility": "eligible", "reason": "detected_japanese", **payload}
    if script_compatible:
        return {"ok": True, "eligibility": "uncertain", "reason": "inconclusive_language_japanese_compatible", **payload}
    return {"ok": False, "eligibility": "ineligible", "reason": "no_safe_japanese_language_evidence", **payload}


def _fallback_language_evidence(audio_path: str, sample_rate: int, model_name: str = "small") -> tuple[bool, AsrTranscript]:
    """Do not let optional language detection crash the product fallback."""
    try:
        audio = load_audio(audio_path, sr=sample_rate)
        _trimmed, region = trim_to_speech(audio.y, audio.sr)
        if not region.detected:
            return False, AsrTranscript(False, "none", model_name, "", "", "no_speech_for_language_detection")
        return True, detect_spoken_language(audio.y, audio.sr, model_name=model_name, provider="auto")
    except Exception as exc:
        return False, AsrTranscript(False, "none", model_name, "", "", f"language_detection_failed:{type(exc).__name__}")


@dataclass(frozen=True)
class SpeechEvalConfig:
    """Stable configuration for external pipeline integration."""

    cache_path: Optional[str] = "cache/ramen_kudasai"
    scoring_config_path: Optional[str] = None
    sample_rate: int = 16000
    tts_backend: str = "pyopenjtalk"
    tts_backend_url: Optional[str] = None
    tts_speaker: Optional[int] = None
    tts_model: Optional[str] = None
    tts_voice: Optional[str] = None
    tts_speed: Optional[float] = None
    tts_style: Optional[str] = None
    tts_prompt: Optional[str] = None
    tts_language: str = "ja-JP"
    special_mora_threshold_profile: str = "default_safe"
    enable_runtime_special_mora_shadow: bool = True
    enable_user_facing_calibrated_special_mora: bool = False
    enable_weak_reference_special_mora_hint: bool = False
    enable_ssl_shadow: bool = False
    ssl_shadow_model: str = "microsoft/wavlm-large"
    ssl_shadow_layer: int = 12
    ssl_shadow_timeout_sec: float = 30.0
    enable_special_mora_v2_shadow: bool = False
    enable_phrase_intonation_shadow: bool = False
    enable_accent_nucleus_shadow: bool = False


@dataclass(frozen=True)
class EvaluationRequest:
    """Request object for one utterance evaluation."""

    audio_path: str
    mode: str = "reference"
    target_text: Optional[str] = None
    user_confirmed_text: Optional[str] = None
    transcript: Optional[str] = None
    cache_path: Optional[str] = None
    scoring_config_path: Optional[str] = None
    sample_rate: Optional[int] = None
    tts_backend: Optional[str] = None
    tts_backend_url: Optional[str] = None
    tts_speaker: Optional[int] = None
    tts_model: Optional[str] = None
    tts_voice: Optional[str] = None
    tts_speed: Optional[float] = None
    tts_style: Optional[str] = None
    tts_prompt: Optional[str] = None
    tts_language: Optional[str] = None


@dataclass(frozen=True)
class EvaluationResponse:
    """Public response returned to external callers."""

    ok: bool
    mode: str
    user_facing: Dict[str, Any]
    raw_result: Dict[str, Any]
    karaoke_timeline: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AsrConfirmResponse:
    """ASR confirmation payload for weak-reference modes."""

    ok: bool
    prompt: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _product_fallback_after_target_mismatch(
    raw: Dict[str, Any],
    request: EvaluationRequest,
    config: SpeechEvalConfig,
) -> Dict[str, Any]:
    """Use broad Japanese scoring when a fixed target mismatches but ASR is valid.

    The fixed-reference result is still preserved as compact debug metadata. If
    no plausible Japanese transcript is available, keep the original result and
    let the normal user-facing policy decide how conservative to be.
    """

    if str(request.mode or "").strip() not in FIXED_REFERENCE_REQUEST_MODES:
        return raw
    details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
    content = details.get("content_match") if isinstance(details.get("content_match"), dict) else {}
    if str(content.get("status") or "") not in {"fail", "failed", "content_mismatch"}:
        return raw

    transcript = str(content.get("transcript") or "").strip()
    if not transcript:
        reliability = details.get("reliability") if isinstance(details.get("reliability"), dict) else {}
        if reliability and float(reliability.get("f0_coverage", 0.0) or 0.0) < 0.08:
            details["transcript_sanity"] = {
                "ok": False, "score": 0.0, "reason": "nonvoice_or_no_speech_without_transcript",
                "normalized_text": "", "metrics": {"f0_coverage": reliability.get("f0_coverage", 0.0)},
            }
            details["scoring_ineligibility_reason"] = "NO_SPEECH_OR_NONVOICE"
        return raw
    sanity = check_asr_transcript_sanity(transcript)
    # ASR may hallucinate fluent Japanese on stationary noise.  Broad fallback
    # eligibility therefore uses independently detected speech/language
    # evidence, not the Japanese-biased target-verification transcript.
    reliability = details.get("reliability") if isinstance(details.get("reliability"), dict) else {}
    f0_coverage = float(reliability.get("f0_coverage", 1.0) or 0.0)
    speech_detected, language_evidence = _fallback_language_evidence(
        str(request.audio_path), request.sample_rate or config.sample_rate
    )
    eligibility = _fallback_language_eligibility(
        transcript,
        speech_detected=speech_detected,
        f0_coverage=f0_coverage,
        evidence=language_evidence,
    )
    if sanity.ok and not eligibility["ok"]:
        payload = sanity.to_dict()
        payload.update({
            "ok": False,
            "score": 0.0,
            "reason": "fallback_audio_or_language_ineligible",
            "metrics": {**payload.get("metrics", {}), "f0_coverage": round(f0_coverage, 4), **eligibility},
        })
        details["transcript_sanity"] = payload
        details["scoring_ineligibility_reason"] = "NON_JAPANESE_OR_NONVOICE_FALLBACK"
        return raw
    if not sanity.ok:
        details["transcript_sanity"] = sanity.to_dict()
        return raw

    sample_rate = request.sample_rate or config.sample_rate
    scoring_config_path = (
        request.scoring_config_path
        if request.scoring_config_path is not None
        else config.scoring_config_path
    )
    general = evaluate_mode(
        "transcript_assisted_light",
        request.audio_path,
        transcript=transcript,
        scoring_config_path=scoring_config_path,
        sample_rate=sample_rate,
    )
    general_details = general.setdefault("details", {})
    general_details["mode"] = "reference_mismatch_general_japanese"
    general_details["task_target_text"] = raw.get("target_text")
    general_details["task_content_match"] = dict(content)
    general_details["transcript_sanity"] = sanity.to_dict()
    general_details["fallback_language_eligibility"] = eligibility
    general_details["fallback_reason"] = "target_mismatch_but_plausible_japanese"
    general_details["fixed_reference_debug"] = {
        "pronunciation_score": raw.get("pronunciation_score"),
        "prosody_score": raw.get("prosody_score"),
        "fluency_score": raw.get("fluency_score"),
        "total_score": raw.get("total_score"),
        "alignment_mode": raw.get("alignment_mode"),
        "cache_prefix": raw.get("cache_prefix"),
    }
    general_details["content_match"] = {
        "status": "general_japanese",
        "content_verified": False,
        "japanese_content_plausible": True,
        "transcript": transcript,
        "note": "broad_scoring_after_fixed_target_mismatch",
    }
    general["feedback"] = [
        "目標文とは違う内容でしたが、日本語として全体の話し方を評価しました。"
    ] + list(general.get("feedback") or [])
    return general


class SpeechEvaluationClient:
    """Small public SDK for calling the Japanese speech evaluation pipeline."""

    def __init__(self, config: SpeechEvalConfig | None = None) -> None:
        self.config = config or SpeechEvalConfig()

    def build_asr_confirmation(
        self,
        audio_path: str | Path,
        *,
        asr_model: str = "small",
        asr_provider: str = "auto",
        sample_rate: Optional[int] = None,
    ) -> AsrConfirmResponse:
        try:
            prompt = build_asr_confirmation_prompt(
                audio_path,
                sample_rate=sample_rate or self.config.sample_rate,
                asr_model=asr_model,
                asr_provider=asr_provider,
            )
        except Exception as exc:
            return AsrConfirmResponse(ok=False, error=str(exc))
        return AsrConfirmResponse(ok=True, prompt=prompt.to_dict())

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        try:
            raw = evaluate_mode(
                request.mode,
                request.audio_path,
                cache_path=request.cache_path if request.cache_path is not None else self.config.cache_path,
                target_text=request.target_text,
                transcript=request.transcript,
                user_confirmed_text=request.user_confirmed_text,
                scoring_config_path=request.scoring_config_path if request.scoring_config_path is not None else self.config.scoring_config_path,
                sample_rate=request.sample_rate or self.config.sample_rate,
                tts_backend=request.tts_backend or self.config.tts_backend,
                tts_backend_url=request.tts_backend_url if request.tts_backend_url is not None else self.config.tts_backend_url,
                tts_speaker=request.tts_speaker if request.tts_speaker is not None else self.config.tts_speaker,
                tts_model=request.tts_model if request.tts_model is not None else self.config.tts_model,
                tts_voice=request.tts_voice if request.tts_voice is not None else self.config.tts_voice,
                tts_speed=request.tts_speed if request.tts_speed is not None else self.config.tts_speed,
                tts_style=request.tts_style if request.tts_style is not None else self.config.tts_style,
                tts_prompt=request.tts_prompt if request.tts_prompt is not None else self.config.tts_prompt,
                tts_language=request.tts_language or self.config.tts_language,
            )
            raw = _product_fallback_after_target_mismatch(raw, request, self.config)
            run_assessment_shadows(
                raw,
                user_audio_path=request.audio_path,
                sample_rate=request.sample_rate or self.config.sample_rate,
                enable_ssl_shadow=self.config.enable_ssl_shadow,
                ssl_model_id=self.config.ssl_shadow_model,
                ssl_layer=self.config.ssl_shadow_layer,
                ssl_timeout_sec=self.config.ssl_shadow_timeout_sec,
                enable_special_mora_v2_shadow=self.config.enable_special_mora_v2_shadow,
                enable_phrase_intonation_shadow=self.config.enable_phrase_intonation_shadow,
                enable_accent_nucleus_shadow=self.config.enable_accent_nucleus_shadow,
            )
            effective_mode = str(raw.get("details", {}).get("mode") or request.mode)
            user_facing = render_user_facing_result(
                raw,
                mode=effective_mode,
                special_mora_threshold_profile=self.config.special_mora_threshold_profile,
                enable_runtime_special_mora_shadow=self.config.enable_runtime_special_mora_shadow,
                enable_user_facing_calibrated_special_mora=self.config.enable_user_facing_calibrated_special_mora,
                enable_weak_reference_special_mora_hint=self.config.enable_weak_reference_special_mora_hint,
            )
        except Exception as exc:
            return EvaluationResponse(
                ok=False,
                mode=request.mode,
                user_facing={},
                raw_result={},
                error=str(exc),
            )
        return EvaluationResponse(
            ok=True,
            mode=effective_mode,
            user_facing=user_facing,
            raw_result=raw,
            karaoke_timeline=build_consumer_karaoke_timeline(raw, user_facing),
        )


def evaluate_speech(request: EvaluationRequest, config: SpeechEvalConfig | None = None) -> Dict[str, Any]:
    """Convenience function for one-shot external calls."""

    return SpeechEvaluationClient(config).evaluate(request).to_dict()


def build_asr_confirmation(audio_path: str | Path, config: SpeechEvalConfig | None = None, **kwargs: Any) -> Dict[str, Any]:
    """Convenience function for the ASR confirmation step."""

    return SpeechEvaluationClient(config).build_asr_confirmation(audio_path, **kwargs).to_dict()
