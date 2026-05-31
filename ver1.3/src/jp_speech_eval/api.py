from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .asr_confirmation import build_asr_confirmation_prompt
from .eval_modes import evaluate_mode
from .feedback_renderer import render_user_facing_result


@dataclass(frozen=True)
class SpeechEvalConfig:
    """Stable configuration for external pipeline integration.

    This is intentionally small. Heavy experiment knobs should stay in lower
    level modules; app or pipeline code should mostly decide the mode, audio
    path, target/cache, and optional TTS provider.
    """

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
    """Public response returned to external callers.

    `user_facing` is the recommended object for product UI. `raw_result` keeps
    current internal metrics for debugging and research inspection.
    """

    ok: bool
    mode: str
    user_facing: Dict[str, Any]
    raw_result: Dict[str, Any]
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


class SpeechEvaluationClient:
    """Small public SDK for calling the Japanese speech evaluation pipeline.

    Typical usage:

        client = SpeechEvaluationClient(SpeechEvalConfig(cache_path="cache/ramen_kudasai"))
        response = client.evaluate(EvaluationRequest(audio_path="user.wav", mode="reference"))

    ASR-based modes should be two-step:
        1. `build_asr_confirmation(audio_path)` to show/edit candidate text.
        2. `evaluate(..., mode="asr_confirmed_weak_reference", user_confirmed_text=...)`.
    """

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
            user_facing = render_user_facing_result(
                raw,
                mode=request.mode,
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
            mode=str(raw.get("details", {}).get("mode") or request.mode),
            user_facing=user_facing,
            raw_result=raw,
        )


def evaluate_speech(request: EvaluationRequest, config: SpeechEvalConfig | None = None) -> Dict[str, Any]:
    """Convenience function for one-shot external calls."""

    return SpeechEvaluationClient(config).evaluate(request).to_dict()


def build_asr_confirmation(audio_path: str | Path, config: SpeechEvalConfig | None = None, **kwargs: Any) -> Dict[str, Any]:
    """Convenience function for the ASR confirmation step."""

    return SpeechEvaluationClient(config).build_asr_confirmation(audio_path, **kwargs).to_dict()
