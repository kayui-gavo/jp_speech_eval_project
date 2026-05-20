from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import soundfile as sf

from .reference_store import (
    ReferenceArtifact,
    ReferenceStore,
    build_reference_config,
    build_reference_hash,
    safe_target_id,
)
from .tts_backends import TTSSynthesis, synthesize_reference


PROVIDER_ALIASES = {
    "pyopenjtalk": "local_pyopenjtalk",
    "local_pyopenjtalk": "local_pyopenjtalk",
    "voicevox": "local_voicevox",
    "voicevox_http": "local_voicevox",
    "local_voicevox": "local_voicevox",
    "aivis": "local_aivis",
    "aivisspeech": "local_aivis",
    "aivis_http": "local_aivis",
    "local_aivis": "local_aivis",
    "openai": "openai",
    "google": "google",
    "elevenlabs": "elevenlabs",
    "azure": "azure",
    "local_style_bert_vits2": "local_style_bert_vits2",
}

LEGACY_BACKENDS = {
    "local_pyopenjtalk": "pyopenjtalk",
    "local_voicevox": "voicevox_http",
    "local_aivis": "aivis_http",
}

RESERVED_PROVIDERS = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_APPLICATION_CREDENTIALS",
    "elevenlabs": "ELEVENLABS_API_KEY",
    "azure": "AZURE_SPEECH_KEY",
    "local_style_bert_vits2": "STYLE_BERT_VITS2_URL",
}


class TTSProviderUnavailableError(RuntimeError):
    """Raised when a requested provider is unavailable or not implemented."""


def canonical_provider_name(provider: str) -> str:
    """Normalize legacy backend names into stable provider ids."""
    normalized = str(provider or "").strip().lower()
    if normalized not in PROVIDER_ALIASES:
        raise ValueError(f"Unknown TTS provider: {provider}")
    return PROVIDER_ALIASES[normalized]


@dataclass(frozen=True)
class TTSRequest:
    """Provider-agnostic synthesis request."""

    text: str
    language: str = "ja-JP"
    voice: str | None = None
    speed: float | None = None
    style: str | None = None
    prompt: str | None = None
    provider: str = "local_pyopenjtalk"
    model: str | None = None
    sample_rate: int = 16000
    target_id: str | None = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_provider(self) -> str:
        return canonical_provider_name(self.provider)

    @property
    def resolved_target_id(self) -> str:
        return self.target_id or safe_target_id(self.text)

    def config(self) -> Dict[str, Any]:
        return build_reference_config(
            text=self.text,
            provider=self.canonical_provider,
            model=self.model,
            voice=self.voice,
            speed=self.speed,
            style=self.style,
            prompt=self.prompt,
            language=self.language,
            sample_rate=self.sample_rate,
        )

    @property
    def config_hash(self) -> str:
        return build_reference_hash(self.config())


class TTSAdapter:
    """Provider-neutral synthesis facade with deterministic artifact caching."""

    def __init__(self, cache_dir: str | Path = "data/tts_cache") -> None:
        self.store = ReferenceStore(cache_dir)

    def list_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """Return availability info without failing if optional providers lack keys."""
        providers = [
            "local_pyopenjtalk",
            "local_voicevox",
            "local_aivis",
            "local_style_bert_vits2",
            "openai",
            "google",
            "elevenlabs",
            "azure",
        ]
        return {provider: self.validate_provider_config(provider) for provider in providers}

    def validate_provider_config(self, provider: str) -> Dict[str, Any]:
        canonical = canonical_provider_name(provider)
        if canonical in LEGACY_BACKENDS:
            return {
                "provider": canonical,
                "available": True,
                "implemented": True,
                "reason": "local_or_http_provider_ready",
            }
        if canonical == "google":
            configured = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
            try:
                import google.cloud.texttospeech  # noqa: F401

                installed = True
            except ImportError:
                installed = False
            return {
                "provider": canonical,
                "available": bool(configured and installed),
                "implemented": True,
                "configured": configured,
                "installed": installed,
                "required_env": "GOOGLE_APPLICATION_CREDENTIALS",
                "reason": "ready" if configured and installed else "missing_credentials_or_package",
                "default_voice": os.environ.get("GOOGLE_TTS_VOICE") or "ja-JP-Chirp3-HD-Achernar",
            }
        env_var = RESERVED_PROVIDERS.get(canonical)
        configured = bool(env_var and os.environ.get(env_var))
        return {
            "provider": canonical,
            "available": False,
            "implemented": False,
            "configured": configured,
            "required_env": env_var,
            "reason": "provider_reserved_not_implemented",
        }

    def get_cached_reference(self, request: TTSRequest) -> ReferenceArtifact | None:
        return self.store.get_cached(
            target_id=request.resolved_target_id,
            provider=request.canonical_provider,
            config_hash=request.config_hash,
        )

    def synthesize(
        self,
        request: TTSRequest,
        *,
        output_path: str | Path | None = None,
        use_cache: bool = True,
        reference_role: str = "pseudo_reference",
    ) -> ReferenceArtifact:
        """Synthesize one reference artifact, reusing identical cached requests."""
        if use_cache:
            cached = self.get_cached_reference(request)
            if cached is not None:
                if output_path is not None:
                    self._copy_audio(cached, output_path)
                return cached

        synth = self._synthesize_uncached(request)
        artifact = self.store.save(
            text=request.text,
            target_id=request.resolved_target_id,
            provider=request.canonical_provider,
            model=request.model,
            voice=request.voice,
            speed=request.speed,
            style=request.style,
            prompt=request.prompt,
            language=request.language,
            sample_rate=request.sample_rate,
            config_hash=request.config_hash,
            y=synth.y,
            reference_role=reference_role,
            provider_metadata={
                "source": synth.source,
                **dict(synth.metadata),
            },
        )
        if output_path is not None:
            self._copy_audio(artifact, output_path)
        return artifact

    def synthesize_reference_set(
        self,
        *,
        text: str,
        target_id: str,
        requests: Iterable[TTSRequest],
        reference_role: str = "pseudo_reference",
    ) -> List[ReferenceArtifact]:
        """Generate or reuse a deterministic set of references for one target."""
        artifacts: List[ReferenceArtifact] = []
        for request in requests:
            if request.text != text:
                raise ValueError("All reference-set requests must use the same text.")
            if request.target_id not in {None, target_id}:
                raise ValueError("Reference-set request target_id conflicts with target_id.")
            normalized = TTSRequest(
                text=request.text,
                language=request.language,
                voice=request.voice,
                speed=request.speed,
                style=request.style,
                prompt=request.prompt,
                provider=request.provider,
                model=request.model,
                sample_rate=request.sample_rate,
                target_id=target_id,
                extra=dict(request.extra),
            )
            artifacts.append(self.synthesize(normalized, reference_role=reference_role))
        return artifacts

    def _synthesize_uncached(self, request: TTSRequest) -> TTSSynthesis:
        provider = request.canonical_provider
        status = self.validate_provider_config(provider)
        if provider == "google":
            if not status["available"]:
                raise TTSProviderUnavailableError(
                    "google TTS is implemented but not available. Configure "
                    "GOOGLE_APPLICATION_CREDENTIALS and install google-cloud-texttospeech."
                )
            return synthesize_reference(
                request.text,
                sr=request.sample_rate,
                backend="google",
                model=request.model,
                voice=request.voice,
                speed=request.speed,
                style=request.style,
                prompt=request.prompt,
                language=request.language,
            )
        if not status["implemented"]:
            raise TTSProviderUnavailableError(
                f"{provider} is reserved but not implemented in this build."
            )
        backend = LEGACY_BACKENDS[provider]
        speaker = request.extra.get("speaker")
        if speaker is None and request.voice is not None:
            try:
                speaker = int(request.voice)
            except (TypeError, ValueError):
                speaker = None
        return synthesize_reference(
            request.text,
            sr=request.sample_rate,
            backend=backend,
            base_url=request.extra.get("base_url"),
            speaker=speaker,
            model=request.model,
            voice=request.voice,
            speed=request.speed,
            style=request.style,
            prompt=request.prompt,
            language=request.language,
        )

    @staticmethod
    def _copy_audio(artifact: ReferenceArtifact, output_path: str | Path) -> None:
        y, sr = sf.read(artifact.audio_path, dtype="float32")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), np.asarray(y), int(sr))
