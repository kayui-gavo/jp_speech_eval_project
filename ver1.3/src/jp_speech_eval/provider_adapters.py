from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union


@dataclass(frozen=True)
class ASRProviderResult:
    transcript: str
    language: str
    confidence: float | None
    segments: List[Dict[str, Any]] = field(default_factory=list)
    word_timestamps: List[Dict[str, Any]] = field(default_factory=list)
    provider_name: str = "fixture"
    model_name: str = "fixture"
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    result_source: str = "offline_fixture"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TTSProviderResult:
    audio_path: str | None
    sample_rate: int | None
    provider_name: str
    model_name: str
    voice: str | None = None
    prompt_or_style: str | None = None
    generated_at: str | None = None
    provenance: str = "synthetic_tts"
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provenance != "synthetic_tts":
            raise ValueError("TTS provider fixtures must use provenance=synthetic_tts")

    @property
    def pitch_target_reliability(self) -> str:
        return "weak"

    @property
    def strong_pitch_reference_allowed(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["pitch_target_reliability"] = self.pitch_target_reliability
        data["strong_pitch_reference_allowed"] = self.strong_pitch_reference_allowed
        return data


ProviderResult = Union[ASRProviderResult, TTSProviderResult]


def provider_result_from_mapping(payload: Mapping[str, Any]) -> ProviderResult:
    kind = str(payload.get("kind") or "").strip().lower()
    if kind == "asr":
        return ASRProviderResult(
            transcript=str(payload.get("transcript") or ""),
            language=str(payload.get("language") or ""),
            confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
            segments=[dict(item) for item in (payload.get("segments") or [])],
            word_timestamps=[dict(item) for item in (payload.get("word_timestamps") or [])],
            provider_name=str(payload.get("provider_name") or "fixture"),
            model_name=str(payload.get("model_name") or "fixture"),
            raw_metadata=dict(payload.get("raw_metadata") or {}),
            result_source=str(payload.get("result_source") or "offline_fixture"),
        )
    if kind == "tts":
        return TTSProviderResult(
            audio_path=str(payload["audio_path"]) if payload.get("audio_path") else None,
            sample_rate=int(payload["sample_rate"]) if payload.get("sample_rate") is not None else None,
            provider_name=str(payload.get("provider_name") or "fixture"),
            model_name=str(payload.get("model_name") or "fixture"),
            voice=str(payload["voice"]) if payload.get("voice") is not None else None,
            prompt_or_style=str(payload["prompt_or_style"]) if payload.get("prompt_or_style") is not None else None,
            generated_at=str(payload["generated_at"]) if payload.get("generated_at") is not None else None,
            provenance=str(payload.get("provenance") or "synthetic_tts"),
            raw_metadata=dict(payload.get("raw_metadata") or {}),
        )
    raise ValueError(f"unknown provider fixture kind: {kind or '<missing>'}")


def load_provider_result_fixture(path: str | Path) -> ProviderResult | List[ProviderResult]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [provider_result_from_mapping(item) for item in payload]
    if not isinstance(payload, Mapping):
        raise ValueError("provider fixture must contain an object or list of objects")
    return provider_result_from_mapping(payload)
