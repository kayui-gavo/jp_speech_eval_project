from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import soundfile as sf


def normalize_reference_text(text: str) -> str:
    """Normalize text only enough to make cache keys reproducible."""
    normalized = unicodedata.normalize("NFKC", str(text)).strip()
    return re.sub(r"\s+", " ", normalized)


def canonical_json(data: Mapping[str, Any]) -> str:
    """Return a deterministic JSON string for cache hashing."""
    return json.dumps(dict(data), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_reference_config(
    *,
    text: str,
    provider: str,
    model: str | None,
    voice: str | None,
    speed: float | None,
    style: str | None,
    prompt: str | None,
    language: str,
    sample_rate: int,
) -> Dict[str, Any]:
    """Build the stable subset of synthesis settings used in cache hashing."""
    return {
        "normalized_text": normalize_reference_text(text),
        "provider": provider,
        "model": model,
        "voice": voice,
        "speed": speed,
        "style": style,
        "prompt": prompt,
        "language": language,
        "sample_rate": int(sample_rate),
    }


def build_reference_hash(config: Mapping[str, Any]) -> str:
    """Hash a canonical synthesis config without timestamps or paths."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def safe_target_id(text: str) -> str:
    """Return an ASCII-safe stable target id for cache directories."""
    digest = hashlib.sha1(normalize_reference_text(text).encode("utf-8")).hexdigest()[:12]
    return f"target_{digest}"


@dataclass(frozen=True)
class ReferenceArtifact:
    """Persisted TTS reference audio plus reproducibility metadata."""

    text: str
    normalized_text: str
    target_id: str
    provider: str
    model: Optional[str]
    voice: Optional[str]
    speed: Optional[float]
    style: Optional[str]
    prompt: Optional[str]
    language: str
    sample_rate: int
    created_at: str
    config_hash: str
    audio_path: str
    metadata_path: str
    reference_role: str
    cache_hit: bool = False
    provider_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReferenceStore:
    """Filesystem-backed store for provider-generated raw reference audio."""

    def __init__(self, cache_dir: str | Path = "data/tts_cache") -> None:
        self.cache_dir = Path(cache_dir)

    def artifact_dir(self, target_id: str, provider: str, config_hash: str) -> Path:
        return self.cache_dir / provider / target_id / config_hash

    def _paths(self, target_id: str, provider: str, config_hash: str) -> tuple[Path, Path]:
        root = self.artifact_dir(target_id, provider, config_hash)
        return root / "audio.wav", root / "metadata.json"

    def get_cached(self, *, target_id: str, provider: str, config_hash: str) -> ReferenceArtifact | None:
        audio_path, metadata_path = self._paths(target_id, provider, config_hash)
        if not audio_path.exists() or not metadata_path.exists():
            return None
        with metadata_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["cache_hit"] = True
        raw.setdefault("metadata_path", str(metadata_path))
        return ReferenceArtifact(**raw)

    def save(
        self,
        *,
        text: str,
        target_id: str,
        provider: str,
        model: str | None,
        voice: str | None,
        speed: float | None,
        style: str | None,
        prompt: str | None,
        language: str,
        sample_rate: int,
        config_hash: str,
        y: np.ndarray,
        reference_role: str = "pseudo_reference",
        provider_metadata: Mapping[str, Any] | None = None,
    ) -> ReferenceArtifact:
        audio_path, metadata_path = self._paths(target_id, provider, config_hash)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(audio_path), np.asarray(y, dtype=np.float32), int(sample_rate))
        artifact = ReferenceArtifact(
            text=text,
            normalized_text=normalize_reference_text(text),
            target_id=target_id,
            provider=provider,
            model=model,
            voice=voice,
            speed=speed,
            style=style,
            prompt=prompt,
            language=language,
            sample_rate=int(sample_rate),
            created_at=datetime.now(timezone.utc).isoformat(),
            config_hash=config_hash,
            audio_path=str(audio_path),
            metadata_path=str(metadata_path),
            reference_role=reference_role,
            cache_hit=False,
            provider_metadata=dict(provider_metadata or {}),
        )
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(artifact.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)
        return artifact
