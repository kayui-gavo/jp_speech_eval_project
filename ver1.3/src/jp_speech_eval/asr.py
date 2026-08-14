from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np


_FASTER_WHISPER_CACHE: Dict[tuple[str, str, str], object] = {}
_OPENAI_WHISPER_CACHE: Dict[str, object] = {}


def _error_note(exc: Exception) -> str:
    """Keep optional-backend errors one-line and CSV/JSONL safe."""
    return f"{type(exc).__name__}: {' '.join(str(exc).split())}"


@dataclass(frozen=True)
class AsrTranscript:
    available: bool
    provider: str
    model: str
    text: str
    language: str
    note: str
    language_probability: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def transcribe_japanese(
    y: np.ndarray,
    sr: int,
    model_name: str = "small",
    provider: str = "auto",
) -> AsrTranscript:
    """Transcribe a recording under an explicit Japanese decoding constraint.

    This helper is intentionally reserved for fixed-target verification, where
    the task already asserts that the expected content is Japanese.  It must
    not be used for free-speaking language eligibility or ASR confirmation,
    because forcing ``language='ja'`` can render English or other languages as
    plausible-looking Japanese text.
    """
    provider = provider.lower().strip()
    if provider in {"auto", "faster-whisper", "faster_whisper"}:
        out = _try_faster_whisper(y, sr, model_name, language="ja")
        if out.available or provider in {"faster-whisper", "faster_whisper"}:
            return out
    if provider in {"auto", "whisper", "openai-whisper", "openai_whisper"}:
        out = _try_openai_whisper(y, sr, model_name, language="ja")
        if out.available or provider != "auto":
            return out
    return AsrTranscript(
        available=False,
        provider=provider,
        model=model_name,
        text="",
        language="ja",
        note="asr_unavailable_install_faster_whisper_or_openai_whisper",
    )


def transcribe_language_aware(
    y: np.ndarray,
    sr: int,
    model_name: str = "small",
    provider: str = "auto",
) -> AsrTranscript:
    """Transcribe without forcing Japanese and always use transcription mode.

    Free-speaking flows must start here.  ``language=None`` lets Whisper keep
    its own language observation, and ``task='transcribe'`` prevents the
    translation task from being requested.  Callers can then reject confident
    non-Japanese speech before creating a Japanese pseudo-reference.
    """
    provider = provider.lower().strip()
    if provider in {"auto", "faster-whisper", "faster_whisper"}:
        out = _try_faster_whisper(y, sr, model_name, language=None)
        if out.available or provider in {"faster-whisper", "faster_whisper"}:
            return out
    if provider in {"auto", "whisper", "openai-whisper", "openai_whisper"}:
        out = _try_openai_whisper(y, sr, model_name, language=None)
        if out.available or provider != "auto":
            return out
    return AsrTranscript(False, provider, model_name, "", "", "language_aware_asr_unavailable")


def detect_spoken_language(
    y: np.ndarray,
    sr: int,
    model_name: str = "small",
    provider: str = "auto",
) -> AsrTranscript:
    """Obtain independent language evidence without forcing Japanese."""
    return transcribe_language_aware(y, sr, model_name=model_name, provider=provider)


def _try_faster_whisper(
    y: np.ndarray,
    sr: int,
    model_name: str,
    language: Optional[str] = "ja",
) -> AsrTranscript:
    try:
        from faster_whisper import WhisperModel
        import soundfile as sf
        import tempfile
    except Exception:
        return AsrTranscript(False, "faster-whisper", model_name, "", language or "", "faster_whisper_not_installed")

    try:
        cache_key = (model_name, "cpu", "int8")
        model = _FASTER_WHISPER_CACHE.get(cache_key)
        if model is None:
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            _FASTER_WHISPER_CACHE[cache_key] = model
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            sf.write(f.name, np.asarray(y, dtype=np.float32), sr)
            segments, info = model.transcribe(
                f.name,
                language=language,
                task="transcribe",
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            text = "".join(seg.text for seg in segments).strip()
            detected_language = getattr(info, "language", language or "") or ""
            probability = getattr(info, "language_probability", None)
            probability = float(probability) if probability is not None else None
        return AsrTranscript(True, "faster-whisper", model_name, text, detected_language, "ok", probability)
    except Exception as exc:
        return AsrTranscript(False, "faster-whisper", model_name, "", language or "", _error_note(exc))


def _try_openai_whisper(
    y: np.ndarray,
    sr: int,
    model_name: str,
    language: Optional[str] = "ja",
) -> AsrTranscript:
    try:
        import whisper
    except Exception:
        return AsrTranscript(False, "openai-whisper", model_name, "", language or "", "whisper_not_installed")

    try:
        audio = np.asarray(y, dtype=np.float32)
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        model = _OPENAI_WHISPER_CACHE.get(model_name)
        if model is None:
            model = whisper.load_model(model_name)
            _OPENAI_WHISPER_CACHE[model_name] = model
        result = model.transcribe(
            audio,
            language=language,
            task="transcribe",
            fp16=False,
            condition_on_previous_text=False,
        )
        return AsrTranscript(
            True,
            "openai-whisper",
            model_name,
            str(result.get("text", "")).strip(),
            str(result.get("language", language or "") or ""),
            "ok",
        )
    except Exception as exc:
        return AsrTranscript(False, "openai-whisper", model_name, "", language or "", _error_note(exc))
