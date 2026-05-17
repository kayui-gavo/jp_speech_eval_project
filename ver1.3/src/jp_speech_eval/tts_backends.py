from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, Dict
from urllib import parse, request

import librosa
import numpy as np
import soundfile as sf

from .audio_features import trim_silence


@dataclass(frozen=True)
class TTSSynthesis:
    y: np.ndarray
    source: str
    metadata: Dict[str, Any]


def _normalize_audio(y: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    wav = np.asarray(y, dtype=np.float64)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0:
        wav = wav / (peak + 1e-9)
    if int(source_sr) != int(target_sr):
        wav = librosa.resample(wav, orig_sr=int(source_sr), target_sr=int(target_sr))
    wav, _ = trim_silence(wav, top_db=30.0)
    return wav.astype(np.float64)


def _pyopenjtalk_tts(text: str, sr: int) -> TTSSynthesis:
    import pyopenjtalk

    out = pyopenjtalk.tts(text)
    if isinstance(out, tuple):
        wav, tts_sr = out
    else:
        wav, tts_sr = out, 48000
    return TTSSynthesis(
        y=_normalize_audio(np.asarray(wav), int(tts_sr), sr),
        source="pyopenjtalk_tts_pseudo_reference",
        metadata={"backend": "pyopenjtalk"},
    )


def _voicevox_compatible_tts(
    text: str,
    sr: int,
    *,
    base_url: str,
    speaker: int,
    backend_name: str,
) -> TTSSynthesis:
    base = base_url.rstrip("/")
    query_url = f"{base}/audio_query?{parse.urlencode({'speaker': speaker, 'text': text})}"
    query_req = request.Request(query_url, data=b"", method="POST")
    with request.urlopen(query_req, timeout=20) as resp:
        audio_query = json.loads(resp.read().decode("utf-8"))

    synthesis_url = f"{base}/synthesis?{parse.urlencode({'speaker': speaker})}"
    body = json.dumps(audio_query, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        synthesis_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=60) as resp:
        wav_bytes = resp.read()
    wav, wav_sr = sf.read(io.BytesIO(wav_bytes), dtype="float64")
    return TTSSynthesis(
        y=_normalize_audio(np.asarray(wav), int(wav_sr), sr),
        source=f"{backend_name}_pseudo_reference",
        metadata={
            "backend": backend_name,
            "base_url": base,
            "speaker": int(speaker),
            "audio_query": audio_query,
        },
    )


def synthesize_reference(
    text: str,
    *,
    sr: int = 16000,
    backend: str = "pyopenjtalk",
    base_url: str | None = None,
    speaker: int | None = None,
) -> TTSSynthesis:
    backend = backend.strip().lower()
    if backend == "pyopenjtalk":
        return _pyopenjtalk_tts(text, sr)
    if backend in {"voicevox", "voicevox_http"}:
        return _voicevox_compatible_tts(
            text,
            sr,
            base_url=base_url or "http://127.0.0.1:50021",
            speaker=1 if speaker is None else int(speaker),
            backend_name="voicevox_http",
        )
    if backend in {"aivis", "aivisspeech", "aivis_http"}:
        if speaker is None:
            raise ValueError("aivis_http backend requires an explicit style/speaker id.")
        return _voicevox_compatible_tts(
            text,
            sr,
            base_url=base_url or "http://127.0.0.1:10101",
            speaker=int(speaker),
            backend_name="aivis_http",
        )
    raise ValueError(f"Unknown TTS backend: {backend}")
