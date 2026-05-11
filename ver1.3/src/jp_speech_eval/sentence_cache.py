from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np

from .audio_features import extract_f0, trim_silence
from .text_frontend import TextInfo, build_text_info


@dataclass(frozen=True)
class SentenceMeta:
    text: str
    kana: str
    moras: List[str]
    target_pitch: List[str]
    pitch_target_source: str
    is_question: bool
    sr: int
    ref_duration_sec: float
    ref_mora_boundaries: List[Tuple[float, float]]
    frontend_raw: List[Dict[str, Any]]
    accent_phrases: List[Dict[str, Any]]
    reference_text: str
    ref_boundary_method: str


@dataclass(frozen=True)
class SentenceCache:
    prefix: Path
    meta: SentenceMeta
    ref_y: np.ndarray
    ref_mfcc: np.ndarray
    ref_f0_times: np.ndarray
    ref_f0: np.ndarray

    @property
    def mora_count(self) -> int:
        return len(self.meta.moras)


def safe_cache_prefix(text: str, cache_dir: str | Path = "cache") -> Path:
    """Create a readable-ish cache prefix from Japanese text."""
    # Keep only a small ascii-safe hash-like suffix through Python hash is not stable,
    # so use sanitized kana/text length fallback. For explicit control, pass --out.
    import hashlib

    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    short = re.sub(r"\W+", "_", text, flags=re.UNICODE).strip("_")[:24] or "sentence"
    return Path(cache_dir) / f"{short}_{digest}"


def tts_reference(text: str, sr: int = 16000) -> np.ndarray:
    """Generate trimmed normalized reference speech using pyopenjtalk.tts."""
    import pyopenjtalk

    out = pyopenjtalk.tts(text)
    if isinstance(out, tuple):
        wav, tts_sr = out
    else:
        wav, tts_sr = out, 48000
    wav = np.asarray(wav, dtype=np.float64)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if np.max(np.abs(wav)) > 0:
        wav = wav / (np.max(np.abs(wav)) + 1e-9)
    if int(tts_sr) != sr:
        wav = librosa.resample(wav, orig_sr=int(tts_sr), target_sr=sr)
    wav, _ = trim_silence(wav, top_db=30.0)
    return wav.astype(np.float64)


def _mfcc(y: np.ndarray, sr: int, hop: int = 160, n_fft: int = 512, n_mfcc: int = 13) -> np.ndarray:
    m = librosa.feature.mfcc(y=y.astype(float), sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
    m = (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-8)
    return m.astype(np.float32)


def _equal_boundaries(duration_sec: float, mora_count: int) -> List[Tuple[float, float]]:
    if mora_count <= 0:
        return []
    step = duration_sec / mora_count
    return [(i * step, (i + 1) * step) for i in range(mora_count)]


def _split_long_segment(segment: str, max_chars: int = 22, min_left: int = 8) -> List[str]:
    segment = segment.strip()
    if not segment:
        return []

    chunks: List[str] = []
    strong_break_tokens = ("研究室の", "研究室", "大学", "学部", "学科")
    has_internal_strong_break = any(
        token in segment and not segment.endswith(token)
        for token in strong_break_tokens
    )
    if len(segment) <= max_chars and not has_internal_strong_break:
        return [segment]
    current = ""
    break_tokens = ("から", "ので", "けど", "ます", "です", "の", "は", "が", "を", "に", "で", "と")
    i = 0
    while i < len(segment):
        current += segment[i]
        remaining = len(segment) - i - 1
        should_break = (
            remaining >= min_left
            and len(current) >= 4
            and any(current.endswith(token) for token in strong_break_tokens)
        )
        if not should_break and len(current) >= min_left:
            should_break = remaining >= min_left and any(current.endswith(token) for token in break_tokens)
        if not should_break:
            should_break = len(current) >= max_chars
        if should_break:
            chunks.append(current)
            current = ""
        i += 1
    if current:
        chunks.append(current)
    return chunks


def _tts_chunks(text: str) -> List[str]:
    """
    Split long/free text into shorter TTS phrases.

    OpenJTalk TTS can produce an over-smooth contour when a long ASR transcript
    is synthesized as one accent phrase. Chunking is a pragmatic debug/product
    choice: it makes generated pseudo-references less misleading for free speech.
    """
    text = re.sub(r"\s+", "、", text.strip())
    if not text:
        return []
    raw_segments = [seg.strip() for seg in re.split(r"[、。！？!?]+", text) if seg.strip()]
    chunks: List[str] = []
    for segment in raw_segments:
        chunks.extend(_split_long_segment(segment))
    return chunks or [text]


def chunked_tts_reference(text: str, sr: int = 16000) -> Tuple[np.ndarray, List[Tuple[float, float]], str, str]:
    chunks = _tts_chunks(text)
    if len(chunks) <= 1:
        y = tts_reference(text, sr=sr)
        mora_count = len(build_text_info(text).moras)
        return y, _equal_boundaries(len(y) / sr, mora_count), text, "equal_mora"

    pieces: List[np.ndarray] = []
    boundaries: List[Tuple[float, float]] = []
    offset = 0.0
    gap = np.zeros(int(round(sr * 0.12)), dtype=np.float64)
    for idx, chunk in enumerate(chunks):
        y = tts_reference(chunk, sr=sr)
        chunk_moras = build_text_info(chunk).moras
        duration = len(y) / sr
        for start, end in _equal_boundaries(duration, len(chunk_moras)):
            boundaries.append((offset + start, offset + end))
        pieces.append(y)
        offset += duration
        if idx < len(chunks) - 1:
            pieces.append(gap)
            offset += len(gap) / sr

    return np.concatenate(pieces), boundaries, "、".join(chunks), "chunked_equal_mora"


def build_sentence_cache(
    text: str,
    out_prefix: str | Path,
    sr: int = 16000,
    save_reference_wav: bool = False,
) -> SentenceCache:
    """
    Build and save cache for a target sentence.

    Slow work done here:
    - OpenJTalk frontend
    - TTS reference waveform
    - reference MFCC
    - reference F0
    """
    import soundfile as sf
    from .text_frontend import run_frontend

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    text_info: TextInfo = build_text_info(text)
    frontend_raw = run_frontend(text)
    ref_y, ref_boundaries, reference_text, boundary_method = chunked_tts_reference(text, sr=sr)
    ref_duration = len(ref_y) / sr
    if len(ref_boundaries) != len(text_info.moras):
        ref_boundaries = _equal_boundaries(ref_duration, len(text_info.moras))
        boundary_method = "equal_mora_fallback"
    ref_mfcc = _mfcc(ref_y, sr=sr)
    f0_times, f0, _method = extract_f0(ref_y, sr)

    meta = SentenceMeta(
        text=text_info.text,
        kana=text_info.kana,
        moras=text_info.moras,
        target_pitch=text_info.target_pitch,
        pitch_target_source=text_info.pitch_target_source,
        is_question=text_info.is_question,
        sr=sr,
        ref_duration_sec=round(float(ref_duration), 6),
        ref_mora_boundaries=[(round(float(s), 6), round(float(e), 6)) for s, e in ref_boundaries],
        frontend_raw=frontend_raw,
        accent_phrases=text_info.accent_phrases,
        reference_text=reference_text,
        ref_boundary_method=boundary_method,
    )

    with (out_prefix.with_suffix(".json")).open("w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    np.savez_compressed(
        out_prefix.with_suffix(".npz"),
        ref_y=ref_y.astype(np.float32),
        ref_mfcc=ref_mfcc.astype(np.float32),
        ref_f0_times=f0_times.astype(np.float32),
        ref_f0=f0.astype(np.float32),
    )

    if save_reference_wav:
        sf.write(str(out_prefix.with_suffix(".ref.wav")), ref_y, sr)

    return load_sentence_cache(out_prefix)


def load_sentence_cache(prefix: str | Path) -> SentenceCache:
    prefix = Path(prefix)
    json_path = prefix if prefix.suffix == ".json" else prefix.with_suffix(".json")
    npz_path = prefix if prefix.suffix == ".npz" else prefix.with_suffix(".npz")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing cache json: {json_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing cache npz: {npz_path}")

    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    target_pitch = list(raw["target_pitch"])
    pitch_target_source = str(raw.get("pitch_target_source", "heuristic"))
    accent_phrases = list(raw.get("accent_phrases", []))
    if not accent_phrases or pitch_target_source == "heuristic":
        try:
            upgraded = build_text_info(str(raw["text"]))
            if len(upgraded.moras) == len(raw["moras"]):
                target_pitch = upgraded.target_pitch
                pitch_target_source = upgraded.pitch_target_source
                accent_phrases = upgraded.accent_phrases
        except Exception:
            pass
    meta = SentenceMeta(
        text=raw["text"],
        kana=raw["kana"],
        moras=list(raw["moras"]),
        target_pitch=target_pitch,
        pitch_target_source=pitch_target_source,
        is_question=bool(raw["is_question"]),
        sr=int(raw["sr"]),
        ref_duration_sec=float(raw["ref_duration_sec"]),
        ref_mora_boundaries=[(float(s), float(e)) for s, e in raw["ref_mora_boundaries"]],
        frontend_raw=list(raw.get("frontend_raw", [])),
        accent_phrases=accent_phrases,
        reference_text=str(raw.get("reference_text", raw["text"])),
        ref_boundary_method=str(raw.get("ref_boundary_method", "equal_mora")),
    )
    data = np.load(npz_path)
    # Prefix should be suffix-less for downstream display.
    clean_prefix = prefix.with_suffix("")
    return SentenceCache(
        prefix=clean_prefix,
        meta=meta,
        ref_y=np.asarray(data["ref_y"], dtype=np.float64),
        ref_mfcc=np.asarray(data["ref_mfcc"], dtype=np.float32),
        ref_f0_times=np.asarray(data["ref_f0_times"], dtype=np.float64),
        ref_f0=np.asarray(data["ref_f0"], dtype=np.float64),
    )


def cache_summary(cache: SentenceCache) -> str:
    return "\n".join([
        f"Text          : {cache.meta.text}",
        f"Kana          : {cache.meta.kana}",
        f"Mora          : {'・'.join(cache.meta.moras)}",
        f"Target pitch  : {' '.join(cache.meta.target_pitch)}",
        f"Pitch source  : {cache.meta.pitch_target_source}",
        f"Accent phrases: {len(cache.meta.accent_phrases)}",
        f"Reference text: {cache.meta.reference_text}",
        f"Boundary mode : {cache.meta.ref_boundary_method}",
        f"Ref duration  : {cache.meta.ref_duration_sec:.3f} sec",
        f"Cache prefix  : {cache.prefix}",
    ])
