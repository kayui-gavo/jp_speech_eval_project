from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .asr import AsrTranscript, transcribe_japanese
from .audio_features import load_audio
from .target_specs import default_scoring_policy, special_mora_metadata
from .text_frontend import build_text_info
from .transcript_sanity import check_asr_transcript_sanity
from .vad import trim_to_speech


@dataclass(frozen=True)
class AsrCandidate:
    id: int
    text: str
    confidence: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AsrConfirmationPrompt:
    mode: str
    session_id: str
    asr_candidates: List[AsrCandidate]
    editable_text: str
    message: str
    asr_raw: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["asr_candidates"] = [item.to_dict() for item in self.asr_candidates]
        return data


def build_asr_confirmation_prompt(
    wav_path: str | Path,
    *,
    sample_rate: int = 16000,
    asr_model: str = "small",
    asr_provider: str = "auto",
) -> AsrConfirmationPrompt:
    audio = load_audio(str(wav_path), sr=sample_rate)
    y_speech, _ = trim_to_speech(audio.y, audio.sr)
    transcript: AsrTranscript = transcribe_japanese(
        y_speech,
        audio.sr,
        model_name=asr_model,
        provider=asr_provider,
    )
    text = transcript.text if transcript.available and transcript.text else ""
    candidates = [AsrCandidate(id=1, text=text, confidence=transcript.confidence)] if text else []
    digest = hashlib.sha1(f"{Path(wav_path).resolve()}|{text}".encode("utf-8")).hexdigest()[:16]
    return AsrConfirmationPrompt(
        mode="asr_confirm",
        session_id=digest,
        asr_candidates=candidates,
        editable_text=text,
        message="猜你想说的是哪一句？如果不对，请手动修改。",
        asr_raw=transcript.to_dict(),
    )


def build_confirmed_weak_target(user_confirmed_text: str) -> Dict[str, Any]:
    if not user_confirmed_text or not user_confirmed_text.strip():
        raise ValueError("user_confirmed_text is required before scoring.")
    sanity = check_asr_transcript_sanity(user_confirmed_text)
    if not sanity.ok:
        raise ValueError(f"confirmed text is not suitable for weak-reference scoring: {sanity.reason}")
    info = build_text_info(user_confirmed_text.strip())
    return {
        "target_source": "user_confirmed_asr",
        "weak_reference": True,
        "text": info.text,
        "kana": info.kana,
        "mora": info.moras,
        "target_pitch": info.target_pitch,
        "pitch_target_source": "auto_pyopenjtalk_weak",
        "special_mora": special_mora_metadata(info.moras),
        "reference_audio_source": "tts_pseudo_reference",
        "scoring_policy": default_scoring_policy("auto_pyopenjtalk", weak_reference=True),
    }
