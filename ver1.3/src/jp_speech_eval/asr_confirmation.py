from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .asr import AsrTranscript, transcribe_language_aware
from .audio_features import load_audio
from .target_specs import default_scoring_policy, special_mora_metadata
from .text_frontend import build_text_info
from .transcript_sanity import check_asr_transcript_sanity
from .vad import trim_to_speech


_JA_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff々〆〤ー]")
_CONTENT_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff々〆〤A-Za-z0-9]")


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
    language_eligible: bool
    language_reason: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["asr_candidates"] = [item.to_dict() for item in self.asr_candidates]
        return data


def _japanese_script_ratio(text: str) -> float:
    normalized = str(text or "").strip()
    content = _CONTENT_CHAR_RE.findall(normalized)
    if not content:
        return 0.0
    return len(_JA_CHAR_RE.findall(normalized)) / len(content)


def _language_eligibility(transcript: AsrTranscript) -> tuple[bool, str]:
    """Conservative free-speech language gate.

    The confirmation flow must never turn clear English/Chinese/etc. speech
    into a Japanese pseudo-reference.  We therefore trust a confident
    non-Japanese Whisper label, and for uncertain labels require the *unforced*
    transcript itself to look substantially Japanese before continuing.
    """
    if not transcript.available:
        return False, "language_aware_asr_unavailable"
    text = str(transcript.text or "").strip()
    if not text:
        return False, "empty_transcript"
    language = str(transcript.language or "").strip().lower()
    probability = transcript.language_probability
    probability = float(probability) if probability is not None else None
    if language == "ja":
        return True, "detected_japanese"
    if language and language != "ja" and (probability is None or probability >= 0.55):
        return False, "detected_non_japanese"
    if _japanese_script_ratio(text) >= 0.55:
        return True, "uncertain_language_but_japanese_script"
    return False, "no_safe_japanese_language_evidence"


def build_asr_confirmation_prompt(
    wav_path: str | Path,
    *,
    sample_rate: int = 16000,
    asr_model: str = "small",
    asr_provider: str = "auto",
) -> AsrConfirmationPrompt:
    audio = load_audio(str(wav_path), sr=sample_rate)
    y_speech, _ = trim_to_speech(audio.y, audio.sr)
    transcript: AsrTranscript = transcribe_language_aware(
        y_speech,
        audio.sr,
        model_name=asr_model,
        provider=asr_provider,
    )
    language_eligible, language_reason = _language_eligibility(transcript)
    text = transcript.text if language_eligible and transcript.available and transcript.text else ""
    confidence = transcript.language_probability
    candidates = [AsrCandidate(id=1, text=text, confidence=confidence)] if text else []
    digest = hashlib.sha1(
        f"{Path(wav_path).resolve()}|{transcript.language}|{transcript.text}".encode("utf-8")
    ).hexdigest()[:16]
    if language_eligible:
        message = "猜你想说的是哪一句？如果不对，请手动修改。"
    else:
        message = "这段录音没有可靠识别为日语。请用日语重新录制。"
    return AsrConfirmationPrompt(
        mode="asr_confirm" if language_eligible else "asr_language_reject",
        session_id=digest,
        asr_candidates=candidates,
        editable_text=text,
        message=message,
        asr_raw=transcript.to_dict(),
        language_eligible=language_eligible,
        language_reason=language_reason,
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
