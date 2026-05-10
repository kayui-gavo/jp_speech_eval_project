from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from .acoustic_evaluator import evaluate_reference_free_acoustic
from .asr import transcribe_japanese
from .audio_features import load_audio
from .config import load_scoring_config
from .evaluator import evaluate_utterance
from .sentence_cache import build_sentence_cache, load_sentence_cache
from .transcript_assisted import evaluate_transcript_assisted_light
from .vad import trim_to_speech


def generated_cache_prefix(text: str, root: str | Path = "outputs/generated_refs") -> Path:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return Path(root) / f"asr_{digest}"


def evaluate_asr_pseudo_reference(
    wav_path: str | Path,
    *,
    base_cache_path: str | Path = "cache/ramen_kudasai",
    scoring_config_path: str | Path | None = None,
    generated_cache_dir: str | Path = "outputs/generated_refs",
) -> Dict[str, Any]:
    t_cache = load_sentence_cache(base_cache_path)
    audio = load_audio(str(wav_path), sr=t_cache.meta.sr)
    y_speech, _region = trim_to_speech(audio.y, audio.sr)
    config = load_scoring_config(scoring_config_path)
    content_cfg = config.get("content_match", {})
    transcript = transcribe_japanese(
        y_speech,
        audio.sr,
        model_name=str(content_cfg.get("asr_model", "small")),
        provider=str(content_cfg.get("asr_provider", "auto")),
    )
    if not transcript.available or not transcript.text:
        result = evaluate_reference_free_acoustic(wav_path, sample_rate=t_cache.meta.sr)
        result["details"]["mode"] = "asr_pseudo_reference_fallback_acoustic"
        result["details"]["asr"] = transcript.to_dict()
        return result

    generated_prefix = generated_cache_prefix(transcript.text, root=generated_cache_dir)
    build_sentence_cache(
        transcript.text,
        generated_prefix,
        sr=t_cache.meta.sr,
        save_reference_wav=True,
    )
    eval_result = evaluate_utterance(
        wav_path=wav_path,
        alignment_mode="cached_dtw",
        cache_path=generated_prefix,
        scoring_config_path=scoring_config_path,
        profile=False,
        use_content_match=False,
    )
    result = eval_result.to_dict()
    result["details"]["mode"] = "asr_pseudo_reference"
    result["details"]["asr"] = transcript.to_dict()
    result["details"]["reference_warning"] = "tts_generated_pseudo_reference_not_native_reference"
    return result


def evaluate_mode(
    mode: str,
    wav_path: str | Path,
    *,
    cache_path: str | Path | None = None,
    target_text: Optional[str] = None,
    transcript: Optional[str] = None,
    scoring_config_path: str | Path | None = None,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    mode = mode.strip()
    if mode in {"reference", "reference_based", "reference_fixed_sentence"}:
        if cache_path is None and not target_text:
            raise ValueError("reference mode requires cache_path or target_text")
        result = evaluate_utterance(
            text=target_text,
            wav_path=wav_path,
            alignment_mode="cached_dtw" if cache_path else "dtw",
            sample_rate=sample_rate,
            cache_path=cache_path,
            scoring_config_path=scoring_config_path,
            profile=False,
        ).to_dict()
        result.setdefault("details", {})["mode"] = "reference_based"
        return result
    if mode in {"acoustic", "reference_free_acoustic"}:
        return evaluate_reference_free_acoustic(wav_path, sample_rate=sample_rate)
    if mode in {"asr_pseudo_reference", "transcript_generated_reference"}:
        if cache_path is None:
            raise ValueError("asr_pseudo_reference mode requires a base cache for sample rate/config context")
        return evaluate_asr_pseudo_reference(
            wav_path,
            base_cache_path=cache_path,
            scoring_config_path=scoring_config_path,
        )
    if mode in {"transcript_assisted", "transcript_assisted_light"}:
        return evaluate_transcript_assisted_light(
            wav_path,
            transcript=transcript,
            sample_rate=sample_rate,
            scoring_config_path=scoring_config_path,
        )
    raise ValueError(f"Unknown evaluation mode: {mode}")
