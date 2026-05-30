from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import soundfile as sf

from .acoustic_evaluator import evaluate_reference_free_acoustic
from .asr_confirmation import build_confirmed_weak_target
from .asr import transcribe_japanese
from .audio_features import load_audio
from .config import load_scoring_config
from .evaluator import evaluate_utterance
from .kanade_reference import generate_voice_conditioned_reference
from .reference_store import build_reference_config, build_reference_hash
from .sentence_cache import build_sentence_cache, load_sentence_cache
from .transcript_sanity import check_asr_transcript_sanity
from .tts_adapter import canonical_provider_name
from .transcript_assisted import evaluate_transcript_assisted_light
from .vad import trim_to_speech


def generated_cache_prefix(
    text: str,
    root: str | Path = "outputs/generated_refs",
    *,
    tts_backend: str = "pyopenjtalk",
    tts_speaker: int | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    tts_speed: float | None = None,
    tts_style: str | None = None,
    tts_prompt: str | None = None,
    tts_language: str = "ja-JP",
    sample_rate: int = 16000,
) -> Path:
    """Return a config-aware prefix so different pseudo-references do not collide."""
    voice = tts_voice if tts_voice is not None else (None if tts_speaker is None else str(tts_speaker))
    config = build_reference_config(
        text=text,
        provider=canonical_provider_name(tts_backend),
        model=tts_model,
        voice=voice,
        speed=tts_speed,
        style=tts_style,
        prompt=tts_prompt,
        language=tts_language,
        sample_rate=sample_rate,
    )
    digest = build_reference_hash(config)[:12]
    return Path(root) / f"asr_{digest}"


def voice_conditioned_cache_prefix(text: str, speaker_wav_path: str | Path, root: str | Path) -> Path:
    speaker_path = Path(speaker_wav_path)
    digest_source = f"{text}|{speaker_path.resolve()}|{speaker_path.stat().st_mtime_ns}|{speaker_path.stat().st_size}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]
    return Path(root) / f"kanade_{digest}"


def _transcribe_for_dynamic_reference(
    wav_path: str | Path,
    *,
    base_cache_path: str | Path,
    scoring_config_path: str | Path | None,
) -> tuple[Any, Any]:
    base_cache = load_sentence_cache(base_cache_path)
    audio = load_audio(str(wav_path), sr=base_cache.meta.sr)
    y_speech, _region = trim_to_speech(audio.y, audio.sr)
    config = load_scoring_config(scoring_config_path)
    content_cfg = config.get("content_match", {})
    transcript = transcribe_japanese(
        y_speech,
        audio.sr,
        model_name=str(content_cfg.get("asr_model", "small")),
        provider=str(content_cfg.get("asr_provider", "auto")),
    )
    return base_cache, transcript


def _reject_dynamic_reference_result(
    wav_path: str | Path,
    *,
    sample_rate: int,
    mode: str,
    transcript: Any,
    sanity: Any,
) -> Dict[str, Any]:
    result = evaluate_reference_free_acoustic(wav_path, sample_rate=sample_rate)
    result["details"]["mode"] = mode
    result["details"]["asr"] = transcript.to_dict()
    result["details"]["transcript_sanity"] = sanity.to_dict()
    result["details"]["reference_warning"] = "pseudo_reference_rejected_before_tts"
    result["pronunciation_score"] = min(int(result.get("pronunciation_score", 0)), 35)
    result["prosody_score"] = min(int(result.get("prosody_score", 0)), 35)
    result["fluency_score"] = min(int(result.get("fluency_score", 0)), 45)
    result["tone_score"] = min(int(result.get("tone_score", 0)), 45)
    result["total_score"] = min(int(result.get("total_score", 0)), 35)
    result["feedback"] = [
        "我没能确认这是一句可评价的日语内容，所以没有生成参考音，也不会认真打分。",
        "请用完整的日语句子再说一次。"
    ]
    reliability = result.setdefault("details", {}).setdefault("reliability", {})
    reliability["overall"] = min(float(reliability.get("overall", 0.0) or 0.0), 0.25)
    reliability["level"] = "low"
    reliability["score_is_diagnostic"] = True
    warnings = list(reliability.get("warnings") or [])
    warnings.append(f"ASR transcript rejected before TTS: {sanity.reason}")
    reliability["warnings"] = warnings
    return result


def _build_dynamic_tts_cache(
    text: str,
    *,
    sr: int,
    generated_cache_dir: str | Path,
    tts_backend: str = "pyopenjtalk",
    tts_backend_url: str | None = None,
    tts_speaker: int | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    tts_speed: float | None = None,
    tts_style: str | None = None,
    tts_prompt: str | None = None,
    tts_language: str = "ja-JP",
) -> Path:
    generated_prefix = generated_cache_prefix(
        text,
        root=generated_cache_dir,
        tts_backend=tts_backend,
        tts_speaker=tts_speaker,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_speed=tts_speed,
        tts_style=tts_style,
        tts_prompt=tts_prompt,
        tts_language=tts_language,
        sample_rate=sr,
    )
    build_sentence_cache(
        text,
        generated_prefix,
        sr=sr,
        save_reference_wav=True,
        tts_backend=tts_backend,
        tts_backend_url=tts_backend_url,
        tts_speaker=tts_speaker,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_speed=tts_speed,
        tts_style=tts_style,
        tts_prompt=tts_prompt,
        tts_language=tts_language,
    )
    return generated_prefix


def evaluate_asr_pseudo_reference(
    wav_path: str | Path,
    *,
    base_cache_path: str | Path = "cache/ramen_kudasai",
    scoring_config_path: str | Path | None = None,
    generated_cache_dir: str | Path = "outputs/generated_refs",
    tts_backend: str = "pyopenjtalk",
    tts_backend_url: str | None = None,
    tts_speaker: int | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    tts_speed: float | None = None,
    tts_style: str | None = None,
    tts_prompt: str | None = None,
    tts_language: str = "ja-JP",
) -> Dict[str, Any]:
    raise ValueError("ASR raw transcript must be confirmed before generating a scoring reference. Use asr_confirmed_weak_reference.")


def evaluate_asr_confirmed_weak_reference(
    wav_path: str | Path,
    *,
    user_confirmed_text: str,
    base_cache_path: str | Path = "cache/ramen_kudasai",
    scoring_config_path: str | Path | None = None,
    generated_cache_dir: str | Path = "outputs/generated_refs",
    tts_backend: str = "pyopenjtalk",
    tts_backend_url: str | None = None,
    tts_speaker: int | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    tts_speed: float | None = None,
    tts_style: str | None = None,
    tts_prompt: str | None = None,
    tts_language: str = "ja-JP",
) -> Dict[str, Any]:
    base_cache = load_sentence_cache(base_cache_path)
    weak_target = build_confirmed_weak_target(user_confirmed_text)
    generated_prefix = _build_dynamic_tts_cache(
        weak_target["text"],
        sr=base_cache.meta.sr,
        generated_cache_dir=generated_cache_dir,
        tts_backend=tts_backend,
        tts_backend_url=tts_backend_url,
        tts_speaker=tts_speaker,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_speed=tts_speed,
        tts_style=tts_style,
        tts_prompt=tts_prompt,
        tts_language=tts_language,
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
    result["details"]["mode"] = "asr_confirmed_weak_reference"
    result["details"]["user_confirmed_text"] = weak_target["text"]
    result["details"]["weak_reference"] = True
    result["details"]["weak_target"] = weak_target
    result["details"]["reference_warning"] = "user_confirmed_tts_pseudo_reference_not_ground_truth"
    result["details"]["verified_level"] = "auto_pyopenjtalk"
    result["details"]["scoring_policy"] = weak_target["scoring_policy"]
    result["details"]["reference_source"] = "tts_pseudo_reference"
    return result


def evaluate_kanade_asr_confirmed_voice_reference(
    wav_path: str | Path,
    *,
    user_confirmed_text: str,
    base_cache_path: str | Path = "cache/ramen_kudasai",
    speaker_wav_path: str | Path | None = None,
    scoring_config_path: str | Path | None = None,
    generated_cache_dir: str | Path = "outputs/generated_refs",
    model_id: str = "frothywater/kanade-25hz-clean",
    tts_backend: str = "pyopenjtalk",
    tts_backend_url: str | None = None,
    tts_speaker: int | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    tts_speed: float | None = None,
    tts_style: str | None = None,
    tts_prompt: str | None = None,
    tts_language: str = "ja-JP",
) -> Dict[str, Any]:
    """Confirmed-ASR weak scoring plus Kanade playback reference.

    The confirmed text is used for scoring-reference generation. Kanade output
    is generated only for listening, and is explicitly excluded from
    pronunciation correctness.
    """
    base_cache = load_sentence_cache(base_cache_path)
    weak_target = build_confirmed_weak_target(user_confirmed_text)
    scoring_prefix = _build_dynamic_tts_cache(
        weak_target["text"],
        sr=base_cache.meta.sr,
        generated_cache_dir=generated_cache_dir,
        tts_backend=tts_backend,
        tts_backend_url=tts_backend_url,
        tts_speaker=tts_speaker,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_speed=tts_speed,
        tts_style=tts_style,
        tts_prompt=tts_prompt,
        tts_language=tts_language,
    )
    scoring_cache = load_sentence_cache(scoring_prefix)
    eval_result = evaluate_utterance(
        wav_path=wav_path,
        alignment_mode="cached_dtw",
        cache_path=scoring_prefix,
        scoring_config_path=scoring_config_path,
        profile=False,
        use_content_match=False,
    )
    result = eval_result.to_dict()

    speaker_path = Path(speaker_wav_path or wav_path)
    voice_prefix = voice_conditioned_cache_prefix(
        weak_target["text"],
        speaker_path,
        root=Path(generated_cache_dir) / "voice_playback",
    )
    voice_prefix.parent.mkdir(parents=True, exist_ok=True)
    voice_ref_y = generate_voice_conditioned_reference(
        scoring_cache.ref_y,
        target_sr=scoring_cache.meta.sr,
        speaker_wav_path=speaker_path,
        model_id=model_id,
    )
    generated_wav = voice_prefix.with_suffix(".voice.ref.wav")
    sf.write(str(generated_wav), voice_ref_y, scoring_cache.meta.sr)
    build_sentence_cache(
        weak_target["text"],
        voice_prefix,
        sr=scoring_cache.meta.sr,
        save_reference_wav=True,
        reference_wav_path=generated_wav,
        reference_source="kanade_voice_conditioned_playback_pseudo_reference",
    )

    result["details"]["mode"] = "kanade_asr_confirmed_voice_reference"
    result["details"]["user_confirmed_text"] = weak_target["text"]
    result["details"]["weak_reference"] = True
    result["details"]["weak_target"] = weak_target
    result["details"]["reference_warning"] = "confirmed_tts_pseudo_reference_used_for_scoring;kanade_reference_is_playback_only"
    result["details"]["playback_reference_source"] = "kanade_voice_conditioned_playback_pseudo_reference"
    result["details"]["voice_reference_cache_prefix"] = str(voice_prefix)
    result["details"]["speaker_reference_audio"] = str(speaker_path)
    result["details"]["kanade_model_id"] = model_id
    result["details"]["verified_level"] = "auto_pyopenjtalk"
    result["details"]["scoring_policy"] = weak_target["scoring_policy"]
    result["details"]["reference_source"] = "tts_pseudo_reference"
    result["details"]["demo_only"] = True
    result["details"]["exclude_from_pronunciation_score"] = True
    return result


def _evaluate_asr_pseudo_reference_legacy(
    wav_path: str | Path,
    *,
    base_cache_path: str | Path = "cache/ramen_kudasai",
    scoring_config_path: str | Path | None = None,
    generated_cache_dir: str | Path = "outputs/generated_refs",
    tts_backend: str = "pyopenjtalk",
    tts_backend_url: str | None = None,
    tts_speaker: int | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    tts_speed: float | None = None,
    tts_style: str | None = None,
    tts_prompt: str | None = None,
    tts_language: str = "ja-JP",
) -> Dict[str, Any]:
    t_cache, transcript = _transcribe_for_dynamic_reference(
        wav_path,
        base_cache_path=base_cache_path,
        scoring_config_path=scoring_config_path,
    )
    if not transcript.available or not transcript.text:
        result = evaluate_reference_free_acoustic(wav_path, sample_rate=t_cache.meta.sr)
        result["details"]["mode"] = "asr_pseudo_reference_fallback_acoustic"
        result["details"]["asr"] = transcript.to_dict()
        return result
    sanity = check_asr_transcript_sanity(transcript.text)
    if not sanity.ok:
        return _reject_dynamic_reference_result(
            wav_path,
            sample_rate=t_cache.meta.sr,
            mode="asr_pseudo_reference_rejected",
            transcript=transcript,
            sanity=sanity,
        )

    generated_prefix = _build_dynamic_tts_cache(
        transcript.text,
        sr=t_cache.meta.sr,
        generated_cache_dir=generated_cache_dir,
        tts_backend=tts_backend,
        tts_backend_url=tts_backend_url,
        tts_speaker=tts_speaker,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_speed=tts_speed,
        tts_style=tts_style,
        tts_prompt=tts_prompt,
        tts_language=tts_language,
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
    result["details"]["transcript_sanity"] = sanity.to_dict()
    result["details"]["reference_warning"] = "tts_generated_pseudo_reference_not_native_reference"
    return result


def evaluate_kanade_voice_reference(
    wav_path: str | Path,
    *,
    base_cache_path: str | Path = "cache/ramen_kudasai",
    speaker_wav_path: str | Path | None = None,
    scoring_config_path: str | Path | None = None,
    generated_cache_dir: str | Path = "outputs/generated_refs",
    model_id: str = "frothywater/kanade-25hz-clean",
    tts_backend: str = "pyopenjtalk",
    tts_backend_url: str | None = None,
    tts_speaker: int | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    tts_speed: float | None = None,
    tts_style: str | None = None,
    tts_prompt: str | None = None,
    tts_language: str = "ja-JP",
) -> Dict[str, Any]:
    base_cache = load_sentence_cache(base_cache_path)
    speaker_path = Path(speaker_wav_path or wav_path)
    generated_prefix = voice_conditioned_cache_prefix(
        base_cache.meta.text,
        speaker_path,
        root=generated_cache_dir,
    )
    generated_prefix.parent.mkdir(parents=True, exist_ok=True)

    voice_ref_y = generate_voice_conditioned_reference(
        base_cache.ref_y,
        target_sr=base_cache.meta.sr,
        speaker_wav_path=speaker_path,
        model_id=model_id,
    )
    generated_wav = generated_prefix.with_suffix(".voice.ref.wav")
    sf.write(str(generated_wav), voice_ref_y, base_cache.meta.sr)
    build_sentence_cache(
        base_cache.meta.text,
        generated_prefix,
        sr=base_cache.meta.sr,
        save_reference_wav=True,
        reference_wav_path=generated_wav,
        reference_source="kanade_voice_conditioned_pseudo_reference",
    )
    eval_result = evaluate_utterance(
        wav_path=wav_path,
        alignment_mode="cached_dtw",
        cache_path=generated_prefix,
        scoring_config_path=scoring_config_path,
        profile=False,
    )
    result = eval_result.to_dict()
    result["details"]["mode"] = "kanade_voice_reference"
    result["details"]["reference_warning"] = "kanade_voice_conditioned_pseudo_reference_not_native_reference"
    result["details"]["speaker_reference_audio"] = str(speaker_path)
    result["details"]["kanade_model_id"] = model_id
    result["details"]["demo_only"] = True
    result["details"]["exclude_from_pronunciation_score"] = True
    return result


def evaluate_kanade_asr_voice_reference(
    wav_path: str | Path,
    *,
    base_cache_path: str | Path = "cache/ramen_kudasai",
    speaker_wav_path: str | Path | None = None,
    scoring_config_path: str | Path | None = None,
    generated_cache_dir: str | Path = "outputs/generated_refs",
    model_id: str = "frothywater/kanade-25hz-clean",
    tts_backend: str = "pyopenjtalk",
    tts_backend_url: str | None = None,
    tts_speaker: int | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    tts_speed: float | None = None,
    tts_style: str | None = None,
    tts_prompt: str | None = None,
    tts_language: str = "ja-JP",
) -> Dict[str, Any]:
    base_cache, transcript = _transcribe_for_dynamic_reference(
        wav_path,
        base_cache_path=base_cache_path,
        scoring_config_path=scoring_config_path,
    )
    if not transcript.available or not transcript.text:
        result = evaluate_reference_free_acoustic(wav_path, sample_rate=base_cache.meta.sr)
        result["details"]["mode"] = "kanade_asr_voice_reference_fallback_acoustic"
        result["details"]["asr"] = transcript.to_dict()
        return result
    sanity = check_asr_transcript_sanity(transcript.text)
    if not sanity.ok:
        return _reject_dynamic_reference_result(
            wav_path,
            sample_rate=base_cache.meta.sr,
            mode="kanade_asr_voice_reference_rejected",
            transcript=transcript,
            sanity=sanity,
        )

    scoring_prefix = _build_dynamic_tts_cache(
        transcript.text,
        sr=base_cache.meta.sr,
        generated_cache_dir=generated_cache_dir,
        tts_backend=tts_backend,
        tts_backend_url=tts_backend_url,
        tts_speaker=tts_speaker,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_speed=tts_speed,
        tts_style=tts_style,
        tts_prompt=tts_prompt,
        tts_language=tts_language,
    )
    scoring_cache = load_sentence_cache(scoring_prefix)
    eval_result = evaluate_utterance(
        wav_path=wav_path,
        alignment_mode="cached_dtw",
        cache_path=scoring_prefix,
        scoring_config_path=scoring_config_path,
        profile=False,
        use_content_match=False,
    )
    result = eval_result.to_dict()

    speaker_path = Path(speaker_wav_path or wav_path)
    voice_prefix = voice_conditioned_cache_prefix(
        transcript.text,
        speaker_path,
        root=Path(generated_cache_dir) / "voice_playback",
    )
    voice_prefix.parent.mkdir(parents=True, exist_ok=True)
    voice_ref_y = generate_voice_conditioned_reference(
        scoring_cache.ref_y,
        target_sr=scoring_cache.meta.sr,
        speaker_wav_path=speaker_path,
        model_id=model_id,
    )
    generated_wav = voice_prefix.with_suffix(".voice.ref.wav")
    sf.write(str(generated_wav), voice_ref_y, scoring_cache.meta.sr)
    build_sentence_cache(
        transcript.text,
        voice_prefix,
        sr=scoring_cache.meta.sr,
        save_reference_wav=True,
        reference_wav_path=generated_wav,
        reference_source="kanade_voice_conditioned_playback_pseudo_reference",
    )

    result["details"]["mode"] = "kanade_asr_voice_reference"
    result["details"]["asr"] = transcript.to_dict()
    result["details"]["transcript_sanity"] = sanity.to_dict()
    result["details"]["reference_warning"] = "asr_tts_pseudo_reference_used_for_scoring;kanade_reference_is_playback_only"
    result["details"]["playback_reference_source"] = "kanade_voice_conditioned_playback_pseudo_reference"
    result["details"]["voice_reference_cache_prefix"] = str(voice_prefix)
    result["details"]["speaker_reference_audio"] = str(speaker_path)
    result["details"]["kanade_model_id"] = model_id
    result["details"]["weak_reference"] = True
    result["details"]["demo_only"] = True
    result["details"]["exclude_from_pronunciation_score"] = True
    return result


def evaluate_mode(
    mode: str,
    wav_path: str | Path,
    *,
    cache_path: str | Path | None = None,
    target_text: Optional[str] = None,
    transcript: Optional[str] = None,
    user_confirmed_text: Optional[str] = None,
    scoring_config_path: str | Path | None = None,
    sample_rate: int = 16000,
    tts_backend: str = "pyopenjtalk",
    tts_backend_url: str | None = None,
    tts_speaker: int | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    tts_speed: float | None = None,
    tts_style: str | None = None,
    tts_prompt: str | None = None,
    tts_language: str = "ja-JP",
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
            tts_backend=tts_backend,
            tts_backend_url=tts_backend_url,
            tts_speaker=tts_speaker,
            tts_model=tts_model,
            tts_voice=tts_voice,
            tts_speed=tts_speed,
            tts_style=tts_style,
            tts_prompt=tts_prompt,
            tts_language=tts_language,
        )
    if mode in {"asr_confirmed_weak_reference", "confirmed_asr_reference"}:
        if cache_path is None:
            raise ValueError("asr_confirmed_weak_reference mode requires a base cache for sample rate/config context")
        if not user_confirmed_text:
            raise ValueError("user_confirmed_text is required for asr_confirmed_weak_reference")
        return evaluate_asr_confirmed_weak_reference(
            wav_path,
            user_confirmed_text=user_confirmed_text,
            base_cache_path=cache_path,
            scoring_config_path=scoring_config_path,
            tts_backend=tts_backend,
            tts_backend_url=tts_backend_url,
            tts_speaker=tts_speaker,
            tts_model=tts_model,
            tts_voice=tts_voice,
            tts_speed=tts_speed,
            tts_style=tts_style,
            tts_prompt=tts_prompt,
            tts_language=tts_language,
        )
    if mode in {"kanade_voice_reference", "voice_conditioned_reference"}:
        if cache_path is None:
            raise ValueError("kanade_voice_reference mode requires a base cache for target text.")
        return evaluate_kanade_voice_reference(
            wav_path,
            base_cache_path=cache_path,
            scoring_config_path=scoring_config_path,
            tts_backend=tts_backend,
            tts_backend_url=tts_backend_url,
            tts_speaker=tts_speaker,
        )
    if mode in {"kanade_asr_voice_reference", "voice_conditioned_asr_reference"}:
        if cache_path is None:
            raise ValueError("kanade_asr_voice_reference mode requires a base cache for sample rate/config context.")
        return evaluate_kanade_asr_voice_reference(
            wav_path,
            base_cache_path=cache_path,
            scoring_config_path=scoring_config_path,
            tts_backend=tts_backend,
            tts_backend_url=tts_backend_url,
            tts_speaker=tts_speaker,
            tts_model=tts_model,
            tts_voice=tts_voice,
            tts_speed=tts_speed,
            tts_style=tts_style,
            tts_prompt=tts_prompt,
            tts_language=tts_language,
        )
    if mode in {"kanade_asr_confirmed_voice_reference", "confirmed_kanade_asr_reference"}:
        if cache_path is None:
            raise ValueError("kanade_asr_confirmed_voice_reference mode requires a base cache for sample rate/config context.")
        if not user_confirmed_text:
            raise ValueError("user_confirmed_text is required for kanade_asr_confirmed_voice_reference")
        return evaluate_kanade_asr_confirmed_voice_reference(
            wav_path,
            user_confirmed_text=user_confirmed_text,
            base_cache_path=cache_path,
            scoring_config_path=scoring_config_path,
            tts_backend=tts_backend,
            tts_backend_url=tts_backend_url,
            tts_speaker=tts_speaker,
            tts_model=tts_model,
            tts_voice=tts_voice,
            tts_speed=tts_speed,
            tts_style=tts_style,
            tts_prompt=tts_prompt,
            tts_language=tts_language,
        )
    if mode in {"transcript_assisted", "transcript_assisted_light"}:
        return evaluate_transcript_assisted_light(
            wav_path,
            transcript=transcript,
            sample_rate=sample_rate,
            scoring_config_path=scoring_config_path,
        )
    raise ValueError(f"Unknown evaluation mode: {mode}")
