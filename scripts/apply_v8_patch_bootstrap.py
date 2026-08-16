from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one match in {path}, got {count}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1) Direct free speech must not inherit pseudo-reference length rules.
p = Path("ver1.3/src/jp_speech_eval/transcript_sanity.py")
text = p.read_text(encoding="utf-8")
marker = "\n\ndef check_free_speech_transcript_sanity("
if marker not in text:
    text += '''


def check_free_speech_transcript_sanity(text: str) -> TranscriptSanityResult:
    """Gate transcript usability for direct free-speech scoring.

    Unlike :func:`check_asr_transcript_sanity`, this function does not impose
    pseudo-reference synthesis limits such as 3--80 content characters. Short
    conversational turns (for example ``はい`` or ``え？``) and longer natural
    answers remain score-eligible when the language gate says Japanese.

    This gate only rejects missing text, clearly non-Japanese script mixtures,
    and obvious repetition/noise-like ASR hallucinations. Utterance duration
    and per-dimension evidence sufficiency are handled later as reliability,
    not as language eligibility.
    """

    normalized = re.sub(r"\s+", "", str(text or "").strip())
    content_chars = _CONTENT_CHAR_RE.findall(normalized)
    ja_chars = _JA_CHAR_RE.findall(normalized)
    content_len = len(content_chars)
    ja_ratio = len(ja_chars) / max(content_len, 1)
    punct_ratio = len(_PUNCT_SPACE_RE.findall(normalized)) / max(len(normalized), 1)
    run_ratio = _longest_run_ratio(normalized)
    uniq_ratio = _unique_ratio(normalized)

    metrics: Dict[str, float | int | str] = {
        "char_count": len(normalized),
        "content_char_count": content_len,
        "ja_ratio": round(float(ja_ratio), 4),
        "punct_ratio": round(float(punct_ratio), 4),
        "longest_run_ratio": round(float(run_ratio), 4),
        "unique_content_ratio": round(float(uniq_ratio), 4),
        "purpose": "direct_free_speech_scoreability",
    }

    reason = "ok"
    score = 1.0
    ok = True
    if content_len <= 0:
        ok = False
        reason = "empty_or_no_content_transcript"
        score = 0.05
    elif ja_ratio < 0.55:
        ok = False
        reason = "not_enough_japanese_content"
        score = 0.20
    elif content_len >= 8 and run_ratio >= 0.45:
        ok = False
        reason = "repetitive_or_shouted_transcript"
        score = 0.20
    elif content_len >= 8 and uniq_ratio < 0.18:
        ok = False
        reason = "low_information_repetition"
        score = 0.25
    elif content_len >= 4 and punct_ratio > 0.60:
        ok = False
        reason = "mostly_punctuation_or_fillers"
        score = 0.20
    elif _contains_any(normalized.lower(), {"www", "ahaha", "哈哈", "呵呵"}):
        ok = False
        reason = "laughter_or_noise_like_transcript"
        score = 0.20

    return TranscriptSanityResult(
        ok=ok,
        score=round(float(score), 4),
        reason=reason,
        normalized_text=normalized,
        metrics=metrics,
    )
'''
    p.write_text(text, encoding="utf-8")

replace_once(
    "ver1.3/src/jp_speech_eval/transcript_assisted.py",
    "from .transcript_sanity import check_asr_transcript_sanity",
    "from .transcript_sanity import check_free_speech_transcript_sanity",
)
replace_once(
    "ver1.3/src/jp_speech_eval/transcript_assisted.py",
    'sanity = check_asr_transcript_sanity(transcript or "")',
    'sanity = check_free_speech_transcript_sanity(transcript or "")',
)

# 2) Expose reference-boundary provenance to downstream consumers.
replace_once(
    "ver1.3/src/jp_speech_eval/evaluator.py",
    '''                "requested_mode": requested_alignment_mode,
            },''',
    '''                "requested_mode": requested_alignment_mode,
                "reference_boundary_method": cache.meta.ref_boundary_method if cache else None,
                "reference_boundary_confidence": (
                    round(float(cache.meta.ref_boundary_confidence), 4) if cache else None
                ),
                "reference_boundary_tier": cache.meta.ref_boundary_tier if cache else None,
                "reference_boundary_source": cache.meta.ref_boundary_source if cache else None,
                "local_boundary_precision_role": "limits_target_local_evidence_not_global_performance",
            },''',
)

old_alignment_state = '''def _alignment_state(
    result: Mapping[str, Any],
    details: Mapping[str, Any],
) -> tuple[bool, bool, str]:
    """Resolve alignment availability from both legacy and current fields.

    Some evaluators historically wrote the fallback state only to the top-level
    ``alignment_mode`` while ``details.alignment`` still looked nominal. A C-end
    dimension must not treat equal-segmentation fallback as trustworthy local
    alignment merely because the nested legacy object is stale.
    """
    alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}
    mode = str(result.get("alignment_mode") or alignment.get("mode") or "")
    mode_lower = mode.lower()
    fallback = bool(alignment.get("used_equal_fallback")) or "fallback" in mode_lower
    available = bool(alignment.get("available", True)) and not fallback
    return available, fallback, mode
'''
new_alignment_state = '''def _alignment_state(
    result: Mapping[str, Any],
    details: Mapping[str, Any],
) -> tuple[bool, bool, str]:
    """Resolve whether target-local alignment is precise enough to consume.

    A good DTW path cannot manufacture precise mora boundaries when the
    reference boundaries it maps from were themselves equal-time placeholders.
    Reference provenance therefore limits *local* timing/F0 evidence while
    leaving broad/global speaking scores available.
    """
    alignment = details.get("alignment") if isinstance(details.get("alignment"), Mapping) else {}
    mode = str(result.get("alignment_mode") or alignment.get("mode") or "")
    mode_lower = mode.lower()
    path_fallback = bool(alignment.get("used_equal_fallback")) or "fallback" in mode_lower
    ref_tier = str(alignment.get("reference_boundary_tier") or "").strip().lower()
    ref_confidence = _number(alignment.get("reference_boundary_confidence"))
    reference_precision_limited = (
        ref_tier in {"equal_fallback", "unknown_alignment"}
        or (ref_confidence is not None and ref_confidence < 0.45)
    )
    fallback = bool(path_fallback or reference_precision_limited)
    available = bool(alignment.get("available", True)) and not fallback
    return available, fallback, mode
'''
replace_once(
    "ver1.3/src/jp_speech_eval/consumer_dimension_policy.py",
    old_alignment_state,
    new_alignment_state,
)
replace_once(
    "ver1.3/src/jp_speech_eval/consumer_dimension_policy.py",
    'alignment_note = " local alignment fell back, so target-local timing/F0 evidence was not used as if it were precise." if alignment_fallback else ""',
    'alignment_note = " local alignment or reference timing is approximate, so target-local timing/F0 evidence was not used as if it were precise." if alignment_fallback else ""',
)

# 3) Reliability gate: low-precision reference timing blocks local claims only.
replace_once(
    "ver1.3/src/jp_speech_eval/reliability_gate.py",
    '''    alignment_mode = str(result.get("alignment_mode") or alignment.get("mode") or "")
    is_fixed_reference = policy.fixed_reference
''',
    '''    alignment_mode = str(result.get("alignment_mode") or alignment.get("mode") or "")
    is_fixed_reference = policy.fixed_reference
    reference_boundary_tier = str(alignment.get("reference_boundary_tier") or "").strip().lower()
    reference_boundary_confidence_raw = alignment.get("reference_boundary_confidence")
    reference_boundary_confidence = (
        None
        if reference_boundary_confidence_raw is None
        else _clip01(reference_boundary_confidence_raw, default=0.0)
    )
    reference_boundary_precision_limited = bool(
        is_fixed_reference
        and (
            reference_boundary_tier in {"equal_fallback", "unknown_alignment"}
            or (reference_boundary_confidence is not None and reference_boundary_confidence < 0.45)
        )
    )
''',
)
replace_once(
    "ver1.3/src/jp_speech_eval/reliability_gate.py",
    '''    allow_pitch = policy.allow_pitch_feedback and is_fixed_reference and not policy.weak_reference and not policy.demo_only

    if not is_fixed_reference:
''',
    '''    allow_pitch = policy.allow_pitch_feedback and is_fixed_reference and not policy.weak_reference and not policy.demo_only

    if reference_boundary_precision_limited:
        practice = "needs_attention"
        allow_special = False
        allow_pitch = False
        allow_detail = False
        blocked.extend(["special_mora", "pitch", "pronunciation_detail"])
        reasons.append("reference_boundary_precision_low_broad_only")
        messages.append("参照音声の細かい拍位置が概算のため、今回は全体的な話し方を中心に表示します。")

    if not is_fixed_reference:
''',
)

# 4) Karaoke must use effective local precision, not DTW path confidence alone.
replace_once(
    "ver1.3/src/jp_speech_eval/app_core/karaoke_timeline.py",
    '''    confidence = _finite(alignment.get("confidence"))
    alignment_mode = str(result.get("alignment_mode") or alignment.get("mode") or "")
    approximate = bool(
''',
    '''    confidence = _finite(alignment.get("confidence"))
    reference_boundary_confidence = _finite(alignment.get("reference_boundary_confidence"))
    reference_boundary_tier = str(alignment.get("reference_boundary_tier") or "").strip().lower()
    reference_boundary_limited = bool(
        reference_boundary_tier in {"equal_fallback", "unknown_alignment"}
        or (reference_boundary_confidence is not None and reference_boundary_confidence < 0.45)
    )
    effective_local_confidence = confidence
    if reference_boundary_confidence is not None:
        effective_local_confidence = (
            reference_boundary_confidence
            if effective_local_confidence is None
            else min(effective_local_confidence, reference_boundary_confidence)
        )
    alignment_mode = str(result.get("alignment_mode") or alignment.get("mode") or "")
    approximate = bool(
''',
)
replace_once(
    "ver1.3/src/jp_speech_eval/app_core/karaoke_timeline.py",
    '''        or alignment_mode.endswith("fallback_equal")
        or confidence is None
        or confidence < 0.50
''',
    '''        or alignment_mode.endswith("fallback_equal")
        or reference_boundary_limited
        or effective_local_confidence is None
        or effective_local_confidence < 0.50
''',
)
replace_once(
    "ver1.3/src/jp_speech_eval/app_core/karaoke_timeline.py",
    '''            "alignment_confidence": None if confidence is None else round(_clip(confidence, 0.0, 1.0), 4),
            "approximate": approximate,
''',
    '''            "alignment_confidence": None if effective_local_confidence is None else round(_clip(effective_local_confidence, 0.0, 1.0), 4),
            "path_alignment_confidence": None if confidence is None else round(_clip(confidence, 0.0, 1.0), 4),
            "reference_boundary_confidence": None if reference_boundary_confidence is None else round(_clip(reference_boundary_confidence, 0.0, 1.0), 4),
            "reference_boundary_tier": reference_boundary_tier or None,
            "approximate": approximate,
''',
)

# 5) Special-mora learner feedback remains explicit opt-in.
replace_once(
    "ver1.3/src/jp_speech_eval/feedback_renderer.py",
    "    enable_user_facing_calibrated_special_mora: bool = True,",
    "    enable_user_facing_calibrated_special_mora: bool = False,",
)

# 6) Public demo should not compute hidden realtime debug rows.
replace_once(
    "ver1.3/scripts/debug_ui.py",
    '''                realtime = _realtime_rows(
                    wav_path=wav_path,
                    cache_prefix=self.server.cache_prefix,  # type: ignore[attr-defined]
                    config_path=self.server.config_path,  # type: ignore[attr-defined]
                    chunk_ms=float(self.server.chunk_ms),  # type: ignore[attr-defined]
                )
''',
    '''                realtime = [] if self.server.public_demo else _realtime_rows(  # type: ignore[attr-defined]
                    wav_path=wav_path,
                    cache_prefix=self.server.cache_prefix,  # type: ignore[attr-defined]
                    config_path=self.server.config_path,  # type: ignore[attr-defined]
                    chunk_ms=float(self.server.chunk_ms),  # type: ignore[attr-defined]
                )
''',
)

replace_once(
    ".github/workflows/baseline-evolution-light-tests.yml",
    "      - hf-space-karaoke-integration-v7\n",
    "      - hf-space-karaoke-integration-v7\n      - scoring-integrity-v8\n",
)

Path("ver1.3/tests/test_scoring_integrity_v8.py").write_text(r'''from __future__ import annotations

import inspect

import numpy as np
import soundfile as sf

from jp_speech_eval.app_core.karaoke_timeline import build_consumer_karaoke_timeline
from jp_speech_eval.consumer_dimension_policy import build_consumer_score_components
from jp_speech_eval.feedback_renderer import render_user_facing_result
from jp_speech_eval.reliability_gate import evaluate_reliability_gate
from jp_speech_eval.scoring_policy import policy_from_result
from jp_speech_eval.transcript_assisted import evaluate_transcript_assisted_light
from jp_speech_eval.transcript_sanity import check_free_speech_transcript_sanity


def _write_voice_like_wav(path, *, sr: int = 16000, duration: float = 0.8) -> None:
    t = np.arange(int(sr * duration), dtype=float) / sr
    y = 0.12 * np.sin(2.0 * np.pi * 190.0 * t)
    sf.write(path, y, sr, subtype="FLOAT")


def _fake_f0(y, sr):
    times = np.linspace(0.0, len(y) / sr, 24, endpoint=False)
    f0 = np.linspace(180.0, 220.0, 24)
    return times, f0, "test_f0"


def test_direct_free_speech_sanity_accepts_short_and_long_japanese() -> None:
    assert check_free_speech_transcript_sanity("はい").ok is True
    assert check_free_speech_transcript_sanity("え？").ok is True
    assert check_free_speech_transcript_sanity("今日はとてもいい天気ですね" * 8).ok is True
    assert check_free_speech_transcript_sanity("I like ramen very much").ok is False
    assert check_free_speech_transcript_sanity("あ" * 12).ok is False


def test_short_valid_japanese_turn_receives_score_in_direct_free_speech(tmp_path, monkeypatch) -> None:
    wav = tmp_path / "short_ja.wav"
    _write_voice_like_wav(wav)
    monkeypatch.setattr("jp_speech_eval.transcript_assisted.extract_f0", _fake_f0)
    raw = evaluate_transcript_assisted_light(wav, transcript="はい")
    assert raw["details"]["score_eligible"] is True
    assert raw["details"]["transcript_sanity"]["ok"] is True
    assert raw["total_score"] is not None
    assert raw["moras"]


def _low_reference_provenance_result():
    return {
        "alignment_mode": "cached_dtw",
        "pronunciation_score": 92,
        "prosody_score": 91,
        "fluency_score": 86,
        "tone_score": 88,
        "moras": ["ラ", "ー", "メ", "ン"],
        "mora_table": [
            {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2, "f0_hz": 180.0},
            {"mora": "ー", "start_sec": 0.2, "end_sec": 0.4, "f0_hz": 195.0},
            {"mora": "メ", "start_sec": 0.4, "end_sec": 0.6, "f0_hz": 188.0},
            {"mora": "ン", "start_sec": 0.6, "end_sec": 0.8, "f0_hz": 175.0},
        ],
        "endpointing": {"detected": True, "raw_duration": 0.9, "speech_start": 0.05, "speech_end": 0.85},
        "details": {
            "mode": "reference",
            "verified_level": "human_checked",
            "content_match": {"status": "pass", "score": 0.9, "kana_similarity": 0.95, "duration_ratio": 1.0},
            "alignment": {
                "available": True,
                "confidence": 0.92,
                "used_equal_fallback": False,
                "mode": "cached_dtw",
                "reference_boundary_method": "equal_mora",
                "reference_boundary_confidence": 0.15,
                "reference_boundary_tier": "equal_fallback",
            },
            "reliability": {
                "overall": 0.95,
                "endpointing": 1.0,
                "alignment": 0.92,
                "mora_evidence": 0.9,
                "f0_coverage": 0.9,
                "duration_ratio_to_reference": 1.0,
            },
            "recording_quality": {"score": 0.95},
            "fluency": {"rate_score": 88.0, "pause_score": 85.0, "speech_rate_mora_per_sec": 5.0},
            "prosody": {"contour_corr": 0.8, "contour_valid_mora_count": 4, "note": "ok"},
            "tone": {"pitch_range_log": 0.4, "pitch_score": 88.0},
            "reference_f0_by_mora": [175.0, 190.0, 185.0, 172.0],
        },
    }


def test_low_reference_boundary_provenance_cannot_become_local_public_evidence() -> None:
    raw = _low_reference_provenance_result()
    components = {item["key"]: item for item in build_consumer_score_components(raw, mode="reference")}
    assert components["mora_timing"]["evidence_tier"] == "alignment_fallback_broad_timing"
    assert components["mora_timing"]["confidence"] == "low"
    assert components["intonation"]["source_field"] != "prosody_score"
    assert components["intonation"]["confidence"] == "low"


def test_low_reference_boundary_provenance_blocks_local_feedback_but_not_whole_score() -> None:
    raw = _low_reference_provenance_result()
    policy = policy_from_result(raw, mode="reference")
    assert policy.allow_pitch_feedback is True
    gate = evaluate_reliability_gate(raw, policy)
    assert gate.practice_check_result == "needs_attention"
    assert gate.allow_pitch_feedback is False
    assert gate.allow_special_mora_feedback is False
    assert gate.allow_pronunciation_detail is False
    assert "reference_boundary_precision_low_broad_only" in gate.reasons


def test_low_reference_boundary_provenance_marks_karaoke_mora_sync_approximate() -> None:
    raw = _low_reference_provenance_result()
    timeline = build_consumer_karaoke_timeline(raw, {"score_dimensions": []})
    assert timeline["sync_mode"] == "mora_alignment_approximate"
    assert timeline["moras"]
    assert all(item["approximate"] for item in timeline["moras"])
    assert all(item["alignment_confidence"] == 0.15 for item in timeline["moras"])


def test_special_mora_user_feedback_is_opt_in_by_default() -> None:
    defaults = inspect.signature(render_user_facing_result).parameters
    assert defaults["enable_user_facing_calibrated_special_mora"].default is False
''', encoding="utf-8")
