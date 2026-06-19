from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .audio_features import median_f0_by_mora
from .sentence_cache import SentenceCache


PROSODY_REFERENCE_CACHE_VERSION = "prosody_reference_cache_v1"
RELIABLE_REFERENCE_SOURCES = (
    "human",
    "native",
    "teacher",
    "recorded",
    "verified",
    "jvs",
    "matched",
)
PSEUDO_REFERENCE_SOURCE_MARKERS = (
    "unverified",
    "pseudo",
    "synthetic",
    "tts",
    "pyopenjtalk",
    "openjtalk",
    "voicevox",
    "aivis",
)


@dataclass(frozen=True)
class ProsodyReferenceTarget:
    pitch_target_source: str
    pitch_target_reliability: str
    reference_f0_by_mora: Optional[List[float]]
    cache_path: Optional[str] = None
    cache_used: bool = False
    runtime_used: bool = False
    f0_coverage: Optional[float] = None
    quality_flags: List[str] = field(default_factory=list)
    mora_timing_source: Optional[str] = None
    reference_source: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean_prefix(prefix: str | Path) -> Path:
    path = Path(prefix)
    if path.suffix in {".json", ".npz", ".wav"}:
        path = path.with_suffix("")
    return path


def prosody_reference_cache_path(prefix: str | Path) -> Path:
    return _clean_prefix(prefix).with_suffix(".prosody_ref.json")


def _jsonable_f0(values: Sequence[float]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            out.append(None)
            continue
        out.append(number if math.isfinite(number) and number > 0 else None)
    return out


def _load_f0(values: Any) -> List[float]:
    out: List[float] = []
    for value in values or []:
        if value is None:
            out.append(float("nan"))
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        out.append(number if math.isfinite(number) and number > 0 else float("nan"))
    return out


def _f0_coverage(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.mean(np.isfinite(arr) & (arr > 0)))


def smooth_f0_by_mora(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr) & (arr > 0)
    if int(np.sum(valid)) < 2:
        return arr.tolist()
    x = np.arange(arr.size)
    filled = arr.copy()
    filled[~valid] = np.interp(x[~valid], x[valid], arr[valid])
    if arr.size >= 3:
        padded = np.pad(filled, (1, 1), mode="edge")
        filled = 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
    filled[~valid] = np.nan
    return filled.tolist()


def pseudo_reference_source(source: str | None) -> bool:
    label = str(source or "").lower()
    return any(marker in label for marker in PSEUDO_REFERENCE_SOURCE_MARKERS)


def trusted_reference_source(source: str | None, *, verified_reference: bool = False) -> bool:
    label = str(source or "").lower()
    if not label:
        return False
    if pseudo_reference_source(label):
        return False
    if verified_reference:
        return True
    return any(token in label for token in RELIABLE_REFERENCE_SOURCES)


def _timing_is_fallback(method: str | None) -> bool:
    label = str(method or "").lower()
    return "fallback" in label


def build_prosody_reference_cache_payload(
    cache: SentenceCache,
    *,
    reference_audio_path: str | Path | None = None,
    verified_reference: bool = False,
    min_f0_coverage: float = 0.50,
) -> Dict[str, Any]:
    values = median_f0_by_mora(cache.ref_f0_times, cache.ref_f0, cache.meta.ref_mora_boundaries)
    values = [float(v) if np.isfinite(v) and float(v) > 0 else float("nan") for v in values]
    smoothed = smooth_f0_by_mora(values)
    coverage = _f0_coverage(values)
    quality_flags: List[str] = []
    if len(values) != cache.mora_count:
        quality_flags.append("mora_count_mismatch")
    if coverage < min_f0_coverage:
        quality_flags.append("low_reference_f0_coverage")
    if _timing_is_fallback(cache.meta.ref_boundary_method):
        quality_flags.append("fallback_mora_timing")
    if "equal" in str(cache.meta.ref_boundary_method).lower():
        quality_flags.append("equal_mora_timing_approx")
    trusted_source = trusted_reference_source(cache.meta.reference_source, verified_reference=verified_reference)
    if verified_reference and pseudo_reference_source(cache.meta.reference_source):
        quality_flags.append("verified_reference_conflicts_with_pseudo_source")
    if verified_reference and not str(cache.meta.reference_source or "").strip():
        quality_flags.append("verified_reference_missing_source_provenance")
    if not trusted_source:
        quality_flags.append("untrusted_reference_source")

    reliable = (
        len(values) == cache.mora_count
        and coverage >= min_f0_coverage
        and "fallback_mora_timing" not in quality_flags
        and "untrusted_reference_source" not in quality_flags
    )
    reference_path = str(reference_audio_path) if reference_audio_path else None
    if reference_path is None:
        sibling_wav = cache.prefix.with_suffix(".ref.wav")
        if sibling_wav.exists():
            reference_path = str(sibling_wav)

    return {
        "cache_version": PROSODY_REFERENCE_CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_id": cache.meta.reference_id or cache.prefix.name,
        "target_text": cache.meta.text,
        "target_kana": cache.meta.kana,
        "target_mora_sequence": list(cache.meta.moras),
        "reference_audio_path": reference_path,
        "reference_source": cache.meta.reference_source,
        "reference_provenance_status": "trusted" if trusted_source else "untrusted",
        "reference_id": cache.meta.reference_id,
        "reference_f0_mora_values": _jsonable_f0(values),
        "reference_f0_smoothed_values": _jsonable_f0(smoothed),
        "voiced_mora_mask": [v is not None for v in _jsonable_f0(values)],
        "mora_timing_source": cache.meta.ref_boundary_method,
        "pitch_target_source": "reference_audio_f0_cache",
        "pitch_target_reliability": "reliable" if reliable else "unreliable",
        "f0_coverage": round(float(coverage), 4),
        "quality_flags": quality_flags,
        "verified_reference": bool(verified_reference),
        "reliable": bool(reliable),
    }


def write_prosody_reference_cache(
    cache: SentenceCache,
    *,
    out_path: str | Path | None = None,
    reference_audio_path: str | Path | None = None,
    verified_reference: bool = False,
    min_f0_coverage: float = 0.50,
) -> Path:
    payload = build_prosody_reference_cache_payload(
        cache,
        reference_audio_path=reference_audio_path,
        verified_reference=verified_reference,
        min_f0_coverage=min_f0_coverage,
    )
    path = Path(out_path) if out_path else prosody_reference_cache_path(cache.prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_prosody_reference_cache(prefix: str | Path) -> Optional[Dict[str, Any]]:
    path = prosody_reference_cache_path(prefix)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["_cache_path"] = str(path)
    return payload


def select_prosody_reference_target(
    cache: SentenceCache | None,
    *,
    fallback_reference_f0: Optional[Sequence[float]] = None,
    text_pitch_target_source: str = "openjtalk_accent_phrase_chain",
    min_f0_coverage: float = 0.50,
) -> ProsodyReferenceTarget:
    if cache is None:
        return ProsodyReferenceTarget(
            pitch_target_source=text_pitch_target_source,
            pitch_target_reliability="heuristic",
            reference_f0_by_mora=None,
            note="no_sentence_cache_available",
        )

    sidecar = load_prosody_reference_cache(cache.prefix)
    if sidecar:
        values = _load_f0(sidecar.get("reference_f0_smoothed_values") or sidecar.get("reference_f0_mora_values"))
        flags = [str(item) for item in sidecar.get("quality_flags") or []]
        coverage = _f0_coverage(values)
        sidecar_source = str(sidecar.get("reference_source") or cache.meta.reference_source or "")
        sidecar_trusted = trusted_reference_source(
            sidecar_source,
            verified_reference=bool(sidecar.get("verified_reference")),
        )
        if not sidecar_trusted and "untrusted_reference_source" not in flags:
            flags.append("untrusted_reference_source")
        if bool(sidecar.get("verified_reference")) and pseudo_reference_source(sidecar_source):
            if "verified_reference_conflicts_with_pseudo_source" not in flags:
                flags.append("verified_reference_conflicts_with_pseudo_source")
        reliable = (
            bool(sidecar.get("reliable"))
            and sidecar_trusted
            and len(values) == cache.mora_count
            and coverage >= min_f0_coverage
        )
        if reliable:
            return ProsodyReferenceTarget(
                pitch_target_source="reference_audio_f0_cache",
                pitch_target_reliability="reliable",
                reference_f0_by_mora=values,
                cache_path=str(sidecar.get("_cache_path") or ""),
                cache_used=True,
                f0_coverage=round(float(coverage), 4),
                quality_flags=flags,
                mora_timing_source=str(sidecar.get("mora_timing_source") or ""),
                reference_source=sidecar_source,
                note="reliable_reference_audio_f0_cache_used",
            )
        return ProsodyReferenceTarget(
            pitch_target_source="reference_audio_f0_cache_unreliable",
            pitch_target_reliability="unreliable",
            reference_f0_by_mora=list(fallback_reference_f0) if fallback_reference_f0 is not None else None,
            cache_path=str(sidecar.get("_cache_path") or ""),
            cache_used=False,
            f0_coverage=round(float(coverage), 4),
            quality_flags=flags or ["invalid_prosody_reference_cache"],
            mora_timing_source=str(sidecar.get("mora_timing_source") or ""),
            reference_source=sidecar_source,
            note="prosody_reference_cache_present_but_unreliable",
        )

    fallback = list(fallback_reference_f0) if fallback_reference_f0 is not None else None
    fallback_coverage = _f0_coverage(fallback or [])
    if (
        fallback
        and len(fallback) == cache.mora_count
        and fallback_coverage >= min_f0_coverage
        and trusted_reference_source(cache.meta.reference_source)
        and not _timing_is_fallback(cache.meta.ref_boundary_method)
    ):
        return ProsodyReferenceTarget(
            pitch_target_source="reference_audio_f0_runtime",
            pitch_target_reliability="reliable",
            reference_f0_by_mora=smooth_f0_by_mora(fallback),
            runtime_used=True,
            f0_coverage=round(float(fallback_coverage), 4),
            quality_flags=["runtime_reference_f0_no_sidecar_cache"],
            mora_timing_source=cache.meta.ref_boundary_method,
            reference_source=cache.meta.reference_source,
            note="trusted_reference_audio_runtime_f0_used",
        )

    if fallback:
        flags: List[str] = []
        if fallback_coverage < min_f0_coverage:
            flags.append("low_reference_f0_coverage")
        if not trusted_reference_source(cache.meta.reference_source):
            flags.append("untrusted_reference_source")
        if _timing_is_fallback(cache.meta.ref_boundary_method):
            flags.append("fallback_mora_timing")
        source = "tts_reference_weak" if "tts" in str(cache.meta.reference_source).lower() or cache.meta.reference_provider else "reference_audio_f0_runtime_weak"
        return ProsodyReferenceTarget(
            pitch_target_source=source,
            pitch_target_reliability="weak",
            reference_f0_by_mora=fallback,
            runtime_used=True,
            f0_coverage=round(float(fallback_coverage), 4),
            quality_flags=flags or ["unverified_reference_target"],
            mora_timing_source=cache.meta.ref_boundary_method,
            reference_source=cache.meta.reference_source,
            note="fallback_reference_contour_is_not_reliable_pitch_ground_truth",
        )

    return ProsodyReferenceTarget(
        pitch_target_source=text_pitch_target_source,
        pitch_target_reliability="heuristic",
        reference_f0_by_mora=None,
        quality_flags=["no_reference_f0_available"],
        mora_timing_source=cache.meta.ref_boundary_method,
        reference_source=cache.meta.reference_source,
        note="fallback_to_text_pitch_target",
    )
