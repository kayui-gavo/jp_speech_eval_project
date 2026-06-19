#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.audio_features import extract_f0, load_audio  # noqa: E402
from jp_speech_eval.prosody_reference_cache import (  # noqa: E402
    pseudo_reference_source,
    trusted_reference_source,
    write_prosody_reference_cache,
)
from jp_speech_eval.sentence_cache import build_sentence_cache  # noqa: E402


def _load_manifest(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing manifest: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("demo fixed target manifest must be a list")
    return [dict(row) for row in rows]


def _find_target(rows: list[Mapping[str, Any]], target_id: str) -> Mapping[str, Any] | None:
    for row in rows:
        if str(row.get("target_id") or "") == target_id:
            return row
    return None


def _manifest_audio_path(row: Mapping[str, Any], *, root: Path) -> Path | None:
    raw = row.get("reference_audio_path") or row.get("reference_audio")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def _has_existing_verified_reference(row: Mapping[str, Any], *, root: Path) -> bool:
    source = str(row.get("reference_source") or row.get("reference_provenance") or row.get("pitch_target_source") or "")
    status = str(row.get("verification_status") or row.get("verified_level") or "").lower()
    audio = _manifest_audio_path(row, root=root)
    return (
        audio is not None
        and audio.exists()
        and not pseudo_reference_source(source)
        and (
            status in {"verified", "human_checked", "native_checked", "teacher_checked"}
            or trusted_reference_source(source, verified_reference=False)
        )
    )


def _audio_probe(path: Path, *, sample_rate: int = 16000) -> Dict[str, Any]:
    import soundfile as sf

    info = sf.info(str(path))
    audio = load_audio(str(path), sr=sample_rate)
    times, f0, method = extract_f0(audio.y, audio.sr)
    voiced = int((f0 > 0).sum())
    coverage = float(voiced / max(len(f0), 1))
    return {
        "readable": True,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_sec": round(float(info.frames) / float(info.samplerate), 4) if info.samplerate else 0.0,
        "analysis_sample_rate": audio.sr,
        "f0_frame_count": int(len(times)),
        "f0_voiced_frame_count": voiced,
        "f0_frame_coverage": round(coverage, 4),
        "f0_method": method,
    }


def plan_import(
    *,
    target_id: str,
    reference_audio_path: Path,
    reference_source: str,
    speaker_id: str,
    take_id: str,
    verified_by: str,
    manifest_path: Path,
    cache_dir: Path,
    root: Path = ROOT,
    replace: bool = False,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    rows = _load_manifest(manifest_path)
    target = _find_target(rows, target_id)
    if target is None:
        blockers.append("unknown_target_id")
        target = {}

    if not reference_audio_path.exists():
        blockers.append("reference_audio_missing")
        audio: Dict[str, Any] = {"readable": False}
    else:
        try:
            audio = _audio_probe(reference_audio_path, sample_rate=sample_rate)
        except Exception as exc:
            blockers.append(f"audio_unreadable:{type(exc).__name__}")
            audio = {"readable": False}
    if audio.get("readable"):
        if int(audio.get("channels") or 0) != 1:
            blockers.append("audio_not_mono")
        if float(audio.get("duration_sec") or 0.0) < 0.35:
            warnings.append("audio_very_short")
        if float(audio.get("f0_frame_coverage") or 0.0) < 0.35:
            blockers.append("low_audio_f0_coverage")

    if not reference_source.strip():
        blockers.append("missing_reference_source")
    if pseudo_reference_source(reference_source):
        blockers.append("tts_or_pseudo_reference_source")
    if not trusted_reference_source(reference_source, verified_reference=True):
        blockers.append("untrusted_reference_source")
    if not speaker_id.strip():
        blockers.append("missing_speaker_id")
    if not take_id.strip():
        blockers.append("missing_take_id")
    if not verified_by.strip():
        blockers.append("missing_verified_by")
    if _has_existing_verified_reference(target, root=root) and not replace:
        blockers.append("existing_verified_reference_requires_replace")

    warnings.append("external_audio_import_uses_equal_mora_timing_until_manual_or_lab_timing_is_added")
    output_prefix = cache_dir / target_id
    reference_out = output_prefix.with_suffix(".ref.wav")
    return {
        "target_id": target_id,
        "target_text": target.get("target_text") or target.get("text") or "",
        "target_kana": target.get("kana") or "",
        "reference_audio_path": str(reference_audio_path),
        "reference_source": reference_source,
        "speaker_id": speaker_id,
        "take_id": take_id,
        "verified_by": verified_by,
        "replace": bool(replace),
        "can_commit": len(blockers) == 0,
        "blocking_reasons": blockers,
        "warning_reasons": warnings,
        "audio": audio,
        "would_write": {
            "cache_prefix": str(output_prefix),
            "reference_audio": str(reference_out),
            "sentence_cache_json": str(output_prefix.with_suffix(".json")),
            "sentence_cache_npz": str(output_prefix.with_suffix(".npz")),
            "prosody_sidecar": str(output_prefix.with_suffix(".prosody_ref.json")),
            "manifest": str(manifest_path),
        },
    }


def commit_import(plan: Mapping[str, Any], *, manifest_path: Path, cache_dir: Path, root: Path = ROOT, notes: str = "") -> Dict[str, Any]:
    if not plan.get("can_commit"):
        raise ValueError(f"cannot import reference audio: {plan.get('blocking_reasons')}")
    rows = _load_manifest(manifest_path)
    target_id = str(plan["target_id"])
    target_index = next(i for i, row in enumerate(rows) if str(row.get("target_id") or "") == target_id)
    row = dict(rows[target_index])
    prefix = cache_dir / target_id
    cache = build_sentence_cache(
        str(row["target_text"]),
        prefix,
        sr=16000,
        save_reference_wav=True,
        reference_wav_path=str(plan["reference_audio_path"]),
        reference_source=str(plan["reference_source"]),
        reference_id=target_id,
    )
    sidecar = write_prosody_reference_cache(
        cache,
        reference_audio_path=prefix.with_suffix(".ref.wav"),
        verified_reference=True,
    )
    row.update({
        "reference_audio": str(prefix.with_suffix(".ref.wav").relative_to(root)),
        "reference_audio_path": str(prefix.with_suffix(".ref.wav").relative_to(root)),
        "reference_source": str(plan["reference_source"]),
        "reference_speaker_id": str(plan["speaker_id"]),
        "reference_take_id": str(plan["take_id"]),
        "verified_by": str(plan["verified_by"]),
        "verification_status": "verified",
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "timing_source": cache.meta.ref_boundary_method,
        "license_or_usage_note": row.get("license_or_usage_note") or "local verified reference; usage must be confirmed before redistribution",
        "reference_import_note": notes,
    })
    rows[target_index] = row
    manifest_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "target_id": target_id,
        "cache_prefix": str(prefix),
        "prosody_sidecar": str(sidecar),
        "timing_source": cache.meta.ref_boundary_method,
        "note": "Imported reference audio. Equal-mora timing is not a strong pitch reference until manual/lab timing is added.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run or import verified human/native/teacher reference audio.")
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--reference-audio-path", required=True)
    parser.add_argument("--reference-source", required=True, help="e.g. native_teacher_recorded_reference")
    parser.add_argument("--speaker-id", required=True)
    parser.add_argument("--take-id", required=True)
    parser.add_argument("--verified-by", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--manifest", default="data/demo_fixed_targets.json")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--commit", action="store_true", help="Write cache/manifest. Default is dry-run only.")
    args = parser.parse_args()

    plan = plan_import(
        target_id=args.target_id,
        reference_audio_path=Path(args.reference_audio_path),
        reference_source=args.reference_source,
        speaker_id=args.speaker_id,
        take_id=args.take_id,
        verified_by=args.verified_by,
        manifest_path=ROOT / args.manifest,
        cache_dir=ROOT / args.cache_dir,
        root=ROOT,
        replace=args.replace,
    )
    if not args.commit:
        print(json.dumps({"dry_run": True, "plan": plan}, ensure_ascii=False, indent=2))
        return
    result = commit_import(plan, manifest_path=ROOT / args.manifest, cache_dir=ROOT / args.cache_dir, root=ROOT, notes=args.notes)
    print(json.dumps({"dry_run": False, "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
