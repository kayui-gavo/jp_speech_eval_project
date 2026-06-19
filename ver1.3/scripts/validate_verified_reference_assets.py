#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.prosody_reference_cache import (  # noqa: E402
    build_prosody_reference_cache_payload,
    load_prosody_reference_cache,
    pseudo_reference_source,
    select_prosody_reference_target,
    trusted_reference_source,
)
from jp_speech_eval.sentence_cache import load_sentence_cache  # noqa: E402
from jp_speech_eval.text_frontend import split_mora  # noqa: E402


VERIFIED_STATUSES = {"verified", "approved", "human_checked", "native_checked", "teacher_checked"}
SUPPORTED_SAMPLE_RATES = {16000, 24000}


def load_demo_manifest(path: Path) -> Dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("target_id") or ""): row for row in rows if row.get("target_id")}


def _cache_prefixes(cache_dir: Path) -> Dict[str, Path]:
    return {path.stem: path.with_suffix("") for path in sorted(cache_dir.glob("*.json"))}


def _bool_text(value: bool) -> str:
    return "yes" if value else "no"


def _dedupe(items: Sequence[str]) -> str:
    return ";".join(dict.fromkeys(str(item) for item in items if item))


def _reference_audio_value(row: Mapping[str, Any]) -> str:
    return str(row.get("reference_audio_path") or row.get("reference_audio") or "")


def _resolve_reference_audio(row: Mapping[str, Any], *, root: Path) -> Path | None:
    value = _reference_audio_value(row)
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def _source_label(row: Mapping[str, Any], cache_source: str = "") -> str:
    return str(
        row.get("reference_source")
        or row.get("reference_provenance")
        or row.get("pitch_target_source")
        or cache_source
        or ""
    )


def _audio_metadata(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {
            "reference_audio_exists": False,
            "audio_readable": False,
            "sample_rate": "",
            "channels": "",
            "duration_sec": "",
            "error": "missing_reference_audio_path",
        }
    if not path.exists():
        return {
            "reference_audio_exists": False,
            "audio_readable": False,
            "sample_rate": "",
            "channels": "",
            "duration_sec": "",
            "error": "reference_audio_missing",
        }
    try:
        import soundfile as sf

        info = sf.info(str(path))
        duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
        return {
            "reference_audio_exists": True,
            "audio_readable": True,
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration_sec": round(duration, 4),
            "error": "",
        }
    except Exception as exc:
        return {
            "reference_audio_exists": True,
            "audio_readable": False,
            "sample_rate": "",
            "channels": "",
            "duration_sec": "",
            "error": f"audio_unreadable:{type(exc).__name__}",
        }


def _verification_status(row: Mapping[str, Any]) -> str:
    return str(row.get("verification_status") or row.get("verified_level") or "")


def _timing_is_weak(label: str) -> bool:
    lower = str(label or "").lower()
    return "fallback" in lower or "equal" in lower


def validation_rows(
    *,
    cache_dir: Path,
    manifest_path: Path,
    root: Path = ROOT,
    min_f0_coverage: float = 0.50,
) -> List[Dict[str, Any]]:
    manifest = load_demo_manifest(manifest_path)
    prefixes = _cache_prefixes(cache_dir)
    target_ids = sorted(set(manifest) | set(prefixes))
    rows: List[Dict[str, Any]] = []
    for target_id in target_ids:
        manifest_row = manifest.get(target_id) or {}
        prefix = prefixes.get(target_id, cache_dir / target_id)
        cache_json = prefix.with_suffix(".json")
        cache_npz = prefix.with_suffix(".npz")
        sidecar = load_prosody_reference_cache(prefix)
        audio_path = _resolve_reference_audio(manifest_row, root=root)
        audio = _audio_metadata(audio_path)

        blockers: List[str] = []
        warnings: List[str] = []
        if not _reference_audio_value(manifest_row):
            blockers.append("missing_reference_audio_path")
        if audio["error"]:
            blockers.append(str(audio["error"]).split(":", 1)[0])
        if audio.get("audio_readable"):
            sr = int(audio["sample_rate"])
            channels = int(audio["channels"])
            duration = float(audio["duration_sec"])
            if sr not in SUPPORTED_SAMPLE_RATES:
                warnings.append("sample_rate_not_repo_standard")
            if channels != 1:
                blockers.append("audio_not_mono")
            if duration < 0.35 or duration > 8.0:
                warnings.append("duration_outside_expected_fixed_reference_range")

        cache_source = ""
        cache_mora_count = ""
        reference_mora_count = ""
        timing_source = ""
        f0_coverage = ""
        sidecar_reliable = False
        selection_source = ""
        selection_reliability = ""
        if not cache_json.exists() or not cache_npz.exists():
            blockers.append("missing_sentence_cache")
        else:
            try:
                cache = load_sentence_cache(prefix)
                cache_source = cache.meta.reference_source
                cache_mora_count = len(cache.meta.moras)
                payload = build_prosody_reference_cache_payload(cache, min_f0_coverage=min_f0_coverage)
                fallback_values = [
                    float(value) if value is not None else float("nan")
                    for value in payload.get("reference_f0_mora_values") or []
                ]
                selection = select_prosody_reference_target(
                    cache,
                    fallback_reference_f0=fallback_values,
                    text_pitch_target_source=cache.meta.pitch_target_source,
                    min_f0_coverage=min_f0_coverage,
                )
                timing_source = str(selection.mora_timing_source or cache.meta.ref_boundary_method or "")
                f0_coverage = selection.f0_coverage if selection.f0_coverage is not None else payload.get("f0_coverage")
                selection_source = selection.pitch_target_source
                selection_reliability = selection.pitch_target_reliability
                if sidecar:
                    values = sidecar.get("reference_f0_smoothed_values") or sidecar.get("reference_f0_mora_values") or []
                    reference_mora_count = len(values)
                    sidecar_reliable = bool(sidecar.get("reliable"))
                else:
                    blockers.append("missing_prosody_sidecar")
                if selection.pitch_target_reliability != "reliable":
                    blockers.append("sidecar_unreliable")
                if _timing_is_weak(timing_source):
                    blockers.append("weak_or_approximate_mora_timing")
                if f0_coverage != "" and float(f0_coverage) < min_f0_coverage:
                    blockers.append("low_f0_coverage")
                if manifest_row.get("target_text") and str(manifest_row.get("target_text")) != cache.meta.text:
                    blockers.append("target_text_cache_mismatch")
                if manifest_row.get("kana") and str(manifest_row.get("kana")) != cache.meta.kana:
                    blockers.append("target_kana_cache_mismatch")
            except Exception as exc:
                blockers.append(f"cache_unreadable:{type(exc).__name__}")

        source = _source_label(manifest_row, cache_source)
        if pseudo_reference_source(source):
            blockers.append("tts_or_pseudo_reference")
        if not trusted_reference_source(source, verified_reference=False):
            blockers.append("untrusted_reference_source")
        status = _verification_status(manifest_row)
        if status.lower() not in VERIFIED_STATUSES:
            blockers.append("verification_status_not_verified")
        verified_by = str(manifest_row.get("verified_by") or "")
        if not verified_by:
            blockers.append("missing_verified_by")

        target_kana = str(manifest_row.get("kana") or "")
        target_mora_count = len(split_mora(target_kana)) if target_kana else cache_mora_count
        if reference_mora_count not in {"", target_mora_count}:
            blockers.append("reference_mora_count_mismatch")

        row = {
            "target_id": target_id,
            "target_text": str(manifest_row.get("target_text") or ""),
            "reference_audio_path": str(audio_path or ""),
            "reference_audio_exists": _bool_text(bool(audio["reference_audio_exists"])),
            "reference_source": source,
            "verification_status": status,
            "verified_by": verified_by,
            "is_tts_or_pseudo": _bool_text(pseudo_reference_source(source)),
            "audio_readable": _bool_text(bool(audio["audio_readable"])),
            "sample_rate": audio["sample_rate"],
            "channels": audio["channels"],
            "duration_sec": audio["duration_sec"],
            "target_mora_count": target_mora_count,
            "reference_mora_count": reference_mora_count,
            "f0_coverage": f0_coverage,
            "timing_source": timing_source,
            "has_prosody_sidecar": _bool_text(bool(sidecar)),
            "sidecar_reliable": _bool_text(sidecar_reliable),
            "pitch_target_source": selection_source,
            "pitch_target_reliability": selection_reliability,
            "strong_pitch_reference": _bool_text(len(blockers) == 0),
            "blocking_reasons": _dedupe(blockers),
            "warning_reasons": _dedupe(warnings),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    strong = [row for row in rows if row.get("strong_pitch_reference") == "yes"]
    lines = [
        "# Verified Reference Asset Validation",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- targets: {len(rows)}",
        f"- strong_pitch_reference_targets: {len(strong)}",
        "",
        "## Validation Table",
        "",
        "| target_id | audio | source | status | sidecar | timing | f0_coverage | strong | blocking_reasons |",
        "|---|---|---|---|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('target_id')} | {row.get('reference_audio_exists')}/{row.get('audio_readable')} | "
            f"{row.get('reference_source')} | {row.get('verification_status')} | "
            f"{row.get('has_prosody_sidecar')}/{row.get('sidecar_reliable')} | {row.get('timing_source')} | "
            f"{row.get('f0_coverage')} | {row.get('strong_pitch_reference')} | {row.get('blocking_reasons')} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    if strong:
        lines.append("- At least one packaged target has a complete verified reference-audio asset path.")
    else:
        lines.append("- No packaged target currently satisfies the strong pitch reference asset requirements.")
    lines.extend([
        "- TTS/OpenJTalk/pseudo references are blockers, even if a manifest claims verification.",
        "- Equal-mora or fallback timing is not accepted as a strong pitch reference alignment source.",
        "- Test-only JVS fixtures are intentionally outside the packaged demo manifest.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fixed-reference audio assets for strong pitch scoring.")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--manifest", default="data/demo_fixed_targets.json")
    parser.add_argument("--out-csv", default="results/calibration/verified_reference_asset_validation.csv")
    parser.add_argument("--out-report", default="reports/verified_reference_asset_validation.md")
    parser.add_argument("--min-f0-coverage", type=float, default=0.50)
    args = parser.parse_args()

    rows = validation_rows(
        cache_dir=ROOT / args.cache_dir,
        manifest_path=ROOT / args.manifest,
        root=ROOT,
        min_f0_coverage=args.min_f0_coverage,
    )
    write_csv(ROOT / args.out_csv, rows)
    write_report(ROOT / args.out_report, rows)
    print(f"wrote {ROOT / args.out_csv}")
    print(f"wrote {ROOT / args.out_report}")
    print(f"strong_pitch_reference_targets={sum(1 for r in rows if r.get('strong_pitch_reference') == 'yes')}/{len(rows)}")


if __name__ == "__main__":
    main()
