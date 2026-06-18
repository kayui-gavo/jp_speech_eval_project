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
    select_prosody_reference_target,
)
from jp_speech_eval.sentence_cache import load_sentence_cache  # noqa: E402


def _load_demo_manifest(path: Path) -> Dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("target_id") or ""): row for row in rows if row.get("target_id")}


def _cache_prefixes(cache_dir: Path) -> Dict[str, Path]:
    return {path.stem: path.with_suffix("") for path in sorted(cache_dir.glob("*.json"))}


def _bool_text(value: bool) -> str:
    return "yes" if value else "no"


def _reason_from_flags(flags: Sequence[str], *, cache_exists: bool, manifest_reference_audio_exists: bool | None = None) -> str:
    reasons: List[str] = []
    if not cache_exists:
        reasons.append("missing_sentence_cache")
    if manifest_reference_audio_exists is False:
        reasons.append("manifest_reference_audio_missing")
    reasons.extend(str(flag) for flag in flags if flag)
    if not reasons:
        return ""
    return ";".join(dict.fromkeys(reasons))


def inventory_rows(*, cache_dir: Path, manifest_path: Path, min_f0_coverage: float) -> List[Dict[str, Any]]:
    manifest = _load_demo_manifest(manifest_path)
    prefixes = _cache_prefixes(cache_dir)
    target_ids = sorted(set(manifest) | set(prefixes))
    rows: List[Dict[str, Any]] = []
    for target_id in target_ids:
        manifest_row = manifest.get(target_id) or {}
        prefix = prefixes.get(target_id, cache_dir / target_id)
        cache_json = prefix.with_suffix(".json")
        cache_npz = prefix.with_suffix(".npz")
        ref_wav = prefix.with_suffix(".ref.wav")
        sidecar = load_prosody_reference_cache(prefix)
        manifest_audio = manifest_row.get("reference_audio")
        manifest_audio_path = (ROOT / str(manifest_audio)).resolve() if manifest_audio else None
        manifest_audio_exists = None if manifest_audio_path is None else manifest_audio_path.exists()

        row: Dict[str, Any] = {
            "target_id": target_id,
            "target_text": manifest_row.get("target_text") or "",
            "target_kana": manifest_row.get("kana") or "",
            "reference_audio_path": str(ref_wav) if ref_wav.exists() else str(manifest_audio_path) if manifest_audio_path else "",
            "manifest_reference_audio": manifest_audio or "",
            "manifest_verified_level": manifest_row.get("verified_level") or "",
            "manifest_pitch_target_source": manifest_row.get("pitch_target_source") or "",
            "manifest_reference_audio_exists": _bool_text(bool(manifest_audio_exists)) if manifest_audio_exists is not None else "",
            "has_cache_json": _bool_text(cache_json.exists()),
            "has_cache_npz": _bool_text(cache_npz.exists()),
            "has_ref_wav": _bool_text(ref_wav.exists()),
            "has_npz_f0": "no",
            "has_prosody_ref_sidecar": _bool_text(sidecar is not None),
            "sidecar_reliable": _bool_text(bool(sidecar and sidecar.get("reliable"))),
            "sidecar_path": str(sidecar.get("_cache_path")) if sidecar else "",
            "timing_source": "",
            "reference_source": "",
            "f0_coverage": "",
            "can_be_strong_pitch_reference": "no",
            "reason_if_not": "",
        }
        if not cache_json.exists() or not cache_npz.exists():
            row["reason_if_not"] = _reason_from_flags(
                [],
                cache_exists=False,
                manifest_reference_audio_exists=manifest_audio_exists,
            )
            rows.append(row)
            continue

        cache = load_sentence_cache(prefix)
        payload = build_prosody_reference_cache_payload(cache, min_f0_coverage=min_f0_coverage)
        fallback_values = [
            float(v) if v is not None else float("nan")
            for v in payload.get("reference_f0_mora_values") or []
        ]
        selection = select_prosody_reference_target(
            cache,
            fallback_reference_f0=fallback_values,
            text_pitch_target_source=cache.meta.pitch_target_source,
            min_f0_coverage=min_f0_coverage,
        )
        flags = list(selection.quality_flags or payload.get("quality_flags") or [])
        if manifest_row and str(manifest_row.get("verified_level") or "") == "human_checked":
            if not selection.pitch_target_reliability == "reliable":
                flags.append("manifest_claims_human_checked_but_cache_not_verified")
        row.update({
            "target_text": cache.meta.text,
            "target_kana": cache.meta.kana,
            "reference_source": cache.meta.reference_source,
            "timing_source": selection.mora_timing_source or cache.meta.ref_boundary_method,
            "has_npz_f0": _bool_text(bool(fallback_values)),
            "f0_coverage": selection.f0_coverage if selection.f0_coverage is not None else payload.get("f0_coverage"),
            "pitch_target_source": selection.pitch_target_source,
            "pitch_target_reliability": selection.pitch_target_reliability,
            "can_be_strong_pitch_reference": _bool_text(selection.pitch_target_reliability == "reliable"),
            "reason_if_not": "" if selection.pitch_target_reliability == "reliable" else _reason_from_flags(
                flags,
                cache_exists=True,
                manifest_reference_audio_exists=manifest_audio_exists,
            ),
        })
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _write_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    strong = [row for row in rows if row.get("can_be_strong_pitch_reference") == "yes"]
    weak = [row for row in rows if row.get("can_be_strong_pitch_reference") != "yes"]
    lines = [
        "# Fixed-reference prosody target inventory",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- targets: {len(rows)}",
        f"- strong_pitch_reference_targets: {len(strong)}",
        "",
        "## Inventory",
        "",
        "| target_id | reference_source | ref_wav | sidecar | sidecar_reliable | timing | f0_coverage | strong_pitch | reason_if_not |",
        "|---|---|---|---|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('target_id')} | {row.get('reference_source')} | {row.get('has_ref_wav')} | "
            f"{row.get('has_prosody_ref_sidecar')} | {row.get('sidecar_reliable')} | {row.get('timing_source')} | "
            f"{row.get('f0_coverage')} | {row.get('can_be_strong_pitch_reference')} | {row.get('reason_if_not')} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    if strong:
        lines.append("- At least one packaged target has a reliable reference-audio F0 target.")
    else:
        lines.append("- No packaged fixed-reference target currently has a verified reliable human/native reference F0 sidecar.")
    if weak:
        lines.append("- Targets without reliable sidecars should remain weak/practice/debug for pitch correctness.")
    lines.append("- Do not use `--verified-reference` on TTS/OpenJTalk pseudo references; verified means provenance has been checked as human/native/teacher audio.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory fixed-reference target prosody provenance.")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--manifest", default="data/demo_fixed_targets.json")
    parser.add_argument("--out-csv", default="results/calibration/fixed_reference_prosody_target_inventory.csv")
    parser.add_argument("--out-report", default="reports/fixed_reference_prosody_target_inventory.md")
    parser.add_argument("--min-f0-coverage", type=float, default=0.50)
    args = parser.parse_args()

    rows = inventory_rows(
        cache_dir=ROOT / args.cache_dir,
        manifest_path=ROOT / args.manifest,
        min_f0_coverage=args.min_f0_coverage,
    )
    _write_csv(ROOT / args.out_csv, rows)
    _write_report(ROOT / args.out_report, rows)
    print(f"wrote {ROOT / args.out_csv}")
    print(f"wrote {ROOT / args.out_report}")
    print(f"strong_pitch_reference_targets={sum(1 for r in rows if r.get('can_be_strong_pitch_reference') == 'yes')}/{len(rows)}")


if __name__ == "__main__":
    main()
