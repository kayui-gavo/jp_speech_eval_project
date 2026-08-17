#!/usr/bin/env python3
"""Freeze and verify the direct free-speech confirmatory evaluation assets.

This module does not score audio and does not tune thresholds.  It records the
exact manifest, promotion protocols, and audio bytes that define a confirmatory
run, and fails closed on split leakage or later drift.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from validate_free_speech_sample_manifest import validate_manifest_file


FREEZE_SCHEMA = "free_speech_confirmatory_freeze_v11"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _canonical_manifest_sha(rows: Iterable[Mapping[str, str]]) -> str:
    normalized = [dict(sorted((str(k), str(v or "")) for k, v in row.items())) for row in rows]
    normalized.sort(key=lambda row: (row.get("sample_id", ""), row.get("audio_path", "")))
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _resolve_audio(audio_root: Path, audio_path: str) -> Path:
    candidate = (audio_root / audio_path).resolve()
    root = audio_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"audio path escapes audio_root: {audio_path}") from exc
    return candidate


def build_freeze(
    manifest_csv: str | Path,
    *,
    audio_root: str | Path,
    protocol_paths: Iterable[str | Path],
) -> dict[str, Any]:
    manifest_path = Path(manifest_csv)
    root = Path(audio_root)
    validation = validate_manifest_file(manifest_path)
    if not validation.get("ok"):
        raise ValueError("sample manifest failed validation: " + ";".join(validation.get("errors") or []))

    rows = _read_manifest(manifest_path)
    speaker_splits: dict[str, set[str]] = {}
    source_splits: dict[str, set[str]] = {}
    audio_inventory: list[dict[str, Any]] = []
    content_to_samples: dict[str, list[str]] = {}
    missing_audio: list[str] = []

    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        speaker_id = str(row.get("speaker_id") or "").strip()
        speaker_group = str(row.get("speaker_group") or "").strip()
        split = str(row.get("split") or "").strip()
        source_id = str(row.get("source_recording_id") or "").strip()
        audio_rel = str(row.get("audio_path") or "").strip()

        if speaker_group in {"learner", "native"} and speaker_id:
            speaker_splits.setdefault(speaker_id, set()).add(split)
        if source_id:
            source_splits.setdefault(source_id, set()).add(split)

        path = _resolve_audio(root, audio_rel)
        if not path.is_file():
            missing_audio.append(sample_id or audio_rel)
            continue
        digest = _sha256_file(path)
        content_to_samples.setdefault(digest, []).append(sample_id)
        audio_inventory.append(
            {
                "sample_id": sample_id,
                "audio_path": audio_rel,
                "split": split,
                "speaker_id": speaker_id,
                "source_recording_id": source_id,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )

    speaker_leaks = sorted(speaker for speaker, splits in speaker_splits.items() if len(splits) > 1)
    source_leaks = sorted(source for source, splits in source_splits.items() if len(splits) > 1)

    cross_split_duplicate_audio: list[dict[str, Any]] = []
    inventory_by_hash: dict[str, list[dict[str, Any]]] = {}
    for item in audio_inventory:
        inventory_by_hash.setdefault(str(item["sha256"]), []).append(item)
    for digest, items in inventory_by_hash.items():
        splits = {str(item["split"]) for item in items}
        sample_ids = sorted(str(item["sample_id"]) for item in items)
        if len(splits) > 1:
            cross_split_duplicate_audio.append({"sha256": digest, "sample_ids": sample_ids, "splits": sorted(splits)})

    protocol_inventory = []
    for protocol in protocol_paths:
        path = Path(protocol)
        if not path.is_file():
            raise FileNotFoundError(f"protocol file not found: {path}")
        protocol_inventory.append({"path": str(path), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size})

    audio_inventory.sort(key=lambda item: str(item["sample_id"]))
    protocol_inventory.sort(key=lambda item: str(item["path"]))
    problems = []
    if missing_audio:
        problems.append("missing_audio")
    if speaker_leaks:
        problems.append("speaker_split_leakage")
    if source_leaks:
        problems.append("source_recording_split_leakage")
    if cross_split_duplicate_audio:
        problems.append("duplicate_audio_bytes_across_splits")

    freeze_core = {
        "schema": FREEZE_SCHEMA,
        "manifest": {
            "path": str(manifest_path),
            "raw_sha256": _sha256_file(manifest_path),
            "canonical_rows_sha256": _canonical_manifest_sha(rows),
            "row_count": len(rows),
        },
        "protocols": protocol_inventory,
        "audio_root": str(root),
        "audio_inventory": audio_inventory,
        "integrity": {
            "missing_audio": sorted(missing_audio),
            "speaker_split_leakage": speaker_leaks,
            "source_recording_split_leakage": source_leaks,
            "duplicate_audio_bytes_across_splits": cross_split_duplicate_audio,
            "problems": problems,
        },
        "scientific_lock": {
            "held_set_is_confirmatory": True,
            "held_set_must_not_be_used_for_threshold_tuning": True,
            "promotion_protocol_must_not_change_after_freeze": True,
            "product_score_changed": False,
        },
    }
    fingerprint_payload = json.dumps(freeze_core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    freeze_core["freeze_fingerprint_sha256"] = _sha256_bytes(fingerprint_payload)
    freeze_core["freeze_ok"] = not problems
    return freeze_core


def verify_freeze(freeze_json: str | Path) -> dict[str, Any]:
    frozen_path = Path(freeze_json)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    manifest = frozen.get("manifest") or {}
    protocols = [item.get("path") for item in frozen.get("protocols") or []]
    current = build_freeze(
        manifest.get("path"),
        audio_root=frozen.get("audio_root"),
        protocol_paths=protocols,
    )
    expected = str(frozen.get("freeze_fingerprint_sha256") or "")
    actual = str(current.get("freeze_fingerprint_sha256") or "")
    return {
        "schema": "free_speech_confirmatory_freeze_verification_v11",
        "freeze_json": str(frozen_path),
        "expected_fingerprint_sha256": expected,
        "actual_fingerprint_sha256": actual,
        "freeze_ok": bool(current.get("freeze_ok")),
        "drift_detected": expected != actual,
        "verification_ok": bool(current.get("freeze_ok")) and expected == actual,
        "current_integrity": current.get("integrity"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    freeze_p = sub.add_parser("freeze")
    freeze_p.add_argument("manifest_csv")
    freeze_p.add_argument("--audio-root", required=True)
    freeze_p.add_argument("--protocol", action="append", required=True)
    freeze_p.add_argument("--out", required=True)

    verify_p = sub.add_parser("verify")
    verify_p.add_argument("freeze_json")

    args = parser.parse_args()
    if args.command == "freeze":
        report = build_freeze(args.manifest_csv, audio_root=args.audio_root, protocol_paths=args.protocol)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["freeze_ok"] else 2

    report = verify_freeze(args.freeze_json)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["verification_ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
