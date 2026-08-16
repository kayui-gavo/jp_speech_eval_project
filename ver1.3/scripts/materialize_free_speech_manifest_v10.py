#!/usr/bin/env python3
"""Materialize a validated free-speech manifest from v10 recording assignments.

The tool never fabricates missing recordings or participant metadata.  It scans
for the planned WAV files, joins optional private speaker metadata, emits only
existing recordings by default, and validates the resulting manifest with the
existing `free_speech_sample_manifest_v1` rules.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from validate_free_speech_sample_manifest import validate_manifest_file


MATERIALIZE_SCHEMA = "free_speech_manifest_materialization_v10"
MANIFEST_FIELDS = [
    "sample_id", "audio_path", "speaker_id", "speaker_group", "l1", "task_mode",
    "prompt_id", "split", "expected_language", "channel_condition", "channel_pair_id",
    "source_recording_id", "context_type", "context_id", "context_text",
    "context_audio_path", "source_note",
]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _text(value: Any) -> str:
    return str(value or "").strip()


def materialize(
    assignments_csv: str | Path,
    audio_root: str | Path,
    output_csv: str | Path,
    *,
    speaker_metadata_csv: str | Path | None = None,
    require_all: bool = False,
) -> dict[str, Any]:
    root = Path(audio_root)
    assignments = _read_csv(assignments_csv)
    metadata = {}
    if speaker_metadata_csv is not None:
        meta_rows = _read_csv(speaker_metadata_csv)
        metadata = {_text(row.get("speaker_id")): row for row in meta_rows if _text(row.get("speaker_id"))}

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    seen_samples: set[str] = set()
    for assignment in assignments:
        sample_id = _text(assignment.get("sample_id"))
        if not sample_id:
            raise ValueError("assignment row missing sample_id")
        if sample_id in seen_samples:
            raise ValueError(f"duplicate assignment sample_id: {sample_id}")
        seen_samples.add(sample_id)
        relpath = _text(assignment.get("audio_relpath"))
        if not relpath:
            raise ValueError(f"assignment {sample_id} missing audio_relpath")
        audio_path = root / relpath
        if not audio_path.is_file():
            missing.append(sample_id)
            continue

        speaker_id = _text(assignment.get("speaker_id"))
        meta = metadata.get(speaker_id, {})
        group = _text(assignment.get("speaker_group"))
        split = _text(assignment.get("split"))
        if meta:
            meta_group = _text(meta.get("speaker_group"))
            meta_split = _text(meta.get("split"))
            if meta_group and meta_group != group:
                raise ValueError(f"speaker group mismatch for {speaker_id}: {group} vs {meta_group}")
            if meta_split and meta_split != split:
                raise ValueError(f"speaker split mismatch for {speaker_id}: {split} vs {meta_split}")

        source_note_parts = [
            "v10_clean_collection",
            f"planned_duration={_text(assignment.get('planned_duration_bucket')) or 'unknown'}",
        ]
        if _text(meta.get("japanese_proficiency")):
            source_note_parts.append(f"proficiency={_text(meta.get('japanese_proficiency'))}")
        if _text(meta.get("recording_device")):
            source_note_parts.append(f"device={_text(meta.get('recording_device'))}")
        if _text(meta.get("consent_or_dataset_provenance")):
            source_note_parts.append(f"provenance={_text(meta.get('consent_or_dataset_provenance'))}")

        rows.append(
            {
                "sample_id": sample_id,
                "audio_path": relpath,
                "speaker_id": speaker_id,
                "speaker_group": group,
                "l1": _text(meta.get("l1")),
                "task_mode": _text(assignment.get("task_mode")),
                "prompt_id": _text(assignment.get("prompt_id")),
                "split": split,
                "expected_language": "ja",
                "channel_condition": "clean",
                "channel_pair_id": "",
                "source_recording_id": sample_id,
                "context_type": _text(assignment.get("context_type")) or "none",
                "context_id": f"ctx_{_text(assignment.get('prompt_id'))}",
                "context_text": _text(assignment.get("context_text")),
                "context_audio_path": "",
                "source_note": ";".join(source_note_parts),
            }
        )

    if require_all and missing:
        raise FileNotFoundError(f"missing {len(missing)} planned recordings; first: {missing[:5]}")
    if not rows:
        raise ValueError("no collected WAV files found for the supplied assignments/audio root")

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    validation = validate_manifest_file(output)
    if not validation.get("ok"):
        raise ValueError("materialized manifest failed validation: " + ";".join(validation.get("errors") or []))

    return {
        "schema": MATERIALIZE_SCHEMA,
        "assignment_count": len(assignments),
        "materialized_sample_count": len(rows),
        "missing_recording_count": len(missing),
        "missing_sample_ids": missing,
        "speaker_metadata_joined": speaker_metadata_csv is not None,
        "gold_transcript_added": False,
        "manifest_validation": validation,
        "output_csv": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("assignments_csv")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--speaker-metadata", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    report = materialize(
        args.assignments_csv,
        args.audio_root,
        args.out,
        speaker_metadata_csv=args.speaker_metadata,
        require_all=args.require_all,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
