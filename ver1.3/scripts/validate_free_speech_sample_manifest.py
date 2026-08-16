#!/usr/bin/env python3
"""Validate the free-speech criterion-evaluation sample manifest.

This validator enforces the metadata assumptions required by the v5 C-end
promotion gate.  It intentionally does not inspect audio content and it never
uses a gold transcript for scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


SCHEMA_VERSION = "free_speech_sample_manifest_v1"
REQUIRED_COLUMNS = (
    "sample_id",
    "audio_path",
    "speaker_id",
    "speaker_group",
    "l1",
    "task_mode",
    "prompt_id",
    "split",
    "expected_language",
    "channel_condition",
    "channel_pair_id",
    "source_recording_id",
    "context_type",
    "context_id",
    "context_text",
    "context_audio_path",
    "source_note",
)
ALLOWED_GROUPS = {"learner", "native", "negative_control"}
ALLOWED_TASKS = {"spontaneous", "controlled_dialogue"}
ALLOWED_SPLITS = {"development", "held"}
ALLOWED_LANGUAGES = {"ja", "en", "zh", "non_speech", "other"}
ALLOWED_CHANNELS = {"clean", "low_level", "moderate_noise", "codec_or_device_change", "other"}
ALLOWED_CONTEXT = {"none", "prompt", "preceding_turn"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def validate_manifest_rows(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Dict[str, Any]:
    missing = [name for name in REQUIRED_COLUMNS if name not in set(fieldnames)]
    if missing:
        return {
            "schema": SCHEMA_VERSION,
            "ok": False,
            "row_count": len(rows),
            "errors": ["missing_required_columns:" + ",".join(missing)],
            "warnings": [],
        }

    errors: list[str] = []
    warnings: list[str] = []
    seen_samples: set[str] = set()
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    channel_pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    counts = Counter()

    for index, row in enumerate(rows, start=2):
        prefix = f"row_{index}"
        sample_id = _text(row.get("sample_id"))
        audio_path = _text(row.get("audio_path"))
        speaker_id = _text(row.get("speaker_id"))
        group = _text(row.get("speaker_group"))
        task = _text(row.get("task_mode"))
        prompt_id = _text(row.get("prompt_id"))
        split = _text(row.get("split"))
        language = _text(row.get("expected_language"))
        channel = _text(row.get("channel_condition"))
        pair_id = _text(row.get("channel_pair_id"))
        source_recording_id = _text(row.get("source_recording_id"))
        context_type = _text(row.get("context_type"))
        context_text = _text(row.get("context_text"))
        context_audio = _text(row.get("context_audio_path"))

        for name, value in (
            ("sample_id", sample_id),
            ("audio_path", audio_path),
            ("speaker_id", speaker_id),
            ("prompt_id", prompt_id),
            ("source_recording_id", source_recording_id),
        ):
            if not value:
                errors.append(f"{prefix}:empty_{name}")

        if sample_id:
            if sample_id in seen_samples:
                errors.append(f"{prefix}:duplicate_sample_id:{sample_id}")
            seen_samples.add(sample_id)

        for value, allowed, name in (
            (group, ALLOWED_GROUPS, "speaker_group"),
            (task, ALLOWED_TASKS, "task_mode"),
            (split, ALLOWED_SPLITS, "split"),
            (language, ALLOWED_LANGUAGES, "expected_language"),
            (channel, ALLOWED_CHANNELS, "channel_condition"),
            (context_type, ALLOWED_CONTEXT, "context_type"),
        ):
            if value not in allowed:
                errors.append(f"{prefix}:invalid_{name}:{value}")

        if group in {"learner", "native"} and language != "ja":
            errors.append(f"{prefix}:learner_native_expected_language_must_be_ja")
        if group == "negative_control" and language == "ja":
            warnings.append(f"{prefix}:negative_control_expected_language_ja;document_edge_case")

        if speaker_id and split and group in {"learner", "native"}:
            speaker_splits[speaker_id].add(split)
        if pair_id:
            channel_pairs[pair_id].append(row)

        if context_type == "none" and (context_text or context_audio):
            errors.append(f"{prefix}:context_payload_present_when_context_type_none")
        if context_type != "none" and not (context_text or context_audio):
            warnings.append(f"{prefix}:context_type_without_payload")
        if context_text and "target_transcript=" in context_text.lower():
            errors.append(f"{prefix}:context_text_appears_to_embed_target_transcript")

        counts[f"group:{group}"] += 1
        counts[f"task:{task}"] += 1
        counts[f"split:{split}"] += 1
        counts[f"language:{language}"] += 1
        counts[f"channel:{channel}"] += 1

    for speaker_id, splits in sorted(speaker_splits.items()):
        if len(splits) > 1:
            errors.append(f"speaker_split_leakage:{speaker_id}:{'|'.join(sorted(splits))}")

    pair_reports: dict[str, Dict[str, Any]] = {}
    for pair_id, pair_rows in sorted(channel_pairs.items()):
        values = {
            key: {_text(row.get(key)) for row in pair_rows}
            for key in (
                "speaker_id",
                "prompt_id",
                "task_mode",
                "split",
                "expected_language",
                "source_recording_id",
            )
        }
        channels = {_text(row.get("channel_condition")) for row in pair_rows}
        pair_errors: list[str] = []
        if len(pair_rows) < 2:
            pair_errors.append("fewer_than_2_samples")
        for key, unique in values.items():
            if len(unique) != 1:
                pair_errors.append(f"conflicting_{key}")
        if len(channels) < 2:
            pair_errors.append("fewer_than_2_channel_conditions")
        if "clean" not in channels:
            pair_errors.append("missing_clean_anchor")
        if pair_errors:
            errors.extend(f"channel_pair:{pair_id}:{item}" for item in pair_errors)
        pair_reports[pair_id] = {
            "sample_count": len(pair_rows),
            "channels": sorted(channels),
            "source_recording_ids": sorted(values["source_recording_id"]),
            "same_source_recording": len(values["source_recording_id"]) == 1,
            "errors": pair_errors,
        }

    return {
        "schema": SCHEMA_VERSION,
        "ok": not errors,
        "row_count": len(rows),
        "speaker_count": len({_text(row.get("speaker_id")) for row in rows if _text(row.get("speaker_id"))}),
        "source_recording_count": len({_text(row.get("source_recording_id")) for row in rows if _text(row.get("source_recording_id"))}),
        "channel_pair_count": len(channel_pairs),
        "counts": dict(sorted(counts.items())),
        "channel_pairs": pair_reports,
        "errors": errors,
        "warnings": warnings,
    }


def validate_manifest_file(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return validate_manifest_rows(rows, list(reader.fieldnames or []))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_csv")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    report = validate_manifest_file(args.manifest_csv)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
