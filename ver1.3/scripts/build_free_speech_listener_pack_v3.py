#!/usr/bin/env python3
"""Build a blinded listener pack for consumer-criterion v3.

The public listener CSV never contains speaker group, L1, channel condition,
source path, or model output.  Source paths are kept in a separate private asset
map.  Context is shown only for the contextual-intonation presentation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from validate_free_speech_sample_manifest import validate_manifest_file


PACK_SCHEMA = "consumer_listener_pack_v3"
ISOLATED_CONSTRUCTS = (
    "clarity_comprehensibility",
    "fluency",
    "rhythm_naturalness",
    "intonation_utterance_naturalness",
)
CONTEXTUAL_CONSTRUCTS = ("intonation_contextual_appropriateness",)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _read_manifest(path: str | Path) -> list[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _asset_id(sample_id: str) -> str:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"audio_{digest}"


def _context_asset_id(sample_id: str) -> str:
    digest = hashlib.sha256(("context:" + sample_id).encode("utf-8")).hexdigest()[:12]
    return f"context_{digest}"


def _assigned_raters(raters: list[str], sample_index: int, count: int, *, offset: int = 0) -> list[str]:
    if not raters:
        raise ValueError("at least one rater id is required")
    take = min(max(1, int(count)), len(raters))
    start = (sample_index + offset) % len(raters)
    return [raters[(start + i) % len(raters)] for i in range(take)]


def build_listener_pack(
    manifest_csv: str | Path,
    *,
    rater_ids: Iterable[str],
    ratings_per_presentation: int = 5,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, Any]]:
    validation = validate_manifest_file(manifest_csv)
    if not validation.get("ok"):
        raise ValueError("sample manifest failed validation: " + ";".join(validation.get("errors") or []))
    raters = sorted({_text(rater) for rater in rater_ids if _text(rater)})
    if not raters:
        raise ValueError("no valid rater ids")

    rows = _read_manifest(manifest_csv)
    listener_rows: list[Dict[str, Any]] = []
    asset_rows: list[Dict[str, Any]] = []
    japanese_sample_count = 0
    contextual_sample_count = 0

    for sample_index, row in enumerate(rows):
        if _text(row.get("expected_language")) != "ja" or _text(row.get("speaker_group")) == "negative_control":
            continue
        japanese_sample_count += 1
        sample_id = _text(row.get("sample_id"))
        task_mode = _text(row.get("task_mode"))
        audio_asset = _asset_id(sample_id)
        context_type = _text(row.get("context_type"))
        context_text = _text(row.get("context_text"))
        context_audio_path = _text(row.get("context_audio_path"))
        has_context = context_type != "none" and bool(context_text or context_audio_path)
        context_asset = _context_asset_id(sample_id) if context_audio_path else ""

        asset_rows.append(
            {
                "sample_id": sample_id,
                "audio_asset_id": audio_asset,
                "source_audio_path": _text(row.get("audio_path")),
                "context_audio_asset_id": context_asset,
                "source_context_audio_path": context_audio_path,
                "private_only": "true",
            }
        )

        for rater_id in _assigned_raters(raters, sample_index, ratings_per_presentation):
            listener_rows.append(
                {
                    "rater_id": rater_id,
                    "sample_id": sample_id,
                    "presentation_id": f"{audio_asset}__isolated__{rater_id}",
                    "presentation_variant": "isolated",
                    "task_mode": task_mode,
                    "audio_asset_id": audio_asset,
                    "context_type": "none",
                    "context_id": "",
                    "context_text": "",
                    "context_audio_asset_id": "",
                    "assigned_constructs": "|".join(ISOLATED_CONSTRUCTS),
                    "analyzable_yes_no": "",
                    "clarity_comprehensibility_1to7": "",
                    "fluency_1to7": "",
                    "rhythm_naturalness_1to7": "",
                    "intonation_utterance_naturalness_1to7": "",
                    "intonation_contextual_appropriateness_1to7": "",
                    "context_presented_yes_no": "no",
                    "timestamp": "",
                    "pack_schema": PACK_SCHEMA,
                }
            )

        if has_context:
            contextual_sample_count += 1
            # Rotate the rater start so, when the pool is larger than the requested
            # panel, contextual ratings are not mechanically assigned to the exact
            # same subset as the isolated presentation.
            for rater_id in _assigned_raters(raters, sample_index, ratings_per_presentation, offset=1):
                listener_rows.append(
                    {
                        "rater_id": rater_id,
                        "sample_id": sample_id,
                        "presentation_id": f"{audio_asset}__contextual__{rater_id}",
                        "presentation_variant": "contextual",
                        "task_mode": task_mode,
                        "audio_asset_id": audio_asset,
                        "context_type": context_type,
                        "context_id": _text(row.get("context_id")) or f"ctx_{sample_id}",
                        "context_text": context_text,
                        "context_audio_asset_id": context_asset,
                        "assigned_constructs": "|".join(CONTEXTUAL_CONSTRUCTS),
                        "analyzable_yes_no": "",
                        "clarity_comprehensibility_1to7": "",
                        "fluency_1to7": "",
                        "rhythm_naturalness_1to7": "",
                        "intonation_utterance_naturalness_1to7": "",
                        "intonation_contextual_appropriateness_1to7": "",
                        "context_presented_yes_no": "yes",
                        "timestamp": "",
                        "pack_schema": PACK_SCHEMA,
                    }
                )

    report = {
        "schema": PACK_SCHEMA,
        "manifest_schema": validation.get("schema"),
        "japanese_sample_count": japanese_sample_count,
        "contextual_sample_count": contextual_sample_count,
        "listener_row_count": len(listener_rows),
        "asset_row_count": len(asset_rows),
        "rater_count": len(raters),
        "ratings_per_presentation_requested": ratings_per_presentation,
        "target_transcript_exposed": False,
        "analysis_metadata_exposed": False,
    }
    return listener_rows, asset_rows, report


def _write_csv(path: str | Path, rows: list[Dict[str, Any]], fields: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_csv")
    parser.add_argument("--raters", required=True, help="comma-separated anonymized rater ids")
    parser.add_argument("--ratings-per-presentation", type=int, default=5)
    parser.add_argument("--listener-out", required=True)
    parser.add_argument("--asset-map-out", required=True)
    args = parser.parse_args()
    listener_rows, asset_rows, report = build_listener_pack(
        args.manifest_csv,
        rater_ids=[item.strip() for item in args.raters.split(",")],
        ratings_per_presentation=args.ratings_per_presentation,
    )
    listener_fields = list(listener_rows[0].keys()) if listener_rows else [
        "rater_id", "sample_id", "presentation_id", "presentation_variant", "task_mode",
        "audio_asset_id", "context_type", "context_id", "context_text", "context_audio_asset_id",
        "assigned_constructs", "analyzable_yes_no", "clarity_comprehensibility_1to7",
        "fluency_1to7", "rhythm_naturalness_1to7", "intonation_utterance_naturalness_1to7",
        "intonation_contextual_appropriateness_1to7", "context_presented_yes_no", "timestamp", "pack_schema",
    ]
    asset_fields = list(asset_rows[0].keys()) if asset_rows else [
        "sample_id", "audio_asset_id", "source_audio_path", "context_audio_asset_id",
        "source_context_audio_path", "private_only",
    ]
    _write_csv(args.listener_out, listener_rows, listener_fields)
    _write_csv(args.asset_map_out, asset_rows, asset_fields)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
