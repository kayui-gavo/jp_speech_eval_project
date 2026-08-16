#!/usr/bin/env python3
"""Generate deterministic recording assignments for the v10 free-speech corpus.

This replaces hand-written recording spreadsheets.  It uses only the operational
collection blueprint and never invents participant metadata such as L1 or
proficiency.  Speaker ids are pseudonymous planning ids and development/held
identities are disjoint by construction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


ASSIGNMENT_SCHEMA = "free_speech_collection_assignment_v10"
SPEAKER_META_SCHEMA = "free_speech_collection_speaker_metadata_v10"
DEFAULT_MIX = [
    "short_spontaneous",
    "short_controlled_dialogue",
    "long_spontaneous",
    "long_controlled_dialogue",
]


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _speaker_prefix(split: str, group: str) -> str:
    return ("H" if split == "held" else "D") + ("L" if group == "learner" else "N")


def _spontaneous_prompt(plan: Mapping[str, Any], bucket: str, prompt_index: int) -> tuple[str, str, str]:
    family = f"{bucket}_spontaneous"
    prompts = list((plan.get("prompt_families") or {}).get(family) or [])
    if not prompts:
        raise ValueError(f"collection plan has no prompts for {family}")
    index = prompt_index % len(prompts)
    return f"{family}_p{index + 1:02d}", str(prompts[index]), "prompt"


def _dialogue_prompt(plan: Mapping[str, Any], bucket: str, prompt_index: int) -> tuple[str, str, str]:
    prompts = list((plan.get("prompt_families") or {}).get("controlled_dialogue") or [])
    if not prompts:
        raise ValueError("collection plan has no controlled_dialogue prompts")
    item = prompts[prompt_index % len(prompts)]
    if not isinstance(item, Mapping):
        raise ValueError("controlled_dialogue prompt must be an object")
    base_id = str(item.get("prompt_id") or f"dialogue_{prompt_index + 1}")
    preceding_turn = str(item.get("preceding_turn") or "").strip()
    if not preceding_turn:
        raise ValueError(f"controlled_dialogue prompt {base_id} has no preceding_turn")
    return f"{base_id}_{bucket}", preceding_turn, "preceding_turn"


def _assignment_mix(plan_section: Mapping[str, Any], clips_per_speaker: int) -> list[str]:
    explicit = list(plan_section.get("per_speaker_target_mix") or [])
    source = explicit or DEFAULT_MIX
    if not source:
        raise ValueError("empty collection mix")
    return [source[index % len(source)] for index in range(clips_per_speaker)]


def build_assignments(plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    speaker_rows: list[dict[str, Any]] = []
    prompt_counters = {
        "short_spontaneous": 0,
        "long_spontaneous": 0,
        "short_controlled_dialogue": 0,
        "long_controlled_dialogue": 0,
    }

    for split, section_key in (("development", "development_japanese_core"), ("held", "held_japanese_core")):
        section = plan.get(section_key) or {}
        for group in ("learner", "native"):
            group_plan = section.get(group) or {}
            speaker_count = int(group_plan.get("speaker_count") or 0)
            clips_per_speaker = int(group_plan.get("clean_clips_per_speaker") or 0)
            mix = _assignment_mix(group_plan, clips_per_speaker)
            prefix = _speaker_prefix(split, group)
            for speaker_index in range(speaker_count):
                speaker_id = f"{prefix}{speaker_index + 1:02d}"
                speaker_rows.append(
                    {
                        "speaker_id": speaker_id,
                        "speaker_group": group,
                        "split": split,
                        "l1": "",
                        "japanese_proficiency": "",
                        "recording_device": "",
                        "consent_or_dataset_provenance": "",
                        "notes": "",
                        "metadata_schema": SPEAKER_META_SCHEMA,
                    }
                )
                for clip_index, target in enumerate(mix, start=1):
                    if target not in prompt_counters:
                        raise ValueError(f"unknown target mix item: {target}")
                    bucket = "short" if target.startswith("short_") else "long"
                    task_mode = "spontaneous" if target.endswith("_spontaneous") else "controlled_dialogue"
                    counter = prompt_counters[target]
                    prompt_counters[target] += 1
                    if task_mode == "spontaneous":
                        prompt_id, context_text, context_type = _spontaneous_prompt(plan, bucket, counter)
                    else:
                        prompt_id, context_text, context_type = _dialogue_prompt(plan, bucket, counter)
                    duration_instruction = str((plan.get("duration_targets") or {}).get(f"{bucket}_instruction") or "")
                    sample_id = f"{speaker_id}_{clip_index:02d}"
                    relpath = f"{split}/{group}/{speaker_id}/{sample_id}.wav"
                    assignments.append(
                        {
                            "sample_id": sample_id,
                            "speaker_id": speaker_id,
                            "speaker_group": group,
                            "split": split,
                            "task_mode": task_mode,
                            "planned_duration_bucket": bucket,
                            "prompt_id": prompt_id,
                            "context_type": context_type,
                            "context_text": context_text,
                            "collection_instruction": duration_instruction,
                            "audio_relpath": relpath,
                            "status": "needed",
                            "assignment_schema": ASSIGNMENT_SCHEMA,
                        }
                    )

    identities_by_split = {
        split: {row["speaker_id"] for row in speaker_rows if row["split"] == split}
        for split in ("development", "held")
    }
    overlap = identities_by_split["development"] & identities_by_split["held"]
    if overlap:
        raise RuntimeError(f"speaker split overlap: {sorted(overlap)}")

    bucket_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    for row in assignments:
        bucket_counts[row["planned_duration_bucket"]] = bucket_counts.get(row["planned_duration_bucket"], 0) + 1
        task_counts[row["task_mode"]] = task_counts.get(row["task_mode"], 0) + 1
    report = {
        "schema": ASSIGNMENT_SCHEMA,
        "assignment_count": len(assignments),
        "speaker_count": len(speaker_rows),
        "development_speaker_count": len(identities_by_split["development"]),
        "held_speaker_count": len(identities_by_split["held"]),
        "speaker_split_overlap_count": 0,
        "planned_duration_bucket_counts": bucket_counts,
        "task_mode_counts": task_counts,
        "participant_metadata_fabricated": False,
        "response_transcript_requested": False,
    }
    return assignments, speaker_rows, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default=str(Path(__file__).resolve().parents[1] / "data" / "human_eval" / "free_speech_v10_minimum_collection_plan.json"),
    )
    parser.add_argument("--assignments-out", required=True)
    parser.add_argument("--speaker-metadata-out", required=True)
    parser.add_argument("--report-out", default=None)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    assignments, speakers, report = build_assignments(plan)
    _write_csv(
        args.assignments_out,
        assignments,
        [
            "sample_id", "speaker_id", "speaker_group", "split", "task_mode",
            "planned_duration_bucket", "prompt_id", "context_type", "context_text",
            "collection_instruction", "audio_relpath", "status", "assignment_schema",
        ],
    )
    _write_csv(
        args.speaker_metadata_out,
        speakers,
        [
            "speaker_id", "speaker_group", "split", "l1", "japanese_proficiency",
            "recording_device", "consent_or_dataset_provenance", "notes", "metadata_schema",
        ],
    )
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
