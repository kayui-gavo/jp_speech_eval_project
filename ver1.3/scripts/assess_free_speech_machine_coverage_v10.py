#!/usr/bin/env python3
"""Preflight v10 held learner coverage before spending human-rating effort.

This reads the v10 machine-evidence export after real product-condition
endpointing. It checks only whether the collected held learner set has enough
actual short/long and spontaneous/dialogue coverage for the frozen v10 gates.
It does not use human ratings, estimate construct validity, or change scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional


PREFLIGHT_SCHEMA = "free_speech_machine_coverage_preflight_v10"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    return _text(value).lower() in {"true", "1", "yes"}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sample_index(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    stable_fields = ("speaker_id", "speaker_group", "task", "subset", "expected_language")
    for row in rows:
        sample_id = _text(row.get("sample_id"))
        if not sample_id:
            continue
        sample = samples.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                "speaker_id": _text(row.get("speaker_id")),
                "speaker_group": _text(row.get("speaker_group")),
                "task": _text(row.get("task")),
                "subset": _text(row.get("subset")),
                "expected_language": _text(row.get("expected_language")),
                "speech_duration_sec": _finite(row.get("speech_duration_sec")),
                "candidate_availability": {},
            },
        )
        for field in stable_fields:
            value = _text(row.get(field))
            if value and sample.get(field) and value != sample[field]:
                raise ValueError(f"inconsistent {field} across evidence rows for {sample_id}")
            if value and not sample.get(field):
                sample[field] = value
        duration = _finite(row.get("speech_duration_sec"))
        if duration is not None:
            old = sample.get("speech_duration_sec")
            if old is not None and abs(float(old) - duration) > 1e-6:
                raise ValueError(f"inconsistent speech_duration_sec across evidence rows for {sample_id}")
            sample["speech_duration_sec"] = duration
        candidate = _text(row.get("candidate"))
        if candidate:
            sample["candidate_availability"][candidate] = _truthy(row.get("available"))
    return samples


def _count_slices(samples: list[Mapping[str, Any]], *, short_max: float, long_min: float) -> dict[str, int]:
    return {
        "all": len(samples),
        "short": sum((_finite(item.get("speech_duration_sec")) or float("inf")) <= short_max for item in samples),
        "long": sum((_finite(item.get("speech_duration_sec")) or -1.0) >= long_min for item in samples),
        "duration_missing": sum(_finite(item.get("speech_duration_sec")) is None for item in samples),
        "spontaneous": sum(_text(item.get("task")) == "spontaneous" for item in samples),
        "controlled_dialogue": sum(_text(item.get("task")) == "controlled_dialogue" for item in samples),
    }


def _minimum_state(value: int, threshold: int) -> str:
    return "pass" if value >= threshold else "insufficient"


def assess(evidence_csv: str | Path, protocol_json: str | Path) -> dict[str, Any]:
    rows = _read_csv(evidence_csv)
    protocol = _load_json(protocol_json)
    gates = dict(protocol.get("consumer_discrimination_gates") or {})
    candidates = list((protocol.get("direct_product_candidates") or {}).keys())
    samples = list(_sample_index(rows).values())
    held_learner = [
        sample for sample in samples
        if _text(sample.get("subset")) == "held"
        and _text(sample.get("speaker_group")) == "learner"
        and _text(sample.get("expected_language")).lower() == "ja"
    ]
    short_max = float(gates["short_utterance_max_speech_sec"])
    long_min = float(gates["long_utterance_min_speech_sec"])
    slice_counts = _count_slices(held_learner, short_max=short_max, long_min=long_min)
    minimums = {
        "all": int(gates["held_learner_construct_matched_pair_count"]),
        "short": int(gates["held_learner_short_construct_matched_pair_count"]),
        "long": int(gates["held_learner_long_construct_matched_pair_count"]),
        "spontaneous": int(gates["held_learner_spontaneous_construct_matched_pair_count"]),
        "controlled_dialogue": int(gates["held_learner_controlled_dialogue_construct_matched_pair_count"]),
    }
    structure_states = {key: _minimum_state(slice_counts[key], minimum) for key, minimum in minimums.items()}
    structure_ready = all(state == "pass" for state in structure_states.values()) and slice_counts["duration_missing"] == 0

    candidate_reports: dict[str, Any] = {}
    for candidate in candidates:
        available = [sample for sample in held_learner if bool((sample.get("candidate_availability") or {}).get(candidate))]
        counts = _count_slices(available, short_max=short_max, long_min=long_min)
        states = {key: _minimum_state(counts[key], minimum) for key, minimum in minimums.items()}
        candidate_reports[candidate] = {
            "available_held_learner_sample_count": len(available),
            "slice_counts": counts,
            "coverage_states": states,
            "coverage_ready_for_v10_human_join": all(state == "pass" for state in states.values()) and counts["duration_missing"] == 0,
        }

    missing_structure = [key for key, state in structure_states.items() if state != "pass"]
    if slice_counts["duration_missing"]:
        missing_structure.append("speech_duration_sec")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "protocol_schema": protocol.get("schema_version"),
        "held_learner_sample_count": len(held_learner),
        "held_learner_speaker_count": len({_text(sample.get("speaker_id")) for sample in held_learner if _text(sample.get("speaker_id"))}),
        "held_learner_slice_counts": slice_counts,
        "frozen_minimums": minimums,
        "collection_structure_states": structure_states,
        "collection_structure_ready_for_final_listener_pack": structure_ready,
        "missing_or_undercovered_structure": missing_structure,
        "candidate_coverage": candidate_reports,
        "human_ratings_used": False,
        "product_score_changed": False,
        "interpretation": (
            "coverage preflight only: pass means the collected held learner structure can support the frozen v10 joins; "
            "it does not mean any candidate is valid or promotable"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_csv")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = assess(args.evidence_csv, args.protocol)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
