#!/usr/bin/env python3
"""Classify historical evaluation assets by what they may safely support in v10.

The audit is intentionally conservative.  Running a fixed-reading recording
through ``transcript_assisted_light`` does not turn it into spontaneous speech,
and a synthetic perturbation does not become a learner error.  This tool makes
those boundaries machine-readable before building a free-speech manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


AUDIT_SCHEMA = "historical_free_speech_reuse_audit_v10"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def classify_inventory_row(row: Mapping[str, Any]) -> dict[str, Any]:
    source = _text(row.get("source"))
    mode = _text(row.get("mode"))
    category = _text(row.get("expected_category"))
    speaker = _text(row.get("speaker"))
    wav_path = _text(row.get("wav_path"))

    if source == "JANON":
        reuse = "fixed_reading_regression"
        reason = "JANON inventory items are isolated fixed-reading words, not spontaneous/controlled-dialogue free speech"
    elif source == "JVS_parallel100":
        reuse = "native_fixed_reading_regression"
        reason = "JVS parallel100 items are scripted read speech; broad-mode execution does not change the elicitation construct"
    elif source == "JVS_controlled_edit":
        reuse = "engineering_robustness_only"
        reason = "controlled edits may test channel/timing robustness but are not independent human productions or learner errors"
    elif source == "demo_recording":
        reuse = "demo_smoke_only"
        reason = "demo recording lacks criterion-grade speaker/task/split provenance"
    elif category in {"non_japanese", "noise", "nonspeech", "negative_control"}:
        reuse = "routing_negative_control_candidate"
        reason = "negative control can test no-score routing if the underlying audio and provenance are still available"
    else:
        reuse = "manual_review_required"
        reason = "source does not provide enough elicitation/provenance information for automatic v10 classification"

    criterion_eligible = False
    if reuse == "manual_review_required" and mode == "transcript_assisted_light":
        reason += "; broad-mode execution alone is insufficient to prove free-speech elicitation"

    return {
        "sample_id": _text(row.get("sample_id")),
        "source": source,
        "speaker": speaker,
        "mode": mode,
        "expected_category": category,
        "wav_path": wav_path,
        "reuse_class": reuse,
        "free_speech_criterion_eligible": criterion_eligible,
        "reason": reason,
    }


def audit(
    inventory_csv: str | Path,
    *,
    learner_plan_csv: str | Path | None = None,
    negative_controls_csv: str | Path | None = None,
) -> dict[str, Any]:
    classified = [classify_inventory_row(row) for row in _read_csv(inventory_csv)]
    counts = Counter(row["reuse_class"] for row in classified)
    broad_mode_fixed_reading = sum(
        row["mode"] == "transcript_assisted_light" and row["source"] == "JVS_parallel100"
        for row in classified
    )

    legacy_plan = None
    if learner_plan_csv is not None:
        rows = _read_csv(learner_plan_csv)
        planned_speakers = {_text(row.get("planned_speaker_id_anonymized")) for row in rows if _text(row.get("planned_speaker_id_anonymized"))}
        statuses = Counter(_text(row.get("status")) for row in rows)
        legacy_plan = {
            "row_count": len(rows),
            "planned_speaker_count": len(planned_speakers),
            "status_counts": dict(statuses),
            "reuse_class": "legacy_fixed_reading_collection_plan",
            "free_speech_four_dimension_ready": False,
            "reason": "planned recordings are isolated fixed targets and do not supply spontaneous fluency or long-utterance rhythm/intonation evidence",
        }

    negative_controls = None
    if negative_controls_csv is not None:
        rows = _read_csv(negative_controls_csv)
        tmp_path_count = sum(_text(row.get("wav_path")).startswith("/private/tmp/") for row in rows)
        negative_controls = {
            "row_count": len(rows),
            "private_tmp_path_count": tmp_path_count,
            "routing_reuse_possible_if_audio_recovered_or_regenerated": True,
            "criterion_rating_role": False,
            "reason": "language/nonspeech controls test routing and no-score behavior, not Japanese four-dimension criterion validity",
        }

    eligible = [row for row in classified if row["free_speech_criterion_eligible"]]
    return {
        "schema": AUDIT_SCHEMA,
        "inventory_row_count": len(classified),
        "reuse_class_counts": dict(sorted(counts.items())),
        "broad_mode_fixed_reading_count": broad_mode_fixed_reading,
        "free_speech_criterion_eligible_count": len(eligible),
        "free_speech_criterion_ready_from_history": bool(eligible),
        "legacy_learner_collection_plan": legacy_plan,
        "historical_negative_controls": negative_controls,
        "sample_classifications": classified,
        "interpretation": (
            "historical assets remain useful for regression/routing/robustness, but fixed-reading or synthetic origin must not be relabeled as free-speech learner criterion data"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory_csv")
    parser.add_argument("--learner-plan", default=None)
    parser.add_argument("--negative-controls", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit(
        args.inventory_csv,
        learner_plan_csv=args.learner_plan,
        negative_controls_csv=args.negative_controls,
    )
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
