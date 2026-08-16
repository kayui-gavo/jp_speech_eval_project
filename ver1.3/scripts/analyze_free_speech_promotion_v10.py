#!/usr/bin/env python3
"""Layer C-end discrimination gates on top of the frozen v5 promotion analysis.

The v5 protocol remains the scientific construct-validity baseline. v10 adds
product-specific checks that prevent a candidate from passing merely because it
separates native and learner groups while collapsing learner scores into a
narrow band, or because native controls rescue a learner-only failure in one
utterance-length or task regime.

No score mapping or threshold is fitted here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

from analyze_free_speech_promotion_v5 import analyze as analyze_v5
from analyze_free_speech_promotion_v5 import aggregate_human


ANALYSIS_SCHEMA = "free_speech_promotion_analysis_v10_v2"


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


def _read_csv(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _spearman(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usable = [
        row for row in rows
        if _finite(row.get("evidence_value")) is not None
        and _finite(row.get("human_rating_mean")) is not None
    ]
    if len(usable) < 3:
        return {"n": len(usable), "rho": None, "reason": "fewer_than_3_pairs"}
    x = np.asarray([float(row["evidence_value"]) for row in usable], dtype=float)
    y = np.asarray([float(row["human_rating_mean"]) for row in usable], dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"n": len(usable), "rho": None, "reason": "constant_input"}
    result = spearmanr(x, y)
    rho = float(result.statistic)
    return {"n": len(usable), "rho": rho if math.isfinite(rho) else None, "reason": "ok"}


def _iqr(rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    values = [_finite(row.get("evidence_value")) for row in rows]
    arr = np.asarray([value for value in values if value is not None], dtype=float)
    if arr.size < 2:
        return None
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def _median(rows: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
    values = [_finite(row.get(field)) for row in rows]
    arr = np.asarray([value for value in values if value is not None], dtype=float)
    return float(np.median(arr)) if arr.size else None


def _gate_min(value: Optional[float], threshold: float) -> str:
    if value is None:
        return "insufficient"
    return "pass" if value >= threshold else "fail"


def _combine(states: Sequence[str]) -> str:
    if any(state == "fail" for state in states):
        return "fail"
    if any(state == "insufficient" for state in states):
        return "insufficient"
    return "pass"


def _joined_rows(
    human_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    criterion: str,
    minimum_human_rating_count: int,
) -> list[dict[str, Any]]:
    human = aggregate_human(human_rows)
    evidence_index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in evidence_rows:
        sample_id = _text(row.get("sample_id"))
        name = _text(row.get("candidate"))
        if sample_id and name:
            evidence_index[(sample_id, name)] = row

    output: list[dict[str, Any]] = []
    for row in human:
        if _text(row.get("criterion")) != criterion or _text(row.get("subset")) != "held":
            continue
        if int(_finite(row.get("human_rating_count")) or 0) < minimum_human_rating_count:
            continue
        evidence = evidence_index.get((_text(row.get("sample_id")), candidate))
        if evidence is None or not _truthy(evidence.get("available")):
            continue
        value = _finite(evidence.get("evidence_value"))
        if value is None:
            continue
        output.append(
            {
                **dict(row),
                "evidence_value": value,
                "speech_duration_sec": _finite(evidence.get("speech_duration_sec")),
                "speaker_group": _text(row.get("speaker_group")) or _text(evidence.get("speaker_group")),
                "task": _text(row.get("task")) or _text(evidence.get("task")),
            }
        )
    return output


def _length_report(rows: Sequence[Mapping[str, Any]], *, short_max: float, long_min: float) -> dict[str, Any]:
    short_rows = [row for row in rows if (_finite(row.get("speech_duration_sec")) or float("inf")) <= short_max]
    long_rows = [row for row in rows if (_finite(row.get("speech_duration_sec")) or -1.0) >= long_min]
    duration_available = sum(_finite(row.get("speech_duration_sec")) is not None for row in rows)
    return {
        "population": "held_learner_only",
        "duration_available_count": duration_available,
        "duration_availability_rate": (duration_available / len(rows)) if rows else None,
        "short": {"pair_count": len(short_rows), "spearman": _spearman(short_rows)},
        "long": {"pair_count": len(long_rows), "spearman": _spearman(long_rows)},
    }


def _task_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"population": "held_learner_only"}
    for task in ("spontaneous", "controlled_dialogue"):
        selected = [row for row in rows if _text(row.get("task")) == task]
        output[task] = {"pair_count": len(selected), "spearman": _spearman(selected)}
    unknown_count = sum(_text(row.get("task")) not in {"spontaneous", "controlled_dialogue"} for row in rows)
    output["unknown_task_pair_count"] = unknown_count
    return output


def _direction_gate(report: Mapping[str, Any], minimum_pair_count: int) -> str:
    pair_count = int(report.get("pair_count") or 0)
    if pair_count < minimum_pair_count:
        return "insufficient"
    rho = _finite((report.get("spearman") if isinstance(report.get("spearman"), Mapping) else {}).get("rho"))
    return "pass" if rho is not None and rho > 0 else "fail"


def _group_direction(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_human_gap: float,
    maximum_machine_gap_when_human_small: float,
) -> dict[str, Any]:
    learner = [row for row in rows if _text(row.get("speaker_group")) == "learner"]
    native = [row for row in rows if _text(row.get("speaker_group")) == "native"]
    learner_human = _median(learner, "human_rating_mean")
    native_human = _median(native, "human_rating_mean")
    learner_machine = _median(learner, "evidence_value")
    native_machine = _median(native, "evidence_value")
    regime = "unavailable"
    if None in {learner_human, native_human, learner_machine, native_machine}:
        state = "insufficient"
        human_gap = machine_gap = None
    else:
        human_gap = float(native_human - learner_human)
        machine_gap = float(native_machine - learner_machine)
        if abs(human_gap) < minimum_human_gap:
            regime = "human_group_gap_small"
            state = "pass" if abs(machine_gap) <= maximum_machine_gap_when_human_small else "fail"
        else:
            regime = "human_group_gap_interpretable"
            state = "pass" if human_gap * machine_gap > 0 else "fail"
    return {
        "learner_pair_count": len(learner),
        "native_pair_count": len(native),
        "learner_human_median": learner_human,
        "native_human_median": native_human,
        "learner_candidate_median": learner_machine,
        "native_candidate_median": native_machine,
        "human_native_minus_learner": human_gap,
        "candidate_native_minus_learner": machine_gap,
        "regime": regime,
        "state": state,
        "interpretation": (
            "native_controls_check_construct_contamination_not_native_likeness: follow_human_direction_when_human_gap_is_interpretable; "
            "otherwise_do_not_invent_a_large_machine_group_gap"
        ),
    }


def analyze(
    human_csv: str | Path,
    evidence_csv: str | Path,
    base_protocol_json: str | Path,
    v10_protocol_json: str | Path,
) -> dict[str, Any]:
    base = analyze_v5(human_csv, evidence_csv, base_protocol_json)
    human_rows, _ = _read_csv(human_csv)
    evidence_rows, evidence_fields = _read_csv(evidence_csv)
    if "speech_duration_sec" not in evidence_fields:
        raise ValueError("v10 evidence file missing speech_duration_sec; use export_free_speech_v10_evidence.py")

    base_protocol = _load_json(base_protocol_json)
    protocol = _load_json(v10_protocol_json)
    gates = dict(protocol.get("consumer_discrimination_gates") or {})
    direct = dict(protocol.get("direct_product_candidates") or {})
    minimum_rating_count = int(
        (base_protocol.get("minimum_promotion_gates") or {}).get("human_rating_count_minimum_for_held_inclusion", 3)
    )

    reports: dict[str, Any] = {}
    for candidate, criterion in direct.items():
        rows = _joined_rows(
            human_rows,
            evidence_rows,
            candidate=candidate,
            criterion=criterion,
            minimum_human_rating_count=minimum_rating_count,
        )
        learner = [row for row in rows if _text(row.get("speaker_group")) == "learner"]
        learner_spearman = _spearman(learner)
        learner_iqr = _iqr(learner)
        length = _length_report(
            learner,
            short_max=float(gates["short_utterance_max_speech_sec"]),
            long_min=float(gates["long_utterance_min_speech_sec"]),
        )
        task = _task_report(learner)
        group_direction = _group_direction(
            rows,
            minimum_human_gap=float(gates["native_vs_learner_group_direction_min_human_gap_1to7"]),
            maximum_machine_gap_when_human_small=float(gates["machine_group_gap_max_when_human_gap_small_points"]),
        )

        short = length.get("short") if isinstance(length.get("short"), Mapping) else {}
        long = length.get("long") if isinstance(length.get("long"), Mapping) else {}
        spontaneous = task.get("spontaneous") if isinstance(task.get("spontaneous"), Mapping) else {}
        controlled = task.get("controlled_dialogue") if isinstance(task.get("controlled_dialogue"), Mapping) else {}
        short_n = int(short.get("pair_count") or 0)
        long_n = int(long.get("pair_count") or 0)
        spontaneous_n = int(spontaneous.get("pair_count") or 0)
        controlled_n = int(controlled.get("pair_count") or 0)
        short_min = int(gates["held_learner_short_construct_matched_pair_count"])
        long_min = int(gates["held_learner_long_construct_matched_pair_count"])
        spontaneous_min = int(gates["held_learner_spontaneous_construct_matched_pair_count"])
        controlled_min = int(gates["held_learner_controlled_dialogue_construct_matched_pair_count"])

        base_decision = ((base.get("candidates") or {}).get(candidate) or {}).get("promotion_decision", "insufficient")
        gate_states = {
            "base_v5_scientific_promotion": base_decision,
            "held_learner_pair_count": _gate_min(
                float(len(learner)), float(gates["held_learner_construct_matched_pair_count"])
            ),
            "held_learner_spearman": _gate_min(
                _finite(learner_spearman.get("rho")), float(gates["held_learner_spearman_rho"])
            ),
            "held_learner_candidate_iqr": _gate_min(
                learner_iqr, float(gates["held_learner_candidate_iqr_min_points"])
            ),
            "held_learner_short_pair_count": _gate_min(float(short_n), float(short_min)),
            "held_learner_long_pair_count": _gate_min(float(long_n), float(long_min)),
            "learner_short_direction_positive": _direction_gate(short, short_min),
            "learner_long_direction_positive": _direction_gate(long, long_min),
            "held_learner_spontaneous_pair_count": _gate_min(float(spontaneous_n), float(spontaneous_min)),
            "held_learner_controlled_dialogue_pair_count": _gate_min(float(controlled_n), float(controlled_min)),
            "learner_spontaneous_direction_positive": _direction_gate(spontaneous, spontaneous_min),
            "learner_controlled_dialogue_direction_positive": _direction_gate(controlled, controlled_min),
            "native_learner_construct_contamination_check": group_direction["state"],
        }
        reports[candidate] = {
            "criterion": criterion,
            "held_available_pair_count": len(rows),
            "held_learner": {
                "pair_count": len(learner),
                "spearman": learner_spearman,
                "candidate_iqr": learner_iqr,
            },
            "held_learner_utterance_length": length,
            "held_learner_task_mode": task,
            "native_learner_group_direction": group_direction,
            "gate_states": gate_states,
            "promotion_readiness": _combine(list(gate_states.values())),
        }

    states = [report["promotion_readiness"] for report in reports.values()]
    return {
        "schema": ANALYSIS_SCHEMA,
        "base_analysis_schema": base.get("schema"),
        "base_protocol_schema": base.get("protocol_schema"),
        "v10_protocol_schema": protocol.get("schema_version"),
        "product_score_changed": False,
        "thresholds_fitted_on_held_data": False,
        "base_v5": base,
        "consumer_discrimination": reports,
        "overall_v10_promotion_readiness": _combine(states) if states else "insufficient",
        "next_step": "only candidates with pass may enter a separate score-contract A/B branch; all others remain shadow",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("human_csv")
    parser.add_argument("evidence_csv")
    parser.add_argument("--base-protocol", required=True)
    parser.add_argument("--v10-protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = analyze(args.human_csv, args.evidence_csv, args.base_protocol, args.v10_protocol)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
