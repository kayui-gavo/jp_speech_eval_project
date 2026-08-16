#!/usr/bin/env python3
"""Execute the frozen free-speech v5 C-end promotion gates.

This analyzer is deliberately stricter than the generic evidence correlation
report.  It evaluates only predeclared construct matches, keeps raw diagnostics
separate from 0-100 score surfaces, checks held-set sample/speaker/task coverage,
channel-paired drift, negative-control routing, transcript-oracle leakage, and
F0-missingness safety.

It never fits thresholds or remaps scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr


ANALYSIS_SCHEMA = "free_speech_promotion_analysis_v5"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool(value: Any) -> Optional[bool]:
    text = _text(value).lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _read_csv(path: str | Path) -> tuple[list[Dict[str, str]], list[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _load_protocol(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def aggregate_human(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    metadata: dict[tuple[str, str], Dict[str, str]] = {}
    meta_fields = (
        "speaker_id", "speaker_group", "l1", "task", "prompt_id", "subset",
        "expected_language", "condition", "channel_pair_id", "context_type",
        "context_id", "presentation_variant",
    )
    for row in rows:
        sample_id = _text(row.get("sample_id"))
        criterion = _text(row.get("criterion"))
        rating = _finite(row.get("human_rating"))
        if not sample_id or not criterion or rating is None:
            continue
        key = (sample_id, criterion)
        grouped[key].append(rating)
        meta = metadata.setdefault(key, {})
        for field in meta_fields:
            value = _text(row.get(field))
            if not value:
                continue
            previous = meta.get(field)
            if previous is not None and previous != value:
                raise ValueError(f"conflicting {field} for sample={sample_id}, criterion={criterion}")
            meta[field] = value
    output: list[Dict[str, Any]] = []
    for (sample_id, criterion), ratings in sorted(grouped.items()):
        output.append(
            {
                "sample_id": sample_id,
                "criterion": criterion,
                "human_rating_mean": float(np.mean(ratings)),
                "human_rating_median": float(np.median(ratings)),
                "human_rating_count": len(ratings),
                **metadata[(sample_id, criterion)],
            }
        )
    return output


def _evidence_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Dict[str, Any]]:
    index: dict[tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        sample_id = _text(row.get("sample_id"))
        candidate = _text(row.get("candidate"))
        if not sample_id or not candidate:
            continue
        key = (sample_id, candidate)
        if key in index:
            raise ValueError(f"duplicate evidence row: {sample_id}|{candidate}")
        index[key] = dict(row)
    return index


def _unique_sample_metadata(evidence_rows: Sequence[Mapping[str, Any]]) -> dict[str, Dict[str, Any]]:
    fields = (
        "speaker_id", "speaker_group", "l1", "task", "prompt_id", "subset",
        "expected_language", "condition", "channel_pair_id", "context_type",
        "context_id", "batch_status", "product_score_available", "language_gate_eligible",
        "language_gate_reason", "scoring_used_gold_transcript",
    )
    output: dict[str, Dict[str, Any]] = {}
    for row in evidence_rows:
        sample_id = _text(row.get("sample_id"))
        if not sample_id:
            continue
        meta = output.setdefault(sample_id, {})
        for field in fields:
            value = _text(row.get(field))
            if not value:
                continue
            previous = _text(meta.get(field))
            if previous and previous != value:
                raise ValueError(f"conflicting evidence metadata {field} for sample={sample_id}")
            meta[field] = value
    return output


def _spearman(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    usable = [
        row for row in rows
        if _finite(row.get("evidence_value")) is not None
        and _finite(row.get("human_rating_mean")) is not None
    ]
    if len(usable) < 3:
        return {"n": len(usable), "rho": None, "pvalue": None, "reason": "fewer_than_3_pairs"}
    x = np.asarray([float(row["evidence_value"]) for row in usable], dtype=float)
    y = np.asarray([float(row["human_rating_mean"]) for row in usable], dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"n": len(usable), "rho": None, "pvalue": None, "reason": "constant_input"}
    result = spearmanr(x, y)
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "n": len(usable),
        "rho": rho if math.isfinite(rho) else None,
        "pvalue": pvalue if math.isfinite(pvalue) else None,
        "reason": "ok",
    }


def _iqr(rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    values = [_finite(row.get("evidence_value")) for row in rows]
    usable = np.asarray([value for value in values if value is not None], dtype=float)
    if usable.size < 2:
        return None
    return float(np.percentile(usable, 75) - np.percentile(usable, 25))


def _join_candidate(
    human: Sequence[Mapping[str, Any]],
    evidence_index: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    candidate: str,
    criterion: str,
    subset: str = "held",
) -> list[Dict[str, Any]]:
    output: list[Dict[str, Any]] = []
    for human_row in human:
        if _text(human_row.get("criterion")) != criterion:
            continue
        if subset and _text(human_row.get("subset")) != subset:
            continue
        sample_id = _text(human_row.get("sample_id"))
        evidence = evidence_index.get((sample_id, candidate))
        if evidence is None:
            output.append({**dict(human_row), "candidate": candidate, "evidence_value": None, "available": False})
            continue
        value = _finite(evidence.get("evidence_value"))
        available = value is not None and _bool(evidence.get("available")) is not False
        output.append(
            {
                **dict(human_row),
                "candidate": candidate,
                "evidence_value": value if available else None,
                "available": available,
                "failure_reason": _text(evidence.get("failure_reason")),
            }
        )
    return output


def _task_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        task = _text(row.get("task"))
        if task:
            grouped[task].append(row)
    reports = {task: _spearman(group_rows) for task, group_rows in sorted(grouped.items())}
    usable_groups = sum(report.get("rho") is not None for report in reports.values())
    return {"groups": reports, "usable_group_count": usable_groups}


def _leave_one_speaker_out(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    speakers = sorted({_text(row.get("speaker_id")) for row in rows if _text(row.get("speaker_id"))})
    values: list[float] = []
    per_speaker: Dict[str, Any] = {}
    for speaker in speakers:
        report = _spearman([row for row in rows if _text(row.get("speaker_id")) != speaker])
        per_speaker[speaker] = report
        if report.get("rho") is not None:
            values.append(float(report["rho"]))
    return {
        "speaker_count": len(speakers),
        "usable_leave_one_out_count": len(values),
        "rho_min": min(values) if values else None,
        "rho_max": max(values) if values else None,
        "all_same_positive_direction": bool(values) and all(value > 0 for value in values),
        "per_left_out_speaker": per_speaker,
    }


def _channel_drift(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    subset: str = "held",
) -> Dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        if _text(row.get("candidate")) != candidate:
            continue
        if subset and _text(row.get("subset")) != subset:
            continue
        if _text(row.get("expected_language")) != "ja":
            continue
        if _finite(row.get("evidence_value")) is None or _bool(row.get("available")) is False:
            continue
        pair_id = _text(row.get("channel_pair_id"))
        if pair_id:
            grouped[pair_id].append(row)

    drifts: list[float] = []
    usable_pairs = 0
    pair_report: Dict[str, Any] = {}
    for pair_id, pair_rows in sorted(grouped.items()):
        clean = [float(row["evidence_value"]) for row in pair_rows if _text(row.get("condition")) == "clean"]
        variants = [float(row["evidence_value"]) for row in pair_rows if _text(row.get("condition")) != "clean"]
        if not clean or not variants:
            pair_report[pair_id] = {"usable": False, "reason": "missing_clean_or_variant"}
            continue
        clean_anchor = float(np.median(np.asarray(clean, dtype=float)))
        pair_drifts = [abs(value - clean_anchor) for value in variants]
        drifts.extend(pair_drifts)
        usable_pairs += 1
        pair_report[pair_id] = {
            "usable": True,
            "clean_anchor": clean_anchor,
            "variant_count": len(variants),
            "median_absolute_drift": float(np.median(np.asarray(pair_drifts, dtype=float))),
            "max_absolute_drift": max(pair_drifts),
        }
    return {
        "usable_pair_count": usable_pairs,
        "comparison_count": len(drifts),
        "median_absolute_drift": float(np.median(np.asarray(drifts, dtype=float))) if drifts else None,
        "max_absolute_drift": max(drifts) if drifts else None,
        "pairs": pair_report,
    }


def _global_dataset_report(sample_meta: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    japanese = [
        (sample_id, meta) for sample_id, meta in sample_meta.items()
        if _text(meta.get("expected_language")) == "ja" and _text(meta.get("speaker_group")) in {"learner", "native"}
    ]
    held = [(sample_id, meta) for sample_id, meta in japanese if _text(meta.get("subset")) == "held"]
    held_speakers = {_text(meta.get("speaker_id")) for _, meta in held if _text(meta.get("speaker_id"))}
    held_learner = {
        _text(meta.get("speaker_id")) for _, meta in held
        if _text(meta.get("speaker_group")) == "learner" and _text(meta.get("speaker_id"))
    }
    held_native = {
        _text(meta.get("speaker_id")) for _, meta in held
        if _text(meta.get("speaker_group")) == "native" and _text(meta.get("speaker_id"))
    }

    speaker_splits: dict[str, set[str]] = defaultdict(set)
    for _, meta in japanese:
        speaker = _text(meta.get("speaker_id"))
        split = _text(meta.get("subset"))
        if speaker and split:
            speaker_splits[speaker].add(split)
    leakage = sorted(speaker for speaker, splits in speaker_splits.items() if len(splits) > 1)

    negative = [
        meta for meta in sample_meta.values()
        if _text(meta.get("speaker_group")) == "negative_control" or _text(meta.get("expected_language")) not in {"", "ja"}
    ]
    negative_no_score = 0
    for meta in negative:
        if _bool(meta.get("product_score_available")) is False:
            negative_no_score += 1
    negative_rate = (negative_no_score / len(negative)) if negative else None

    oracle_usage = sum(_bool(meta.get("scoring_used_gold_transcript")) is True for meta in sample_meta.values())
    return {
        "held_speaker_count": len(held_speakers),
        "held_learner_speaker_count": len(held_learner),
        "held_native_speaker_count": len(held_native),
        "speaker_split_leakage": leakage,
        "negative_control_count": len(negative),
        "negative_control_no_score_count": negative_no_score,
        "negative_control_no_score_rate": negative_rate,
        "gold_transcript_scoring_usage_count": oracle_usage,
    }


def _f0_missingness_safety(
    evidence_index: Mapping[tuple[str, str], Mapping[str, Any]],
    sample_meta: Mapping[str, Mapping[str, Any]],
    *,
    minimum_candidate_score: float,
) -> Dict[str, Any]:
    cases = 0
    unsafe = 0
    details: list[Dict[str, Any]] = []
    for sample_id, meta in sorted(sample_meta.items()):
        if _text(meta.get("expected_language")) != "ja" or _text(meta.get("subset")) != "held":
            continue
        raw_f0 = evidence_index.get((sample_id, "intonation.robust_range_semitones"))
        surface = evidence_index.get((sample_id, "intonation.shadow_candidate_score"))
        if raw_f0 is None:
            continue
        raw_available = _finite(raw_f0.get("evidence_value")) is not None and _bool(raw_f0.get("available")) is not False
        if raw_available:
            continue
        cases += 1
        score = _finite(surface.get("evidence_value")) if surface is not None else None
        is_unsafe = score is not None and score < minimum_candidate_score
        unsafe += int(is_unsafe)
        details.append({"sample_id": sample_id, "candidate_score": score, "unsafe_low_penalty": is_unsafe})
    return {
        "f0_missing_case_count": cases,
        "unsafe_low_penalty_count": unsafe,
        "minimum_allowed_candidate_score_when_present": minimum_candidate_score,
        "pass": (unsafe == 0) if cases else None,
        "cases": details,
    }


def _gate(value: Optional[float], threshold: float, *, op: str) -> str:
    if value is None:
        return "insufficient"
    if op == "min":
        return "pass" if value >= threshold else "fail"
    if op == "max":
        return "pass" if value <= threshold else "fail"
    raise ValueError(op)


def _combine_gate_states(states: Iterable[str]) -> str:
    values = list(states)
    if any(value == "fail" for value in values):
        return "fail"
    if any(value == "insufficient" for value in values):
        return "insufficient"
    return "pass"


def analyze(
    human_csv: str | Path,
    evidence_csv: str | Path,
    protocol_json: str | Path,
) -> Dict[str, Any]:
    human_rows, human_fields = _read_csv(human_csv)
    evidence_rows, evidence_fields = _read_csv(evidence_csv)
    for required, fields, label in (
        ({"sample_id", "criterion", "human_rating", "subset", "speaker_id", "task"}, set(human_fields), "human"),
        ({"sample_id", "candidate", "candidate_construct", "evidence_value", "subset", "speaker_id", "condition"}, set(evidence_fields), "evidence"),
    ):
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{label} file missing columns: {','.join(missing)}")

    protocol = _load_protocol(protocol_json)
    thresholds = dict(protocol.get("minimum_promotion_gates") or {})
    candidate_types = dict(protocol.get("candidate_types") or {})
    construct_match = dict(protocol.get("candidate_construct_match") or {})
    direct = dict(protocol.get("direct_product_promotion_eligibility") or {})

    human = aggregate_human(human_rows)
    evidence_index = _evidence_index(evidence_rows)
    sample_meta = _unique_sample_metadata(evidence_rows)
    global_report = _global_dataset_report(sample_meta)
    f0_safety = _f0_missingness_safety(
        evidence_index,
        sample_meta,
        minimum_candidate_score=float(thresholds.get("f0_failure_intonation_candidate_min_points", 65.0)),
    )

    candidate_reports: Dict[str, Any] = {}
    for candidate, candidate_type in sorted(candidate_types.items()):
        matched_criteria = list(construct_match.get(candidate) or [])
        criterion_reports: Dict[str, Any] = {}
        for criterion in matched_criteria:
            joined = _join_candidate(human, evidence_index, candidate=candidate, criterion=criterion, subset="held")
            available_count = sum(bool(row.get("available")) for row in joined)
            availability = available_count / len(joined) if joined else None
            correlation = _spearman(joined)
            criterion_reports[criterion] = {
                "held_pair_count": len(joined),
                "held_available_count": available_count,
                "availability_rate": availability,
                "spearman": correlation,
                "candidate_iqr": _iqr(joined),
                "task_stability": _task_report(joined),
                "leave_one_speaker_out": _leave_one_speaker_out(joined),
            }

        report: Dict[str, Any] = {
            "candidate_type": candidate_type,
            "construct_match": matched_criteria,
            "criteria": criterion_reports,
        }
        if candidate_type != "score_surface" or candidate not in direct:
            report["promotion_decision"] = "diagnostic_only"
            candidate_reports[candidate] = report
            continue

        criterion = list(direct[candidate])[0]
        stats = criterion_reports.get(criterion, {})
        drift = _channel_drift(evidence_rows, candidate=candidate, subset="held")
        task_usable = ((stats.get("task_stability") or {}).get("usable_group_count"))
        rho = ((stats.get("spearman") or {}).get("rho"))
        gate_states = {
            "held_pair_count": _gate(_finite(stats.get("held_pair_count")), float(thresholds["held_construct_matched_pair_count"]), op="min"),
            "availability_rate": _gate(_finite(stats.get("availability_rate")), float(thresholds["candidate_availability_rate"]), op="min"),
            "construct_matched_spearman": _gate(_finite(rho), float(thresholds["construct_matched_spearman_rho"]), op="min"),
            "candidate_iqr_points": _gate(_finite(stats.get("candidate_iqr")), float(thresholds["candidate_component_iqr_min_points"]), op="min"),
            "task_group_coverage": _gate(_finite(task_usable), float(thresholds["minimum_task_groups_with_usable_pairs"]), op="min"),
            "held_speaker_count": _gate(_finite(global_report.get("held_speaker_count")), float(thresholds["minimum_speakers"]), op="min"),
            "held_learner_speaker_count": _gate(_finite(global_report.get("held_learner_speaker_count")), float(thresholds["minimum_learner_speakers"]), op="min"),
            "held_native_speaker_count": _gate(_finite(global_report.get("held_native_speaker_count")), float(thresholds["minimum_native_speakers"]), op="min"),
            "channel_pair_coverage": "pass" if drift.get("usable_pair_count", 0) > 0 else "insufficient",
            "channel_drift_points": _gate(_finite(drift.get("median_absolute_drift")), float(thresholds["channel_paired_median_absolute_candidate_score_drift_max_points"]), op="max"),
            "negative_control_coverage": "pass" if global_report.get("negative_control_count", 0) > 0 else "insufficient",
            "negative_control_no_score_rate": _gate(_finite(global_report.get("negative_control_no_score_rate")), float(thresholds["negative_controls_no_score_rate_min"]), op="min"),
            "speaker_split_disjoint": "pass" if not global_report.get("speaker_split_leakage") else "fail",
            "gold_transcript_not_used": "pass" if global_report.get("gold_transcript_scoring_usage_count") == 0 else "fail",
        }
        if candidate == "intonation.shadow_candidate_score":
            if f0_safety.get("pass") is None:
                gate_states["f0_missingness_safety"] = "insufficient"
            else:
                gate_states["f0_missingness_safety"] = "pass" if f0_safety.get("pass") else "fail"

        report.update(
            {
                "channel_drift": drift,
                "gate_states": gate_states,
                "promotion_decision": _combine_gate_states(gate_states.values()),
                "promotion_scope": "separate_product_AB_only_with_score_contract_version_bump",
            }
        )
        candidate_reports[candidate] = report

    direct_decisions = [
        report.get("promotion_decision") for candidate, report in candidate_reports.items()
        if candidate in direct
    ]
    return {
        "schema": ANALYSIS_SCHEMA,
        "protocol_schema": protocol.get("schema_version"),
        "human_rating_rows": len(human_rows),
        "human_sample_criterion_count": len(human),
        "evidence_rows": len(evidence_rows),
        "global_dataset": global_report,
        "f0_missingness_safety": f0_safety,
        "candidates": candidate_reports,
        "overall_direct_promotion_state": _combine_gate_states(direct_decisions) if direct_decisions else "insufficient",
        "interpretation": "frozen-gate report only; no threshold fitting, score remapping, or automatic ProductScore promotion",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument(
        "--protocol",
        default="data/research_eval/free_speech_v5_promotion_protocol.json",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    report = analyze(args.human, args.evidence, args.protocol)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
