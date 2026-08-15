#!/usr/bin/env python3
"""Audit legacy reliability caps over current or historical evaluator results.

Supported input shapes:

1. CSV manifest: ``sample_id,wav_path,result_json[,sample_rate]``
2. Existing C-end acceptance JSONL: each row contains ``sample`` and
   ``response.raw_result``.

Historical replay is deliberately conservative. The script exports whether
current scorer replay reproduces the stored post-cap score and whether the
historical pre-cap score is identifiable. A raw current-scorer delta is never
silently labelled as a historical cap effect.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from jp_speech_eval.reliability_counterfactual import (
    reliability_cap_triggers,
    rescore_without_reliability_caps,
    summarize_counterfactual_reports,
)


SCORE_KEYS = ("pronunciation", "prosody", "fluency", "tone", "total")
TRIGGER_KEYS = (
    "alignment_equal_fallback",
    "mora_evidence_below_threshold",
    "f0_coverage_below_0_50",
    "overall_reliability_below_0_75",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _flatten(sample_id: str, report: Mapping[str, Any], *, category: str = "", mode: str = "") -> Dict[str, Any]:
    triggers = _mapping(report.get("cap_triggers"))
    consistency = _mapping(report.get("replay_consistency"))
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "category": category,
        "mode": mode,
        "applicable": bool(triggers.get("applicable")),
        "formula_replay_available": bool(report.get("available")),
        "availability_reason": report.get("availability_reason"),
        "counterfactual_trust_level": report.get("counterfactual_trust_level"),
        "counterfactual_trustworthy": report.get("counterfactual_trustworthy"),
        "historical_replay_compatible": consistency.get("historical_replay_compatible"),
        "weighted_components_compatible": consistency.get("weighted_components_compatible"),
        "total_formula_matches": consistency.get("total_formula_matches"),
        "source_wav_available": report.get("source_wav_available"),
        "tone_replayed": report.get("tone_replayed"),
    }
    observed = _mapping(report.get("observed_legacy_evaluator_scores"))
    candidate = _mapping(report.get("candidate_pre_cap_scores_from_current_scorer"))
    delta = _mapping(report.get("candidate_pre_cap_minus_observed"))
    expected = _mapping(report.get("expected_post_cap_scores_from_replay"))
    for key in SCORE_KEYS:
        row[f"observed_{key}"] = observed.get(key)
        row[f"candidate_pre_cap_{key}"] = candidate.get(key)
        row[f"candidate_delta_{key}"] = delta.get(key)
        row[f"expected_post_cap_{key}"] = expected.get(key)
        item = _mapping(consistency.get(key))
        row[f"replay_checked_{key}"] = item.get("checked")
        row[f"replay_consistent_{key}"] = item.get("consistent")
        row[f"replay_identifiability_{key}"] = item.get("identifiability")
    row["expected_pre_overall_cap_total"] = expected.get("pre_overall_cap_total")
    for key in TRIGGER_KEYS:
        row[f"trigger_{key}"] = bool(triggers.get(key))
    row["judgement_count"] = triggers.get("judgement_count")
    row["judgement_needed"] = triggers.get("judgement_needed")
    row["f0_coverage"] = triggers.get("f0_coverage")
    row["overall_reliability"] = triggers.get("overall_reliability")
    return row


def run_manifest(
    manifest_path: Path,
    *,
    scoring_config_path: Path | None = None,
    same_run: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base = manifest_path.parent
    reports: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "result_json"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("manifest missing columns: " + ",".join(sorted(missing)))
        for row in reader:
            sample_id = str(row.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError("sample_id must not be empty")
            result_path = _resolve(base, str(row.get("result_json") or ""))
            if not result_path.exists():
                raise FileNotFoundError(result_path)
            wav_text = str(row.get("wav_path") or "").strip()
            wav_path = _resolve(base, wav_text) if wav_text else None
            if wav_path is not None and not wav_path.is_file():
                wav_path = None
            result = json.loads(result_path.read_text(encoding="utf-8"))
            sample_rate = int(str(row.get("sample_rate") or "16000"))
            report = rescore_without_reliability_caps(
                result,
                wav_path=wav_path,
                scoring_config_path=scoring_config_path,
                sample_rate=sample_rate,
                same_run=same_run,
            )
            reports.append(report)
            flat_rows.append(_flatten(sample_id, report, category=str(row.get("category") or ""), mode=str(row.get("mode") or "")))
    return flat_rows, summarize_counterfactual_reports(reports)


def _iter_acceptance_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            yield record


def run_acceptance_jsonl(
    results_path: Path,
    *,
    scoring_config_path: Path | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []
    for index, record in enumerate(_iter_acceptance_records(results_path), start=1):
        sample = _mapping(record.get("sample"))
        response = _mapping(record.get("response"))
        result = _mapping(response.get("raw_result"))
        sample_id = str(sample.get("sample_id") or f"row_{index}")
        category = str(sample.get("expected_category") or "")
        mode = str(response.get("mode") or sample.get("mode") or "")
        if not result:
            triggers = {"applicable": False, **{key: False for key in TRIGGER_KEYS}}
            report: Dict[str, Any] = {
                "available": False,
                "availability_reason": "missing_raw_result",
                "cap_triggers": triggers,
                "product_behavior_changed": False,
                "user_facing": False,
            }
        else:
            wav_text = str(sample.get("wav_path") or "").strip()
            wav_path = Path(wav_text) if wav_text else None
            if wav_path is not None and not wav_path.is_file():
                wav_path = None
            try:
                report = rescore_without_reliability_caps(
                    result,
                    wav_path=wav_path,
                    scoring_config_path=scoring_config_path,
                    sample_rate=16000,
                    same_run=False,
                )
            except ValueError as exc:
                report = {
                    "available": False,
                    "availability_reason": f"replay_unavailable:{type(exc).__name__}:{exc}",
                    "cap_triggers": reliability_cap_triggers(result),
                    "product_behavior_changed": False,
                    "user_facing": False,
                }
        reports.append(report)
        flat_rows.append(_flatten(sample_id, report, category=category, mode=mode))

    summary = summarize_counterfactual_reports(reports)
    summary["input"] = str(results_path)
    summary["input_format"] = "c_end_acceptance_jsonl_historical"
    summary["source_wav_available_count"] = sum(bool(row.get("source_wav_available")) for row in flat_rows)
    categories: Dict[str, int] = {}
    for row in flat_rows:
        category = str(row.get("category") or "unknown")
        categories[category] = categories.get(category, 0) + 1
    summary["category_counts"] = dict(sorted(categories.items()))
    return flat_rows, summary


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="CSV manifest or historical C-end acceptance JSONL")
    parser.add_argument("--input-format", choices=["auto", "manifest", "acceptance-jsonl"], default="auto")
    parser.add_argument("--same-run", action="store_true", help="Manifest results were produced by the same scorer/config revision as this audit")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--scoring-config", default=None)
    args = parser.parse_args()

    source = Path(args.input).resolve()
    fmt = args.input_format
    if fmt == "auto":
        fmt = "acceptance-jsonl" if source.suffix.lower() == ".jsonl" else "manifest"
    config_path = None if args.scoring_config is None else Path(args.scoring_config).resolve()
    if fmt == "acceptance-jsonl":
        if args.same_run:
            raise ValueError("--same-run is not allowed for historical acceptance JSONL")
        rows, summary = run_acceptance_jsonl(source, scoring_config_path=config_path)
    else:
        rows, summary = run_manifest(source, scoring_config_path=config_path, same_run=args.same_run)

    _write_csv(Path(args.out_csv), rows)
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
