#!/usr/bin/env python3
"""Run the frozen reliability-cap counterfactual over existing result files.

Input CSV columns:
  sample_id,wav_path,result_json[,sample_rate]

Paths are resolved relative to the CSV location.  This script never rewrites
product results and never chooses a winning policy; it only exports row-level
counterfactuals and a descriptive summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from jp_speech_eval.reliability_counterfactual import (
    rescore_without_reliability_caps,
    summarize_counterfactual_reports,
)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _flatten(sample_id: str, report: Dict[str, Any]) -> Dict[str, Any]:
    observed = report["observed_legacy_product_scores"]
    candidate = report["counterfactual_without_reliability_caps"]
    delta = report["counterfactual_minus_observed"]
    triggers = report["cap_triggers"]
    row: Dict[str, Any] = {"sample_id": sample_id, "available": bool(report.get("available"))}
    for key in ("pronunciation", "prosody", "fluency", "tone", "total"):
        row[f"observed_{key}"] = observed[key]
        row[f"cap_free_{key}"] = candidate[key]
        row[f"delta_{key}"] = delta[key]
    for key in (
        "alignment_equal_fallback",
        "mora_evidence_below_threshold",
        "f0_coverage_below_0_50",
        "overall_reliability_below_0_75",
    ):
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
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base = manifest_path.parent
    reports: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "wav_path", "result_json"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("manifest missing columns: " + ",".join(sorted(missing)))
        for row in reader:
            sample_id = str(row.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError("sample_id must not be empty")
            wav_path = _resolve(base, str(row.get("wav_path") or ""))
            result_path = _resolve(base, str(row.get("result_json") or ""))
            if not wav_path.exists():
                raise FileNotFoundError(wav_path)
            if not result_path.exists():
                raise FileNotFoundError(result_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            sample_rate = int(str(row.get("sample_rate") or "16000"))
            report = rescore_without_reliability_caps(
                result,
                wav_path=wav_path,
                scoring_config_path=scoring_config_path,
                sample_rate=sample_rate,
            )
            reports.append(report)
            flat_rows.append(_flatten(sample_id, report))
    return flat_rows, summarize_counterfactual_reports(reports)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--scoring-config", default=None)
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    rows, summary = run_manifest(
        manifest,
        scoring_config_path=None if args.scoring_config is None else Path(args.scoring_config).resolve(),
    )
    _write_csv(Path(args.out_csv), rows)
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
