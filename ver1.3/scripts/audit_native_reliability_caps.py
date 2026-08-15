#!/usr/bin/env python3
"""Summarize evaluator-native pre/post reliability-cap telemetry.

Input may be a single raw-result JSON or a C-end JSONL whose rows contain
``response.raw_result``. Historical rows without native telemetry are reported
as unavailable rather than replayed with a newer scorer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from jp_speech_eval.native_cap_telemetry import (
    has_native_cap_telemetry,
    report_from_native_cap_telemetry,
)
from jp_speech_eval.reliability_counterfactual import summarize_counterfactual_reports


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _iter_results(path: Path) -> Iterable[tuple[str, Mapping[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample = _mapping(row.get("sample"))
                sample_id = str(sample.get("sample_id") or f"row_{index}")
                response = _mapping(row.get("response"))
                result = _mapping(response.get("raw_result")) or _mapping(row.get("raw_result"))
                yield sample_id, result
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = _mapping(payload.get("raw_result")) or _mapping(payload)
    yield path.stem, result


def audit(path: Path) -> Dict[str, Any]:
    reports = []
    native_count = 0
    missing_ids = []
    for sample_id, result in _iter_results(path):
        if result and has_native_cap_telemetry(result):
            reports.append(report_from_native_cap_telemetry(result))
            native_count += 1
        else:
            missing_ids.append(sample_id)
            reports.append(
                {
                    "available": False,
                    "counterfactual_trustworthy": False,
                    "counterfactual_trust_level": "native_telemetry_missing",
                    "cap_triggers": {"applicable": False},
                }
            )
    summary = summarize_counterfactual_reports(reports)
    summary.update(
        {
            "input": str(path),
            "native_telemetry_count": native_count,
            "native_telemetry_missing_count": len(missing_ids),
            "native_telemetry_missing_sample_ids": missing_ids,
            "interpretation": "exact native telemetry only; no historical scorer replay performed",
        }
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    summary = audit(Path(args.input).resolve())
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
