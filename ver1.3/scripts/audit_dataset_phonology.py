#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.calibration_bench import (
    append_jsonl,
    evaluation_to_audit_row,
    read_manifest,
    summarize_audit_rows,
    write_csv,
)
from jp_speech_eval.config import load_scoring_config
from jp_speech_eval.evaluator import evaluate_utterance


def _resolve_path(base_dir: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _evaluate_row(
    row: Dict[str, str],
    *,
    manifest_dir: Path,
    scoring_config_path: str | None,
    sample_rate: int,
    prefer_cache: bool,
) -> Dict[str, Any]:
    audio_path = _resolve_path(manifest_dir, row.get("audio_path"))
    cache_path = _resolve_path(manifest_dir, row.get("cache_path")) if prefer_cache else None
    if not audio_path:
        raise ValueError("audio_path is required")
    if cache_path:
        result = evaluate_utterance(
            wav_path=audio_path,
            cache_path=cache_path,
            alignment_mode="cached_dtw",
            scoring_config_path=scoring_config_path,
            sample_rate=sample_rate,
            use_content_match=True,
        )
    else:
        text = row.get("text")
        if not text:
            raise ValueError("text is required when cache_path is absent")
        result = evaluate_utterance(
            text=text,
            wav_path=audio_path,
            alignment_mode="equal",
            scoring_config_path=scoring_config_path,
            sample_rate=sample_rate,
            use_content_match=False,
        )
    return evaluation_to_audit_row({**row, "audio_path": audio_path, "cache_path": cache_path or ""}, result)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest_dir = manifest_path.parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest_path)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("Manifest has no usable rows. Required: audio_path and text or cache_path.")

    per_rows: List[Dict[str, Any]] = []
    errors_path = out_dir / "errors.jsonl"
    if errors_path.exists():
        errors_path.unlink()

    for index, row in enumerate(rows, start=1):
        try:
            per_rows.append(
                _evaluate_row(
                    row,
                    manifest_dir=manifest_dir,
                    scoring_config_path=args.config,
                    sample_rate=args.sr,
                    prefer_cache=not args.no_cache,
                )
            )
        except Exception as exc:
            append_jsonl(
                errors_path,
                {
                    "row_index": index,
                    "audio_path": row.get("audio_path"),
                    "text": row.get("text"),
                    "dataset": row.get("dataset"),
                    "split": row.get("split"),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

    summary_rows = summarize_audit_rows(per_rows)
    per_csv = out_dir / "phonology_audit_per_utterance.csv"
    summary_csv = out_dir / "phonology_audit_summary.csv"
    write_csv(per_csv, per_rows)
    write_csv(summary_csv, summary_rows)
    config_snapshot = {
        "scoring_config_path": args.config,
        "effective_thresholds": {
            "fluency": load_scoring_config(args.config).get("fluency", {}),
            "pronunciation": load_scoring_config(args.config).get("pronunciation", {}),
            "prosody": load_scoring_config(args.config).get("prosody", {}),
        },
        "manifest": str(manifest_path),
        "n_manifest_rows": len(rows),
        "n_success": len(per_rows),
        "n_errors": sum(1 for _ in errors_path.open("r", encoding="utf-8")) if errors_path.exists() else 0,
        "interpretation": (
            "Use native rows to find false penalties and L2 rows to find common learner deviations. "
            "Do not treat these statistics as final labels without human/listener validation."
        ),
    }
    snapshot_path = out_dir / "phonology_audit_config.json"
    snapshot_path.write_text(json.dumps(config_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "per_utterance_csv": str(per_csv),
        "summary_csv": str(summary_csv),
        "errors_jsonl": str(errors_path),
        "config_json": str(snapshot_path),
        "n_success": len(per_rows),
        "n_errors": config_snapshot["n_errors"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit pronunciation/prosody proxy behavior on datasets such as JVS and JANON. "
            "This calibrates and diagnoses thresholds; it does not train a model."
        )
    )
    parser.add_argument("--manifest", required=True, help="CSV: audio_path,text,dataset,split,speaker_id[,cache_path]")
    parser.add_argument("--out-dir", default="outputs/calibration_bench")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true", help="Ignore manifest cache_path and use text/equal alignment")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
