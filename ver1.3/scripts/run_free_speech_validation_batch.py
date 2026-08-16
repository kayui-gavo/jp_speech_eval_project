#!/usr/bin/env python3
"""Run the real free-speech C-end path over a validation manifest.

The runner deliberately passes no gold/manual transcript into the evaluator.
It records raw evidence plus the current product score policy so validation uses
real product conditions rather than an oracle-transcript research shortcut.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from jp_speech_eval.transcript_assisted import evaluate_transcript_assisted_light
from jp_speech_eval.user_score_policy import apply_user_score_policy
from validate_free_speech_sample_manifest import validate_manifest_file


RUN_SCHEMA = "free_speech_validation_batch_v1"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _read_manifest(path: str | Path) -> list[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok" and _text(row.get("sample_id")):
                completed.add(_text(row.get("sample_id")))
    return completed


def run_batch(
    manifest_csv: str | Path,
    output_jsonl: str | Path,
    *,
    audio_root: str | Path | None = None,
    asr_model: str = "small",
    asr_provider: str = "auto",
    resume: bool = True,
    evaluator: Callable[..., Mapping[str, Any]] = evaluate_transcript_assisted_light,
    user_policy: Callable[..., Mapping[str, Any]] = apply_user_score_policy,
) -> Dict[str, Any]:
    validation = validate_manifest_file(manifest_csv)
    if not validation.get("ok"):
        raise ValueError("sample manifest failed validation: " + ";".join(validation.get("errors") or []))

    rows = _read_manifest(manifest_csv)
    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(output) if resume else set()
    root = Path(audio_root) if audio_root else Path(manifest_csv).parent

    written = 0
    skipped = 0
    failed = 0
    mode = "a" if resume and output.exists() else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for row in rows:
            sample_id = _text(row.get("sample_id"))
            if sample_id in completed:
                skipped += 1
                continue
            source_path = Path(_text(row.get("audio_path")))
            wav_path = source_path if source_path.is_absolute() else root / source_path
            payload: Dict[str, Any] = {
                "run_schema": RUN_SCHEMA,
                "sample_id": sample_id,
                "metadata": {
                    key: _text(row.get(key))
                    for key in (
                        "speaker_id", "speaker_group", "l1", "task_mode", "prompt_id", "split",
                        "expected_language", "channel_condition", "channel_pair_id", "source_recording_id",
                        "context_type", "context_id", "source_note",
                    )
                },
                "audio_path": _text(row.get("audio_path")),
                "scoring_used_gold_transcript": False,
            }
            try:
                # Critical product-validity rule: transcript is always None here.
                raw = dict(
                    evaluator(
                        wav_path,
                        transcript=None,
                        asr_model=asr_model,
                        asr_provider=asr_provider,
                    )
                )
                product = dict(user_policy(raw, mode="transcript_assisted_light"))
                payload.update(
                    {
                        "status": "ok",
                        "raw_result": raw,
                        "user_score": product,
                    }
                )
                written += 1
            except Exception as exc:  # batch collection must preserve per-sample failures
                payload.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                failed += 1
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    return {
        "schema": RUN_SCHEMA,
        "manifest": str(manifest_csv),
        "output": str(output),
        "manifest_row_count": len(rows),
        "written_ok": written,
        "skipped_completed": skipped,
        "failed": failed,
        "scoring_used_gold_transcript": False,
        "asr_model": asr_model,
        "asr_provider": asr_provider,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--asr-provider", default="auto")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    report = run_batch(
        args.manifest_csv,
        args.out,
        audio_root=args.audio_root,
        asr_model=args.asr_model,
        asr_provider=args.asr_provider,
        resume=not args.no_resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
