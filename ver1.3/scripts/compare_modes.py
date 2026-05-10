from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.eval_modes import evaluate_mode
from jp_speech_eval.evaluation_log import append_jsonl, export_feature_table
from jp_speech_eval.unified_result import unify_evaluation_result


def _read_manifest(path: str | Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("audio_path"):
                rows.append(dict(row))
    return rows


def _wav_rows(wavs: Iterable[str]) -> List[Dict[str, str]]:
    return [{"audio_path": wav} for wav in wavs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the same wavs through multiple evaluation modes.")
    parser.add_argument("--wav", nargs="*", default=[], help="Wav files to evaluate.")
    parser.add_argument("--manifest", default=None, help="Optional CSV with audio_path,target_text,transcript columns.")
    parser.add_argument("--cache", default="cache/ramen_kudasai", help="Base cache for reference/asr_pseudo_reference.")
    parser.add_argument("--target-text", default=None, help="Fallback target text for reference mode without manifest target_text.")
    parser.add_argument("--modes", default="reference,acoustic,transcript_assisted_light,asr_pseudo_reference")
    parser.add_argument("--jsonl", default="outputs/eval_log.jsonl")
    parser.add_argument("--csv", default=None, help="Optional feature-table CSV exported from the JSONL after comparison.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    rows = _read_manifest(args.manifest) if args.manifest else _wav_rows(args.wav)
    if not rows:
        parser.error("Provide --wav or --manifest.")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    summary: List[Dict[str, object]] = []
    for row in rows:
        wav_path = row["audio_path"]
        target_text = row.get("target_text") or args.target_text
        transcript = row.get("transcript") or None
        for mode in modes:
            try:
                result = evaluate_mode(
                    mode,
                    wav_path,
                    cache_path=args.cache if mode in {"reference", "reference_based", "reference_fixed_sentence", "asr_pseudo_reference"} else None,
                    target_text=target_text,
                    transcript=transcript,
                    scoring_config_path=args.config,
                    sample_rate=args.sr,
                )
                unified = unify_evaluation_result(
                    result,
                    mode=result.get("details", {}).get("mode") or mode,
                    audio_path=wav_path,
                    target_text=target_text,
                    extra_input={"manifest_target_text": target_text, "manifest_transcript": transcript},
                )
                append_jsonl(args.jsonl, unified)
                summary.append({
                    "audio_path": wav_path,
                    "mode": unified.mode,
                    "total": unified.scores.get("total"),
                    "pronunciation": unified.scores.get("pronunciation"),
                    "prosody": unified.scores.get("prosody"),
                    "fluency": unified.scores.get("fluency"),
                    "reliability": unified.reliability.get("overall"),
                    "latency_ms": unified.latency_ms,
                    "warnings": len(unified.warnings),
                })
            except Exception as exc:
                summary.append({
                    "audio_path": wav_path,
                    "mode": mode,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved JSONL: {args.jsonl}")
    if args.csv:
        count = export_feature_table([args.jsonl], args.csv)
        print(f"Exported {count} rows to {args.csv}")


if __name__ == "__main__":
    main()
