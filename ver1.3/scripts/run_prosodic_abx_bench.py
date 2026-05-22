#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from select_prosody_layer import run as run_layer_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small Prosodic ABX bench for pitch-accent diagnostics. "
            "Use this to test whether a representation can distinguish minimal pairs before using it for scoring."
        )
    )
    parser.add_argument("--dataset", default="data/prosody_minimal_pairs.json")
    parser.add_argument("--audio-root", default="data/prosody_audio")
    parser.add_argument("--models", nargs="+", default=["mfcc"])
    parser.add_argument("--layers", default="all")
    parser.add_argument("--mode", choices=["in_context", "out_of_context"], default="in_context")
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--results-dir", default="outputs/prosodic_abx_bench")
    parser.add_argument("--bootstrap-tts", action="store_true", help="Only for pipeline sanity checks; not scientific evidence.")
    parser.add_argument("--tts-backend", default="pyopenjtalk")
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-trials-for-confidence", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_layer_selection(args)
    summary["bench_note"] = (
        "ABX accuracy/margin are representation diagnostics. "
        "Bootstrap TTS trials are smoke tests only; use native verified audio for model/layer claims."
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
