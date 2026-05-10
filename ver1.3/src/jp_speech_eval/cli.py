from __future__ import annotations

import argparse
from pathlib import Path

from .evaluator import evaluate_utterance, plot_evaluation, print_result
from .evaluation_log import append_jsonl
from .sentence_cache import load_sentence_cache
from .unified_result import unify_evaluation_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Japanese speech evaluation: cached sentence-final scoring")
    parser.add_argument("--text", default=None, help="Target Japanese sentence, e.g. ラーメンをください")
    parser.add_argument("--cache", default=None, help="Cache prefix created by scripts/prepare_cache.py")
    parser.add_argument("--wav", required=True, help="Path to user wav file")
    parser.add_argument(
        "--alignment",
        default=None,
        choices=["cached_dtw", "dtw", "equal"],
        help="Mora alignment mode. Default: cached_dtw if --cache is given, otherwise dtw.",
    )
    parser.add_argument("--json", default=None, help="Optional output json path")
    parser.add_argument("--plot", default=None, help="Optional output png path")
    parser.add_argument("--no-plot", action="store_true", help="Ignore --plot and skip matplotlib drawing")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate for analysis when no cache is used")
    parser.add_argument("--config", default=None, help="Optional scoring config json path")
    parser.add_argument("--profile", action="store_true", help="Print per-module timing")
    parser.add_argument("--log-jsonl", default=None, help="Append a unified evaluation log record for model training.")
    args = parser.parse_args()

    if args.cache is None and args.text is None:
        parser.error("Either --text or --cache must be provided.")

    alignment = args.alignment or ("cached_dtw" if args.cache else "dtw")

    result = evaluate_utterance(
        text=args.text,
        wav_path=args.wav,
        alignment_mode=alignment,
        sample_rate=args.sr,
        cache_path=args.cache,
        scoring_config_path=args.config,
        profile=args.profile,
    )
    print_result(result)

    if args.json:
        result.save_json(args.json)
        print(f"\nSaved JSON: {args.json}")

    if args.log_jsonl:
        unified = unify_evaluation_result(
            result,
            mode="reference_based",
            audio_path=args.wav,
            target_text=args.text,
        )
        append_jsonl(args.log_jsonl, unified)
        print(f"Appended JSONL log: {args.log_jsonl}")

    if args.plot and not args.no_plot:
        sample_rate = args.sr
        if args.cache:
            sample_rate = load_sentence_cache(args.cache).meta.sr
        plot_evaluation(result, args.wav, args.plot, sample_rate=sample_rate)
        print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
