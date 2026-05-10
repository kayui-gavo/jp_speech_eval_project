from __future__ import annotations

import argparse
import statistics

from jp_speech_eval.evaluator import evaluate_utterance


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sentence-final evaluation speed")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--alignment", default=None, choices=["cached_dtw", "dtw", "equal"])
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    alignment = args.alignment or ("cached_dtw" if args.cache else "dtw")
    totals = []
    last = None
    for _ in range(args.repeat):
        last = evaluate_utterance(
            text=args.text,
            wav_path=args.wav,
            cache_path=args.cache,
            alignment_mode=alignment,
        )
        totals.append(last.timing["total"])

    print("Benchmark")
    print(f"repeat: {args.repeat}")
    print(f"alignment: {alignment}")
    print(f"mean total: {statistics.mean(totals) * 1000:.2f} ms")
    print(f"min total : {min(totals) * 1000:.2f} ms")
    print(f"max total : {max(totals) * 1000:.2f} ms")
    if last:
        print("last timing:")
        for k, v in last.timing.items():
            print(f"  {k:16s}: {v * 1000:.2f} ms")


if __name__ == "__main__":
    main()
