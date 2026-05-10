from __future__ import annotations

import argparse
from pathlib import Path

from jp_speech_eval.sentence_cache import build_sentence_cache, cache_summary, safe_cache_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cache for a target Japanese sentence")
    parser.add_argument("--text", required=True, help="Target Japanese sentence")
    parser.add_argument("--out", default=None, help="Output prefix, e.g. cache/ramen_kudasai")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--save-ref-wav", action="store_true", help="Also save TTS reference wav")
    args = parser.parse_args()

    out = Path(args.out) if args.out else safe_cache_prefix(args.text)
    cache = build_sentence_cache(args.text, out, sr=args.sr, save_reference_wav=args.save_ref_wav)
    print("Prepared sentence cache")
    print(cache_summary(cache))
    print(f"JSON: {cache.prefix.with_suffix('.json')}")
    print(f"NPZ : {cache.prefix.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
