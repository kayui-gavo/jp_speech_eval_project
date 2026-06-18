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

from jp_speech_eval.prosody_reference_cache import (  # noqa: E402
    build_prosody_reference_cache_payload,
    prosody_reference_cache_path,
    write_prosody_reference_cache,
)
from jp_speech_eval.sentence_cache import load_sentence_cache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a sidecar mora-level F0 cache for a verified reference audio target."
    )
    parser.add_argument("--cache", required=True, help="Sentence cache prefix, e.g. cache/ramen_kudasai")
    parser.add_argument("--out", default=None, help="Output JSON path. Defaults to <cache>.prosody_ref.json")
    parser.add_argument(
        "--reference-audio",
        default=None,
        help="Optional provenance path stored in the sidecar. If omitted, <cache>.ref.wav is used when present.",
    )
    parser.add_argument(
        "--verified-reference",
        action="store_true",
        help="Mark this reference source as verified human/native/teacher audio when provenance has been checked.",
    )
    parser.add_argument("--min-f0-coverage", type=float, default=0.50)
    parser.add_argument("--dry-run", action="store_true", help="Print the payload without writing the sidecar.")
    args = parser.parse_args()

    cache = load_sentence_cache(args.cache)
    payload = build_prosody_reference_cache_payload(
        cache,
        reference_audio_path=args.reference_audio,
        verified_reference=args.verified_reference,
        min_f0_coverage=args.min_f0_coverage,
    )
    out = Path(args.out) if args.out else prosody_reference_cache_path(cache.prefix)
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    path = write_prosody_reference_cache(
        cache,
        out_path=out,
        reference_audio_path=args.reference_audio,
        verified_reference=args.verified_reference,
        min_f0_coverage=args.min_f0_coverage,
    )
    print(f"wrote: {path}")
    print(f"reliable: {payload['reliable']}")
    print(f"pitch_target_source: {payload['pitch_target_source']}")
    print(f"f0_coverage: {payload['f0_coverage']}")
    print(f"quality_flags: {', '.join(payload['quality_flags']) or '-'}")


if __name__ == "__main__":
    main()
