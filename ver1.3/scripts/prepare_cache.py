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
    parser.add_argument(
        "--reference-wav",
        default=None,
        help="Optional externally generated pseudo-reference wav for the same text.",
    )
    parser.add_argument(
        "--reference-source",
        default=None,
        help="Optional provenance label stored in cache metadata, e.g. kanade_voice_conditioned_pseudo_reference.",
    )
    parser.add_argument("--tts-backend", default="pyopenjtalk", help="pyopenjtalk, voicevox_http, or aivis_http")
    parser.add_argument("--tts-url", default=None, help="Optional backend base URL, e.g. http://127.0.0.1:10101")
    parser.add_argument("--tts-speaker", type=int, default=None, help="VOICEVOX speaker id or AivisSpeech style id")
    args = parser.parse_args()

    out = Path(args.out) if args.out else safe_cache_prefix(args.text)
    cache = build_sentence_cache(
        args.text,
        out,
        sr=args.sr,
        save_reference_wav=args.save_ref_wav,
        reference_wav_path=args.reference_wav,
        reference_source=args.reference_source,
        tts_backend=args.tts_backend,
        tts_backend_url=args.tts_url,
        tts_speaker=args.tts_speaker,
    )
    print("Prepared sentence cache")
    print(cache_summary(cache))
    print(f"JSON: {cache.prefix.with_suffix('.json')}")
    print(f"NPZ : {cache.prefix.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
