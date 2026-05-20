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
    parser.add_argument("--reference-id", default=None, help="Optional stable id such as female_normal or neutral_teacher.")
    parser.add_argument("--tts-backend", default="pyopenjtalk", help="pyopenjtalk, voicevox_http, aivis_http, or google")
    parser.add_argument("--tts-url", default=None, help="Optional backend base URL, e.g. http://127.0.0.1:10101")
    parser.add_argument("--tts-speaker", type=int, default=None, help="VOICEVOX speaker id or AivisSpeech style id")
    parser.add_argument("--tts-model", default=None, help="Optional provider model id recorded in cache metadata.")
    parser.add_argument("--tts-voice", default=None, help="Optional provider voice id, e.g. ja-JP-Chirp3-HD-Achernar for Google.")
    parser.add_argument("--tts-speed", type=float, default=None, help="Optional requested speaking speed recorded in metadata and used when supported.")
    parser.add_argument("--tts-style", default=None, help="Optional style label recorded in metadata.")
    parser.add_argument("--tts-prompt", default=None, help="Optional style/instruction prompt recorded in metadata.")
    parser.add_argument("--tts-language", default="ja-JP", help="BCP-47 language tag recorded in metadata.")
    args = parser.parse_args()

    out = Path(args.out) if args.out else safe_cache_prefix(args.text)
    cache = build_sentence_cache(
        args.text,
        out,
        sr=args.sr,
        save_reference_wav=args.save_ref_wav,
        reference_wav_path=args.reference_wav,
        reference_source=args.reference_source,
        reference_id=args.reference_id,
        tts_backend=args.tts_backend,
        tts_backend_url=args.tts_url,
        tts_speaker=args.tts_speaker,
        tts_model=args.tts_model,
        tts_voice=args.tts_voice,
        tts_speed=args.tts_speed,
        tts_style=args.tts_style,
        tts_prompt=args.tts_prompt,
        tts_language=args.tts_language,
    )
    print("Prepared sentence cache")
    print(cache_summary(cache))
    print(f"JSON: {cache.prefix.with_suffix('.json')}")
    print(f"NPZ : {cache.prefix.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
