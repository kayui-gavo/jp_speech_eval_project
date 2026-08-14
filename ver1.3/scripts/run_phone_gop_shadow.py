#!/usr/bin/env python3
"""Run the hardened Japanese phone-CTC GOP shadow on one utterance.

This script is research/preflight-only. It does not modify ProductScore and it
does not map GOP evidence to /100.

By default the pinned model must already exist in the local Hugging Face cache.
Use ``--allow-download`` only when intentionally provisioning the research
backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.audio_features import load_audio  # noqa: E402
from jp_speech_eval.japanese_phoneme_gop import (  # noqa: E402
    DEFAULT_PHONE_CTC_REVISION,
    JapanesePhoneCtcBackend,
)
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.phoneme_gop import DEFAULT_PHONE_CTC_MODEL  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Japanese phone-CTC GOP shadow; no product score mapping."
    )
    parser.add_argument("--wav", required=True, help="Learner/user WAV path")
    parser.add_argument("--text", required=True, help="Known or user-confirmed Japanese transcript")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_PHONE_CTC_MODEL,
        help="Hugging Face phone-CTC model id",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_PHONE_CTC_REVISION,
        help="Pinned Hugging Face model revision",
    )
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow the pinned Hugging Face model to download. Default is local-files-only.",
    )
    parser.add_argument("--device", default=None, help="torch device override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    audio = load_audio(str(wav_path), sr=16000)
    speech, region = trim_to_speech(audio.y, audio.sr)
    target = build_japanese_target_evidence(args.text)
    backend = JapanesePhoneCtcBackend(
        model_id=args.model_id,
        revision=args.revision,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    result = backend.evaluate(speech, target.phones, sr=audio.sr)

    payload = {
        "schema": "japanese_phone_gop_shadow_v2",
        "human_recording_gate": "blocked_until_automated_preflight_passes",
        "product_score_changed": False,
        "score_mapped": False,
        "product_calibrated": False,
        "target": {
            "text": target.surface_text,
            "kana": target.reading_kana,
            "phones": target.phones,
            "reading_source": target.reading_source,
        },
        "backend": {
            "model_id": args.model_id,
            "revision": args.revision,
            "local_files_only": not args.allow_download,
        },
        "speech_region": region.to_dict(),
        "gop": result.to_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    if result.available:
        summary = result.summary
        print(
            "Japanese phone GOP shadow: "
            f"n={summary.get('supported_phone_count')} "
            f"posterior_median={summary.get('posterior_gop_margin', {}).get('median')} "
            f"logit_median={summary.get('mean_logit_margin', {}).get('median')} "
            f"greedy_edit={summary.get('greedy_phone_edit_distance')}"
        )
    else:
        print(f"phone GOP unavailable: {result.summary.get('reason')}")


if __name__ == "__main__":
    main()
