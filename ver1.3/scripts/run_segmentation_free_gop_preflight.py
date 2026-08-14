#!/usr/bin/env python3
"""Evaluate alignment-free FGOP-SF-SD-style features on bundled audio only."""

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
from jp_speech_eval.japanese_phoneme_gop import JapanesePhoneCtcBackend  # noqa: E402
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.segmentation_free_gop import evaluate_backend_fgop_sf_sd_shadow  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402


DEFAULT_WAV = ROOT / "assets" / "reference_cache" / "ramen_kudasai_aivis.ref.wav"
DEFAULT_OUTPUT = ROOT / "outputs" / "segmentation_free_gop_preflight_v1.json"
CORRECT_TEXT = "ラーメンをください。"
WRONG_TEXT = "コーヒーをください。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alignment-free Japanese phone feature preflight")
    parser.add_argument("--wav", default=str(DEFAULT_WAV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--wrong-text", default=WRONG_TEXT)
    return parser.parse_args()


def _weakest_rows(result, limit: int = 8):
    if not result.available:
        return []
    rows = sorted(
        result.evidence,
        key=lambda row: row.best_noncanonical_log_posterior_ratio,
    )
    return [
        {
            "phone_index": row.phone_index,
            "phone": row.canonical_phone,
            "best_noncanonical_type": row.best_noncanonical_alternative_type,
            "best_noncanonical_phone": row.best_noncanonical_alternative_phone,
            "best_noncanonical_lpr": row.best_noncanonical_log_posterior_ratio,
            "deletion_lpr": row.deletion_log_posterior_ratio,
            "gop_sf_sd": row.gop_sf_sd,
        }
        for row in rows[:limit]
    ]


def main() -> None:
    args = parse_args()
    audio = load_audio(str(Path(args.wav)), sr=16000)
    speech, region = trim_to_speech(audio.y, audio.sr)
    correct_target = build_japanese_target_evidence(CORRECT_TEXT)
    wrong_target = build_japanese_target_evidence(str(args.wrong_text))
    backend = JapanesePhoneCtcBackend(
        device=args.device,
        local_files_only=not args.allow_download,
    )

    correct = evaluate_backend_fgop_sf_sd_shadow(
        backend, speech, correct_target.phones, sr=audio.sr
    )
    wrong = evaluate_backend_fgop_sf_sd_shadow(
        backend, speech, wrong_target.phones, sr=audio.sr
    )

    payload = {
        "schema": "segmentation_free_gop_bundled_preflight_v1",
        "product_score_changed": False,
        "score_mapped": False,
        "human_recording_allowed": False,
        "speech_region": region.to_dict(),
        "correct_target": correct_target.to_dict(),
        "wrong_target": wrong_target.to_dict(),
        "correct": correct.to_dict(),
        "wrong": wrong.to_dict(),
        "diagnostic": {
            "correct_weakest_rows": _weakest_rows(correct),
            "wrong_weakest_rows": _weakest_rows(wrong),
            "forced_viterbi_not_required_for_features": True,
            "note": (
                "This artifact evaluates enumerated LPP/LPR substitution+deletion features. "
                "It is not a /100 score and does not implement Occ(i) normalization or the "
                "paper's optimized alternative graph."
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    print("correct available:", correct.available)
    if correct.available:
        print("correct phones where noncanonical SD alternative wins:", correct.summary["phones_where_noncanonical_outscores_canonical"])
        for row in _weakest_rows(correct, limit=5):
            print(row)
    print("HUMAN RECORDING GATE: BLOCKED")
    if not correct.available or not wrong.available:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
