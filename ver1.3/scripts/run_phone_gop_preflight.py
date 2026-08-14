#!/usr/bin/env python3
"""Run the Stage-0 Japanese phone-GOP preflight using bundled audio only.

This script is deliberately designed to protect human time. It does not ask for
new recordings and it never changes ProductScore. The first run may explicitly
allow downloading the pinned HuBERT phone-CTC model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


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
from jp_speech_eval.phone_gop_preflight import build_phone_gop_preflight_report  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402


DEFAULT_WAV = ROOT / "assets" / "reference_cache" / "ramen_kudasai_aivis.ref.wav"
DEFAULT_MANIFEST = ROOT / "data" / "phone_gop_manual_validation_manifest_v2.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "phone_gop_preflight_v1.json"
CORRECT_TEXT = "ラーメンをください。"
WRONG_TEXT = "コーヒーをください。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-0 Japanese phone-GOP preflight; uses bundled audio only."
    )
    parser.add_argument("--wav", default=str(DEFAULT_WAV), help="Known Japanese reference WAV")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Pilot manifest for target inventory coverage")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSON report path")
    parser.add_argument("--allow-download", action="store_true", help="Allow download of the pinned HuBERT phone-CTC model")
    parser.add_argument("--device", default=None, help="torch device override")
    parser.add_argument("--wrong-text", default=WRONG_TEXT, help="Deliberately wrong target for same-audio ranking check")
    return parser.parse_args()


def _manifest_targets(path: Path) -> dict[str, object]:
    targets: dict[str, object] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            text = str(row.get("target_text") or "").strip()
            if not text:
                continue
            label = str(row.get("repeat_group") or row.get("clip_id") or text).strip()
            key = f"{label}:{text}"
            if key not in targets:
                targets[key] = build_japanese_target_evidence(text)
    return targets


def _gain_variant(audio: np.ndarray, gain: float) -> np.ndarray:
    # Avoid clipping while preserving a pure amplitude perturbation.
    y = np.asarray(audio, dtype=np.float32) * float(gain)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.999:
        y = y * (0.999 / peak)
    return y.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    audio = load_audio(str(wav_path), sr=16000)
    speech, region = trim_to_speech(audio.y, audio.sr)
    if speech.size == 0:
        raise RuntimeError("bundled preflight audio contains no detected speech")

    correct_target = build_japanese_target_evidence(CORRECT_TEXT)
    wrong_target = build_japanese_target_evidence(str(args.wrong_text))
    extra_targets = _manifest_targets(Path(args.manifest))

    backend = JapanesePhoneCtcBackend(
        device=args.device,
        local_files_only=not args.allow_download,
    )

    correct_result = backend.evaluate(speech, correct_target.phones, sr=audio.sr)
    wrong_result = backend.evaluate(speech, wrong_target.phones, sr=audio.sr)
    gain_results = {
        "gain_0p80": backend.evaluate(_gain_variant(speech, 0.80), correct_target.phones, sr=audio.sr),
        "gain_1p20": backend.evaluate(_gain_variant(speech, 1.20), correct_target.phones, sr=audio.sr),
    }

    report = build_phone_gop_preflight_report(
        backend=backend,
        correct_target=correct_target,
        wrong_target=wrong_target,
        extra_targets=extra_targets,
        correct_result=correct_result,
        wrong_result=wrong_result,
        gain_results=gain_results,
    )
    payload = report.to_dict()
    payload["runtime"] = {
        "wav": str(wav_path),
        "speech_region": region.to_dict(),
        "correct_text": CORRECT_TEXT,
        "wrong_text": str(args.wrong_text),
        "model_revision": DEFAULT_PHONE_CTC_REVISION,
        "model_download_allowed": bool(args.allow_download),
        "product_score_changed": False,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {output}")
    for check in report.checks:
        print(f"[{check.status.upper():5}] {check.name}: {check.detail}")
    print(
        "HUMAN RECORDING GATE: "
        + ("OPEN" if report.human_recording_allowed else "BLOCKED")
    )
    if not report.human_recording_allowed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
