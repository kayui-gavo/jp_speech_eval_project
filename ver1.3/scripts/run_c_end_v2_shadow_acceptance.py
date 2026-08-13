"""Exercise opt-in C-end shadow analyzers on a small real-audio panel."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


DEFAULT_PANEL = [
    "native_jvs001_s001",
    "native_jvs002_s001",
    "learner_chf1_i74",
    "learner_chf1_i63",
    "learner_enf1_i77",
    "learner_chf1_i60",
    "mismatch_jvs_s002_as_ramen",
]


def load_inventory(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def evaluate_one(sample: dict[str, str], *, enable_ssl: bool) -> dict:
    from jp_speech_eval.api import EvaluationRequest, SpeechEvalConfig, SpeechEvaluationClient

    config = SpeechEvalConfig(
        enable_ssl_shadow=enable_ssl,
        enable_special_mora_v2_shadow=not enable_ssl,
        enable_phrase_intonation_shadow=not enable_ssl,
        enable_accent_nucleus_shadow=not enable_ssl,
        ssl_shadow_timeout_sec=180.0,
    )
    started = time.perf_counter()
    response = SpeechEvaluationClient(config).evaluate(
        EvaluationRequest(
            audio_path=sample["wav_path"],
            mode=sample["mode"],
            target_text=sample["target_text"] or None,
            transcript=sample["transcript"] or None,
            user_confirmed_text=sample["user_confirmed_text"] or None,
            cache_path=sample["cache_path"] or None,
        )
    ).to_dict()
    return {
        "sample": sample,
        "enable_ssl": enable_ssl,
        "wall_latency_sec": round(time.perf_counter() - started, 6),
        "response": response,
    }


def pitch_shift_case(sample: dict[str, str], output_dir: Path) -> dict:
    import librosa
    import soundfile as sf

    source = Path(sample["wav_path"])
    audio, sr = librosa.load(source, sr=None, mono=True)
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=3.0)
    wav = output_dir / "pitch_shift_plus3_native_jvs001_s001.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav, shifted, sr, subtype="PCM_16")
    shifted_sample = dict(sample)
    shifted_sample["sample_id"] = "pitch_shift_plus3_native_jvs001_s001"
    shifted_sample["wav_path"] = str(wav.resolve())
    shifted_sample["expected_category"] = "phrase_invariance_control"
    shifted_sample["notes"] = "Offline +3 semitone transform; audit-only invariance check"
    return evaluate_one(shifted_sample, enable_ssl=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ssl", action="store_true")
    parser.add_argument("--pitch-shift", action="store_true")
    parser.add_argument("--sample", action="append", default=[])
    args = parser.parse_args()
    inventory = load_inventory(args.inventory)
    panel = args.sample or DEFAULT_PANEL
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_id in panel:
        record = evaluate_one(inventory[sample_id], enable_ssl=args.ssl)
        records.append(record)
        shadow = (record["response"].get("raw_result") or {}).get("details", {}).get("shadow", {})
        print(sample_id, record["response"].get("ok"), record["wall_latency_sec"], list(shadow))
    if args.pitch_shift and not args.ssl:
        record = pitch_shift_case(inventory["native_jvs001_s001"], args.output.parent)
        records.append(record)
        print(record["sample"]["sample_id"], record["response"].get("ok"), record["wall_latency_sec"])
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
