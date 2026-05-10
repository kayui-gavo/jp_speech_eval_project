from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import librosa
import numpy as np

from jp_speech_eval.audio_features import load_audio
from jp_speech_eval.config import load_scoring_config
from jp_speech_eval.realtime_evaluator import RealtimeEvaluator
from jp_speech_eval.sentence_cache import load_sentence_cache
from jp_speech_eval.streaming_features import StreamingFeatureExtractor
from jp_speech_eval.vad import detect_speech_region


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate realtime karaoke-like feedback from a wav file")
    parser.add_argument("--cache", required=True, help="Cache prefix created by scripts/prepare_cache.py")
    parser.add_argument("--wav", required=True, help="User wav file")
    parser.add_argument("--chunk-ms", type=float, default=None, help="Chunk size in milliseconds")
    parser.add_argument("--config", default=None, help="Optional scoring config json path")
    parser.add_argument("--csv", default=None, help="Optional csv output path")
    parser.add_argument("--sleep", action="store_true", help="Sleep between chunks to mimic wall-clock realtime")
    parser.add_argument("--print-every", type=int, default=3, help="Print every N chunks")
    args = parser.parse_args()

    config = load_scoring_config(args.config)
    cache = load_sentence_cache(args.cache)
    sr = cache.meta.sr
    chunk_ms = float(args.chunk_ms or config["realtime"]["chunk_ms"])
    chunk_samples = max(1, int(sr * chunk_ms / 1000.0))

    audio = load_audio(args.wav, sr=sr)
    y = audio.y
    speech_region = detect_speech_region(y, sr)
    expected_user_duration = speech_region.speech_duration if speech_region.detected else len(y) / sr

    extractor = StreamingFeatureExtractor.from_config(config, sr=sr)
    evaluator = RealtimeEvaluator.from_config(cache, config, expected_user_duration_sec=expected_user_duration)

    rows = []
    start = time.perf_counter()
    for i in range(0, len(y), chunk_samples):
        chunk = y[i : i + chunk_samples]
        feat = extractor.process_chunk(chunk)
        fb = evaluator.update(feat)
        row = fb.to_dict()
        row.update({"rms": feat.rms, "dbfs": feat.dbfs, "is_voiced": feat.is_voiced})
        rows.append(row)
        if len(rows) % max(1, args.print_every) == 0:
            f0 = "--" if fb.f0_hz is None else f"{fb.f0_hz:6.1f}Hz"
            print(
                f"{fb.time_sec:6.2f}s  state={fb.endpoint_state:<18s} mora={fb.mora_index:02d}:{fb.mora:<3s} "
                f"T={fb.target_pitch}  f0={f0}  vol={fb.volume_state:<9s} pitch={fb.pitch_state}"
            )
        if args.sleep:
            time.sleep(chunk_samples / sr)

    elapsed = time.perf_counter() - start
    audio_duration = len(y) / sr
    print("\nRealtime simulation summary")
    print(f"Audio duration   : {audio_duration:.3f} sec")
    print(f"Speech duration  : {expected_user_duration:.3f} sec")
    print(f"Processing time  : {elapsed:.3f} sec")
    print(f"Realtime factor  : {audio_duration / max(elapsed, 1e-6):.2f}x")
    print(f"Chunks           : {len(rows)}")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV: {out}")


if __name__ == "__main__":
    main()
