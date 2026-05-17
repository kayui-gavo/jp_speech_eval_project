from __future__ import annotations

import argparse
import csv
from pathlib import Path

from jp_speech_eval.audio_features import extract_f0
from jp_speech_eval.sentence_cache import build_sentence_cache


DEFAULT_TEXTS = [
    "今日はいい天気ですね。",
    "ラーメンをください。",
    "きってを三枚買ってきました。",
    "東京大学の研究室で日本語を勉強しています。",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate side-by-side Japanese TTS audition samples.")
    parser.add_argument("--out-dir", default="outputs/tts_audition")
    parser.add_argument("--text", action="append", default=None, help="Repeat to audition custom texts.")
    parser.add_argument("--backend", action="append", default=None, help="Repeat: pyopenjtalk, voicevox_http, aivis_http")
    parser.add_argument("--tts-url", action="append", default=[], help="Optional backend URL entries matching --backend order.")
    parser.add_argument("--tts-speaker", action="append", type=int, default=[], help="Optional speaker/style ids matching --backend order.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = args.text or DEFAULT_TEXTS
    backends = args.backend or ["pyopenjtalk"]
    rows = []
    for backend_idx, backend in enumerate(backends):
        backend_url = args.tts_url[backend_idx] if backend_idx < len(args.tts_url) else None
        speaker = args.tts_speaker[backend_idx] if backend_idx < len(args.tts_speaker) else None
        for text_idx, text in enumerate(texts, start=1):
            prefix = out_dir / backend / f"{text_idx:02d}"
            try:
                cache = build_sentence_cache(
                    text,
                    prefix,
                    save_reference_wav=True,
                    tts_backend=backend,
                    tts_backend_url=backend_url,
                    tts_speaker=speaker,
                )
                _times, f0, method = extract_f0(cache.ref_y, cache.meta.sr)
                voiced = f0[f0 > 0]
                rows.append({
                    "backend": backend,
                    "text_index": text_idx,
                    "text": text,
                    "reference_source": cache.meta.reference_source,
                    "duration_sec": cache.meta.ref_duration_sec,
                    "f0_method": method,
                    "voiced_frames": int(len(voiced)),
                    "wav_path": str(prefix.with_suffix(".ref.wav")),
                    "status": "ok",
                    "error": "",
                })
            except Exception as exc:
                rows.append({
                    "backend": backend,
                    "text_index": text_idx,
                    "text": text,
                    "reference_source": "",
                    "duration_sec": "",
                    "f0_method": "",
                    "voiced_frames": "",
                    "wav_path": "",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    out_csv = out_dir / "summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")
    for row in rows:
        print(row["backend"], row["text_index"], row["status"], row["wav_path"] or row["error"])


if __name__ == "__main__":
    main()
