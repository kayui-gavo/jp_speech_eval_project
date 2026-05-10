from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser(description="Record microphone audio to wav")
    parser.add_argument("--out", required=True, help="Output wav path")
    parser.add_argument("--seconds", type=float, default=4.0, help="Recording duration")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording {args.seconds:.1f} sec at {args.sr} Hz. Speak now...")
    audio = sd.rec(int(args.seconds * args.sr), samplerate=args.sr, channels=1, dtype="float32")
    sd.wait()
    audio = np.asarray(audio).squeeze()
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / (peak + 1e-9) * 0.95
    sf.write(out, audio, args.sr)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
