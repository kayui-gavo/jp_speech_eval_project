from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_features import load_audio, trim_silence


class KanadeUnavailableError(RuntimeError):
    pass


def _default_worker_python() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / ".venv-kanade" / "bin" / "python"


def generate_voice_conditioned_reference(
    target_reference_y: np.ndarray,
    *,
    target_sr: int,
    speaker_wav_path: str | Path,
    model_id: str = "frothywater/kanade-25hz-clean",
    worker_python: str | Path | None = None,
) -> np.ndarray:
    """
    Ask the isolated Kanade worker to resynthesize target content in another timbre.

    The main evaluator stays on its existing Python environment; only the worker
    needs the heavier Python 3.12 Kanade stack.
    """
    worker = Path(worker_python or os.environ.get("KANADE_PYTHON") or _default_worker_python())
    if not worker.exists():
        raise KanadeUnavailableError(
            f"Missing Kanade worker Python at {worker}. Create .venv-kanade with Python 3.12 first."
        )

    script = Path(__file__).resolve().parents[2] / "scripts" / "run_kanade_reference.py"
    with tempfile.TemporaryDirectory(prefix="kanade_ref_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        target_wav = tmp_dir_path / "target.wav"
        output_wav = tmp_dir_path / "output.wav"
        sf.write(str(target_wav), np.asarray(target_reference_y, dtype=np.float64), target_sr)
        proc = subprocess.run(
            [
                str(worker),
                str(script),
                "--target-wav",
                str(target_wav),
                "--speaker-wav",
                str(speaker_wav_path),
                "--out-wav",
                str(output_wav),
                "--model-id",
                model_id,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or proc.stdout or "").strip()
            raise KanadeUnavailableError(f"Kanade worker failed: {stderr or 'unknown error'}")
        audio = load_audio(str(output_wav), sr=target_sr)

    y, _ = trim_silence(audio.y, top_db=30.0)
    return y
