#!/usr/bin/env python3
"""Download official JVS sample clips ephemerally for research preflight.

The JVS project page publishes Google Drive links for three
VOICEACTRESS100_001 samples. They are downloaded into ``outputs`` only for the
CI research run; audio is never committed or uploaded as a workflow artifact.

A previous Stage-0 revision incorrectly made the raw WAV byte hash a hard gate.
Google Drive can serve the same waveform in a different WAV container/sample-
rate representation (the first successful jvs001 artifact was ~778 kB, while a
later official response was ~414 kB and still represented the same ~8.621 s
utterance). Reproducibility therefore cannot be defined by transport bytes.

The hard gate is now semantic and source-aware:

* official Google Drive file ID must be the reviewed one;
* payload must be a readable mono PCM WAV;
* duration must match the previously observed official utterance within a tight
  tolerance;
* sample rate / sample width are recorded, not silently normalized here;
* raw byte SHA-256 is retained as provenance and transport drift is reported,
  but a new container hash alone does not invalidate the acoustic anchor.

The downstream benchmark resamples through the project's audio loader and
re-checks target-conditioned model behavior. This is safer than pretending a
mutable HTTP representation is an immutable corpus checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import wave


OFFICIAL_PROJECT_PAGE = "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus"
SAMPLES = {
    "jvs001": {
        "file_id": "142aj-qFJOhoteWKqgRzvNoq02JbZIsaG",
        "expected_duration_sec": 8.6210625,
        "historical_raw_variants": {
            "dc9fd6e4caefc6e1781ad225f0b41ca13153da4afe2fb92f39f175fa3d9d85a7": 778284,
        },
    },
    "jvs002": {
        "file_id": "1idCghceyP9HldFnBKKx9_2ENqXWnr7IP",
        "expected_duration_sec": 7.509375,
        "historical_raw_variants": {
            "d91e5199508d89b45d68f18473c013f90bbfd68c1940ad039f5ea0b183f61ae2": 642764,
        },
    },
    "jvs003": {
        "file_id": "1plvIsG5Y0l-lYYAMIBH8YHRkDr_prwLM",
        "expected_duration_sec": 8.910125,
        "historical_raw_variants": {
            "7b164601457b27c8a89c6aaef971967e5a6c7cf9bf2d46c303e21d1e8e41ed15": 661004,
        },
    },
}
TARGET_TEXT = "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。"
DURATION_TOLERANCE_SEC = 0.015


def _download_url(file_id: str) -> str:
    return (
        "https://drive.usercontent.google.com/download?"
        f"id={file_id}&export=download&confirm=t"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_wav_semantics(data: bytes) -> dict[str, object]:
    """Return transport-independent basic acoustic semantics for a WAV payload."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = int(handle.getnchannels())
            sample_width_bytes = int(handle.getsampwidth())
            sample_rate = int(handle.getframerate())
            frame_count = int(handle.getnframes())
            compression = str(handle.getcomptype())
    except (wave.Error, EOFError) as exc:
        raise RuntimeError(f"official JVS payload is not a readable PCM WAV: {exc}") from exc
    if channels != 1:
        raise RuntimeError(f"official JVS sample must be mono, got {channels} channels")
    if compression != "NONE":
        raise RuntimeError(f"official JVS sample must be uncompressed PCM WAV, got {compression}")
    if sample_width_bytes not in {2, 3, 4}:
        raise RuntimeError(f"unexpected JVS PCM sample width: {sample_width_bytes} bytes")
    if sample_rate <= 0 or frame_count <= 0:
        raise RuntimeError("official JVS WAV has invalid sample rate/frame count")
    duration = frame_count / float(sample_rate)
    return {
        "channels": channels,
        "sample_width_bytes": sample_width_bytes,
        "sample_rate": sample_rate,
        "frame_count": frame_count,
        "duration_sec": duration,
        "compression": compression,
    }


def verify_official_sample(speaker: str, data: bytes) -> dict[str, object]:
    spec = SAMPLES[str(speaker)]
    semantics = inspect_wav_semantics(data)
    expected_duration = float(spec["expected_duration_sec"])
    observed_duration = float(semantics["duration_sec"])
    if abs(observed_duration - expected_duration) > DURATION_TOLERANCE_SEC:
        raise RuntimeError(
            "official JVS acoustic source drift for "
            f"{speaker}: expected duration≈{expected_duration:.6f}s, "
            f"observed={observed_duration:.6f}s; inspect upstream before using this anchor"
        )
    digest = _sha256(data)
    historical = dict(spec.get("historical_raw_variants") or {})
    return {
        **semantics,
        "raw_bytes": len(data),
        "raw_sha256": digest,
        "raw_transport_variant_previously_observed": digest in historical,
        "raw_transport_hash_is_acoustic_identity": False,
        "semantic_duration_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/jvs_official_samples")
    parser.add_argument("--manifest", default="outputs/jvs_official_samples_manifest.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for speaker, spec in SAMPLES.items():
        file_id = str(spec["file_id"])
        url = _download_url(file_id)
        request = urllib.request.Request(url, headers={"User-Agent": "jp-speech-eval-stage0/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
            content_type = str(response.headers.get("Content-Type") or "")
        if len(data) < 1024:
            raise RuntimeError(f"official JVS sample download too small for {speaker}: {len(data)} bytes")
        if data[:4] not in {b"RIFF", b"RF64"}:
            prefix = data[:80].decode("utf-8", errors="replace")
            raise RuntimeError(
                f"official JVS sample is not WAV for {speaker}; content_type={content_type}; prefix={prefix!r}"
            )
        semantic = verify_official_sample(speaker, data)
        path = output_dir / f"{speaker}_VOICEACTRESS100_001.wav"
        path.write_bytes(data)
        rows.append(
            {
                "speaker": speaker,
                "google_drive_file_id": file_id,
                "path": str(path),
                "target_text": TARGET_TEXT,
                "source": "official_JVS_project_page_sample_link",
                **semantic,
            }
        )
        print(
            f"downloaded+verified {speaker}: duration={semantic['duration_sec']:.6f}s "
            f"sr={semantic['sample_rate']} raw_sha256={semantic['raw_sha256']}"
        )

    payload = {
        "schema": "jvs_official_samples_manifest_v3",
        "project_page": OFFICIAL_PROJECT_PAGE,
        "source_identity": "reviewed_official_file_id_plus_verified_audio_semantics",
        "raw_http_bytes_are_immutable_source_identity": False,
        "source_drift_policy": "fail_on_audio_semantic_drift_not_container_hash_only",
        "audio_committed_to_repository": False,
        "audio_should_be_uploaded_as_artifact": False,
        "research_use_only": True,
        "samples": rows,
    }
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
