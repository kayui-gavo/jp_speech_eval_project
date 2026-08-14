#!/usr/bin/env python3
"""Download three frozen official JVS sample clips for research preflight.

The JVS project page publishes Google Drive links for exactly these three
VOICEACTRESS100_001 samples. They are downloaded ephemerally into ``outputs``
for CI research only; audio is never committed or uploaded as a workflow
artifact.

The exact byte sizes and SHA-256 digests were frozen after the first successful
official-download/native-anchor run on 2026-08-15. Any future source drift now
fails closed instead of silently changing the acoustic benchmark. Updating a
hash therefore requires explicit human review of the replacement source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


OFFICIAL_PROJECT_PAGE = "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus"
SAMPLES = {
    "jvs001": {
        "file_id": "142aj-qFJOhoteWKqgRzvNoq02JbZIsaG",
        "bytes": 778284,
        "sha256": "dc9fd6e4caefc6e1781ad225f0b41ca13153da4afe2fb92f39f175fa3d9d85a7",
    },
    "jvs002": {
        "file_id": "1idCghceyP9HldFnBKKx9_2ENqXWnr7IP",
        "bytes": 642764,
        "sha256": "d91e5199508d89b45d68f18473c013f90bbfd68c1940ad039f5ea0b183f61ae2",
    },
    "jvs003": {
        "file_id": "1plvIsG5Y0l-lYYAMIBH8YHRkDr_prwLM",
        "bytes": 661004,
        "sha256": "7b164601457b27c8a89c6aaef971967e5a6c7cf9bf2d46c303e21d1e8e41ed15",
    },
}
TARGET_TEXT = "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。"


def _download_url(file_id: str) -> str:
    return (
        "https://drive.usercontent.google.com/download?"
        f"id={file_id}&export=download&confirm=t"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_frozen_sample(speaker: str, data: bytes) -> tuple[int, str]:
    spec = SAMPLES[str(speaker)]
    observed_size = len(data)
    observed_hash = _sha256(data)
    expected_size = int(spec["bytes"])
    expected_hash = str(spec["sha256"])
    if observed_size != expected_size or observed_hash != expected_hash:
        raise RuntimeError(
            "official JVS frozen sample drift for "
            f"{speaker}: expected bytes={expected_size} sha256={expected_hash}, "
            f"observed bytes={observed_size} sha256={observed_hash}; "
            "do not update automatically—inspect the upstream replacement first"
        )
    return observed_size, observed_hash


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
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "jp-speech-eval-stage0/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
            content_type = str(response.headers.get("Content-Type") or "")
        if len(data) < 1024:
            raise RuntimeError(f"official JVS sample download too small for {speaker}: {len(data)} bytes")
        # WAV may be RIFF or RF64. HTML/error pages must never silently enter
        # the acoustic benchmark.
        if data[:4] not in {b"RIFF", b"RF64"}:
            prefix = data[:80].decode("utf-8", errors="replace")
            raise RuntimeError(
                f"official JVS sample is not WAV for {speaker}; content_type={content_type}; prefix={prefix!r}"
            )
        observed_size, observed_hash = verify_frozen_sample(speaker, data)
        path = output_dir / f"{speaker}_VOICEACTRESS100_001.wav"
        path.write_bytes(data)
        rows.append(
            {
                "speaker": speaker,
                "google_drive_file_id": file_id,
                "path": str(path),
                "bytes": observed_size,
                "sha256": observed_hash,
                "frozen_bytes_verified": True,
                "target_text": TARGET_TEXT,
                "source": "official_JVS_project_page_sample_link",
            }
        )
        print(f"downloaded+verified {speaker}: {observed_size} bytes sha256={observed_hash}")

    payload = {
        "schema": "jvs_official_samples_manifest_v2",
        "project_page": OFFICIAL_PROJECT_PAGE,
        "frozen_after_run": "2026-08-15_first_successful_native_anchor_preflight",
        "source_drift_policy": "fail_closed_require_explicit_review",
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
