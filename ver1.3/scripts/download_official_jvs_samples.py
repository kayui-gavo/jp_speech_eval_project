#!/usr/bin/env python3
"""Download the three tiny official JVS sample clips for research preflight.

The JVS project page publishes Google Drive links for exactly these three
VOICEACTRESS100_001 samples.  They are downloaded ephemerally into ``outputs``
for CI research only; audio is never committed and should not be uploaded as a
workflow artifact.  The script records SHA-256 hashes so the exact bytes used
by a run are auditable and can later be frozen after inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


OFFICIAL_PROJECT_PAGE = "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus"
SAMPLES = {
    "jvs001": "142aj-qFJOhoteWKqgRzvNoq02JbZIsaG",
    "jvs002": "1idCghceyP9HldFnBKKx9_2ENqXWnr7IP",
    "jvs003": "1plvIsG5Y0l-lYYAMIBH8YHRkDr_prwLM",
}
TARGET_TEXT = "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。"


def _download_url(file_id: str) -> str:
    return (
        "https://drive.usercontent.google.com/download?"
        f"id={file_id}&export=download&confirm=t"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/jvs_official_samples")
    parser.add_argument("--manifest", default="outputs/jvs_official_samples_manifest.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for speaker, file_id in SAMPLES.items():
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
        # WAV may be RIFF or RF64.  HTML/error pages must never silently enter
        # the acoustic benchmark.
        if data[:4] not in {b"RIFF", b"RF64"}:
            prefix = data[:80].decode("utf-8", errors="replace")
            raise RuntimeError(
                f"official JVS sample is not WAV for {speaker}; content_type={content_type}; prefix={prefix!r}"
            )
        path = output_dir / f"{speaker}_VOICEACTRESS100_001.wav"
        path.write_bytes(data)
        rows.append(
            {
                "speaker": speaker,
                "google_drive_file_id": file_id,
                "path": str(path),
                "bytes": len(data),
                "sha256": _sha256(data),
                "target_text": TARGET_TEXT,
                "source": "official_JVS_project_page_sample_link",
            }
        )
        print(f"downloaded {speaker}: {len(data)} bytes sha256={rows[-1]['sha256']}")

    payload = {
        "schema": "jvs_official_samples_manifest_v1",
        "project_page": OFFICIAL_PROJECT_PAGE,
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
