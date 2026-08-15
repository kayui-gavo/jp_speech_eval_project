#!/usr/bin/env python3
"""Download public UME-JRF test-listening samples for research preflight.

NII-SRC publishes exactly these five WAV links on the UME-JRF corpus page as
speech samples by native speakers of Chinese.  The corpus is research-use only.
This script therefore downloads the public samples ephemerally for CI research,
records their provenance, and never commits or uploads the audio itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import wave


OFFICIAL_PAGE = "https://research.nii.ac.jp/src/en/UME-JRF.html"
BASE = "https://research.nii.ac.jp/src/sample/UME-JRF"
SAMPLES = (
    {
        "sample_id": "A1_001",
        "category": "phonetically_balanced_sentence",
        "filename": "A1_001.wav",
        "target_text": "六百人のお客さんの人いきれにむし暑くて扇子を使わずにいられない。",
    },
    {
        "sample_id": "C1_001",
        "category": "difficult_sentence",
        "filename": "C1_001.wav",
        "target_text": "次郎はおどる？",
    },
    {
        "sample_id": "B1_001",
        "category": "prosody_sentence",
        "filename": "B1_001.wav",
        "target_text": "天気が悪いので、電気をつけた。",
    },
    {
        "sample_id": "D1_001",
        "category": "minimal_pair_word",
        "filename": "D1_001.wav",
        "target_text": "じぶつ",
        "minimal_pair_partner": "じんぶつ",
    },
    {
        "sample_id": "D1_002",
        "category": "minimal_pair_word",
        "filename": "D1_002.wav",
        "target_text": "じんぶつ",
        "minimal_pair_partner": "じぶつ",
    },
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _wave_metadata(path: Path) -> dict:
    with wave.open(str(path), "rb") as handle:
        frames = int(handle.getnframes())
        sample_rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frames,
        "duration_sec": frames / sample_rate if sample_rate > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/ume_jrf_public_samples")
    parser.add_argument("--manifest", default="outputs/ume_jrf_public_samples_manifest.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in SAMPLES:
        url = f"{BASE}/{spec['filename']}"
        request = urllib.request.Request(url, headers={"User-Agent": "jp-speech-eval-stage0/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
            content_type = str(response.headers.get("Content-Type") or "")
        if len(data) < 1024 or data[:4] not in {b"RIFF", b"RF64"}:
            prefix = data[:80].decode("utf-8", errors="replace")
            raise RuntimeError(
                f"UME-JRF sample is not usable WAV: {spec['sample_id']} "
                f"bytes={len(data)} content_type={content_type!r} prefix={prefix!r}"
            )
        path = output_dir / spec["filename"]
        path.write_bytes(data)
        metadata = _wave_metadata(path)
        row = {
            **spec,
            "path": str(path),
            "source_url": url,
            "source_page": OFFICIAL_PAGE,
            "source_description": "official_NII_SRC_UME_JRF_public_test_listening_sample",
            "speaker_l1": "Chinese",
            "research_use_only": True,
            "public_sample_has_teacher_grade_on_web_page": False,
            "raw_bytes": len(data),
            "raw_sha256": _sha256(data),
            **metadata,
        }
        rows.append(row)
        print(
            f"downloaded {row['sample_id']}: {row['raw_bytes']} bytes, "
            f"{row['sample_rate']} Hz, {row['duration_sec']:.3f} s"
        )

    payload = {
        "schema": "ume_jrf_public_samples_manifest_v1",
        "official_page": OFFICIAL_PAGE,
        "corpus": "UME-JRF",
        "corpus_license_summary": "research_purpose_only",
        "speaker_description_from_official_page": "utterances_by_native_speakers_of_Chinese",
        "audio_committed_to_repository": False,
        "audio_should_be_uploaded_as_artifact": False,
        "teacher_grading_available_in_full_corpus": True,
        "teacher_grading_available_for_public_samples_on_page": False,
        "samples": rows,
    }
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
