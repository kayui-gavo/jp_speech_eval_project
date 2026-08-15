#!/usr/bin/env python3
"""Download official JVS sample clips ephemerally for research preflight.

The JVS project page publishes Google Drive links for three
VOICEACTRESS100_001 samples. They are downloaded into ``outputs`` only for the
CI research run; audio is never committed or uploaded as a workflow artifact.

Two target-provenance failures were found before human recording:

1. surface kanji ``明王`` can be analysed as a personal-name reading instead of
   lexical ``みょうおう``;
2. even a kana reading override can be morphologically/contextually re-analysed
   by the text frontend, causing the second identical ``みょうおう`` occurrence
   to receive a different phone sequence from the first.

For this audited native anchor, a reviewed full-sentence kana reading is
therefore accompanied by an explicit, segment-auditable logical-phone target.
Phone positions also carry a construct role. Long-vowel extension morae are
marked ``long_vowel_timing`` rather than silently counted as ordinary
segmental-clarity events; this is important because repeated-vowel CTC evidence
can reflect mora timing/length realization rather than vowel identity.
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
ORDINARY_ROLE = "ordinary_segmental_clarity"
LONG_VOWEL_ROLE = "long_vowel_timing"
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
TARGET_READING = "また、とうじのように、ごだいみょうおうとよばれる、しゅようなみょうおうのちゅうおうにはいされることもおおい。"


def _roles(length: int, *long_positions: int) -> tuple[str, ...]:
    role_values = [ORDINARY_ROLE] * int(length)
    for position in long_positions:
        if position < 0 or position >= length:
            raise ValueError(f"long-vowel role position outside segment: {position}/{length}")
        role_values[position] = LONG_VOWEL_ROLE
    return tuple(role_values)


# Phone chunks are reviewed against the intended reading. ``roles`` annotate
# only the phone that realizes the *extension mora* of a long vowel; the base
# vowel remains ordinary. For みょうおう, moras are ミョ・ウ・オ・ウ, hence
# the second and fourth /o/ tokens are timing extensions.
TARGET_PHONE_SEGMENTS = (
    {"surface": "また", "reading": "また", "phones": ("m", "a", "t", "a"), "roles": _roles(4)},
    {"surface": "東寺", "reading": "とうじ", "phones": ("t", "o", "o", "j", "i"), "roles": _roles(5, 2)},
    {"surface": "の", "reading": "の", "phones": ("n", "o"), "roles": _roles(2)},
    {"surface": "ように", "reading": "ように", "phones": ("y", "o", "u", "n", "i"), "roles": _roles(5, 2)},
    {"surface": "五大", "reading": "ごだい", "phones": ("g", "o", "d", "a", "i"), "roles": _roles(5)},
    {"surface": "明王", "reading": "みょうおう", "phones": ("my", "o", "o", "o", "o"), "roles": _roles(5, 2, 4)},
    {"surface": "と", "reading": "と", "phones": ("t", "o"), "roles": _roles(2)},
    {"surface": "呼ばれる", "reading": "よばれる", "phones": ("y", "o", "b", "a", "r", "e", "r", "u"), "roles": _roles(8)},
    {"surface": "主要", "reading": "しゅよう", "phones": ("sh", "u", "y", "o", "o"), "roles": _roles(5, 4)},
    {"surface": "な", "reading": "な", "phones": ("n", "a"), "roles": _roles(2)},
    {"surface": "明王", "reading": "みょうおう", "phones": ("my", "o", "o", "o", "o"), "roles": _roles(5, 2, 4)},
    {"surface": "の", "reading": "の", "phones": ("n", "o"), "roles": _roles(2)},
    {"surface": "中央", "reading": "ちゅうおう", "phones": ("ch", "u", "u", "o", "o"), "roles": _roles(5, 2, 4)},
    {"surface": "に", "reading": "に", "phones": ("n", "i"), "roles": _roles(2)},
    {"surface": "配される", "reading": "はいされる", "phones": ("h", "a", "i", "s", "a", "r", "e", "r", "u"), "roles": _roles(9)},
    {"surface": "ことも", "reading": "ことも", "phones": ("k", "o", "t", "o", "m", "o"), "roles": _roles(6)},
    {"surface": "多い", "reading": "おおい", "phones": ("o", "o", "i"), "roles": _roles(3, 1)},
)
TARGET_PHONES = tuple(
    phone
    for segment in TARGET_PHONE_SEGMENTS
    for phone in segment["phones"]
)
TARGET_PHONE_ROLES = tuple(
    role
    for segment in TARGET_PHONE_SEGMENTS
    for role in segment["roles"]
)


def _phone_index_metadata() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    phone_index = 0
    for segment_index, segment in enumerate(TARGET_PHONE_SEGMENTS):
        segment_phones = tuple(str(phone) for phone in segment["phones"])
        segment_roles = tuple(str(role) for role in segment["roles"])
        if len(segment_phones) != len(segment_roles):
            raise ValueError(f"JVS phone/role length mismatch in segment {segment_index}")
        for segment_phone_index, (phone, role) in enumerate(zip(segment_phones, segment_roles)):
            if role not in {ORDINARY_ROLE, LONG_VOWEL_ROLE}:
                raise ValueError(f"unsupported JVS phone construct role: {role}")
            rows.append(
                {
                    "phone_index": phone_index,
                    "phone": phone,
                    "construct_role": role,
                    "segment_index": segment_index,
                    "segment_phone_index": segment_phone_index,
                    "segment_surface": str(segment["surface"]),
                    "segment_reading": str(segment["reading"]),
                }
            )
            phone_index += 1
    return tuple(rows)


TARGET_PHONE_INDEX_METADATA = _phone_index_metadata()
READING_PROVENANCE = {
    "東寺": "とうじ",
    "五大明王": "ごだいみょうおう",
    "主要": "しゅよう",
    "明王": "みょうおう",
    "中央": "ちゅうおう",
    "配される": "はいされる",
}
PHONE_PROVENANCE = {
    "policy": "reviewed_logical_phone_override_v1",
    "frontend_family": "pyopenjtalk-plus_compatible_Japanese_phone_inventory",
    "segment_auditable": True,
    "construct_role_auditable": True,
    "long_vowel_extension_role": LONG_VOWEL_ROLE,
    "identical_lexeme_constraint": "both 明王/みょうおう occurrences use [my,o,o,o,o]",
    "known_kana_reanalysis_failure": "second みょうおう was previously re-analysed as [m,i,y,o,u,o,u]",
    "purpose": "native_anchor_target_provenance_not_general_text_frontend_replacement",
}
DURATION_TOLERANCE_SEC = 0.015


def _download_url(file_id: str) -> str:
    return "https://drive.usercontent.google.com/download?" f"id={file_id}&export=download&confirm=t"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_wav_semantics(data: bytes) -> dict[str, object]:
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
    serializable_segments = [
        {
            "surface": str(segment["surface"]),
            "reading": str(segment["reading"]),
            "phones": [str(phone) for phone in segment["phones"]],
            "roles": [str(role) for role in segment["roles"]],
        }
        for segment in TARGET_PHONE_SEGMENTS
    ]
    serializable_index = [dict(row) for row in TARGET_PHONE_INDEX_METADATA]
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
                "target_reading": TARGET_READING,
                "target_reading_source": "reviewed_manual_reading_override",
                "target_reading_provenance": READING_PROVENANCE,
                "target_phones": list(TARGET_PHONES),
                "target_phone_roles": list(TARGET_PHONE_ROLES),
                "target_phone_segments": serializable_segments,
                "target_phone_index_metadata": serializable_index,
                "target_phone_source": "reviewed_logical_phone_override_v1",
                "target_phone_provenance": PHONE_PROVENANCE,
                "automatic_surface_g2p_is_safe_for_anchor": False,
                "automatic_kana_g2p_is_phone_exact_for_anchor": False,
                "known_surface_g2p_failure": "明王 can be analysed as あきらおう instead of みょうおう",
                "known_kana_g2p_failure": "second みょうおう can be re-analysed differently from the first identical lexeme",
                "source": "official_JVS_project_page_sample_link",
                **semantic,
            }
        )
        print(
            f"downloaded+verified {speaker}: duration={semantic['duration_sec']:.6f}s "
            f"sr={semantic['sample_rate']} raw_sha256={semantic['raw_sha256']}"
        )

    payload = {
        "schema": "jvs_official_samples_manifest_v6",
        "project_page": OFFICIAL_PROJECT_PAGE,
        "source_identity": "reviewed_official_file_id_plus_verified_audio_semantics",
        "raw_http_bytes_are_immutable_source_identity": False,
        "source_drift_policy": "fail_on_audio_semantic_drift_not_container_hash_only",
        "target_reading_override_required": True,
        "target_phone_override_required": True,
        "target_phone_segments_required": True,
        "target_phone_construct_roles_required": True,
        "long_vowel_extension_is_ordinary_segmental_clarity": False,
        "text_or_kana_g2p_is_authoritative_phone_source": False,
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
