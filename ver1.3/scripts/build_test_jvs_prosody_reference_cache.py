#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.alignment_evidence.phone_mora_mapper import map_phones_to_moras  # noqa: E402
from jp_speech_eval.alignment_evidence.textgrid_parser import parse_lab_phone_segments  # noqa: E402
from jp_speech_eval.audio_features import extract_f0, load_audio, trim_silence  # noqa: E402
from jp_speech_eval.prosody_reference_cache import write_prosody_reference_cache  # noqa: E402
from jp_speech_eval.sentence_cache import SentenceCache, SentenceMeta, _mfcc, load_sentence_cache  # noqa: E402
from jp_speech_eval.text_frontend import build_text_info, run_frontend  # noqa: E402


def transcript_for(jvs_root: Path, speaker_id: str, utterance_id: str) -> str:
    transcript_path = jvs_root / speaker_id / "parallel100" / "transcripts_utf8.txt"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Missing JVS transcript: {transcript_path}")
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        utt, text = line.split(":", 1)
        if utt == utterance_id:
            return text.strip()
    raise ValueError(f"Utterance {utterance_id} not found in {transcript_path}")


def jvs_item(jvs_root: Path, speaker_id: str, utterance_id: str) -> Dict[str, str]:
    text = transcript_for(jvs_root, speaker_id, utterance_id)
    wav = jvs_root / speaker_id / "parallel100" / "wav24kHz16bit" / f"{utterance_id}.wav"
    lab = jvs_root / speaker_id / "parallel100" / "lab" / "mon" / f"{utterance_id}.lab"
    if not wav.exists():
        raise FileNotFoundError(f"Missing JVS wav: {wav}")
    if not lab.exists():
        raise FileNotFoundError(f"Missing JVS lab: {lab}")
    return {
        "speaker_id": speaker_id,
        "utterance_id": utterance_id,
        "target_text": text,
        "audio_path": str(wav),
        "lab_path": str(lab),
    }


def lab_mora_boundaries(
    *,
    lab_path: Path,
    moras: List[str],
    trim_start_sec: float,
    trim_duration_sec: float,
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    phones = parse_lab_phone_segments(lab_path)
    segments, mapping = map_phones_to_moras(phones, moras)
    if len(segments) != len(moras):
        raise ValueError(
            f"phone-to-mora mapping failed: segments={len(segments)} moras={len(moras)} "
            f"flags={mapping.get('mapping_warning_flags')}"
        )
    boundaries: List[Tuple[float, float]] = []
    for segment in segments:
        start = max(0.0, float(segment.start) - trim_start_sec)
        end = min(trim_duration_sec, float(segment.end) - trim_start_sec)
        if end <= start:
            end = min(trim_duration_sec, start + 0.02)
        boundaries.append((round(float(start), 6), round(float(end), 6)))
    return boundaries, {
        "mapping_success": bool(mapping.get("mapping_success")),
        "mapping_warning_flags": list(mapping.get("mapping_warning_flags") or []),
    }


def build_test_jvs_cache(
    *,
    jvs_root: Path,
    speaker_id: str,
    utterance_id: str,
    out_prefix: Path,
    sample_rate: int = 16000,
    write_sidecar: bool = True,
) -> SentenceCache:
    item = jvs_item(jvs_root, speaker_id, utterance_id)
    text_info = build_text_info(item["target_text"])
    audio = load_audio(item["audio_path"], sr=sample_rate)
    ref_y, trim_idx = trim_silence(audio.y, top_db=30.0)
    trim_start_sec = float(trim_idx[0]) / float(sample_rate)
    ref_duration_sec = len(ref_y) / sample_rate
    boundaries, mapping_meta = lab_mora_boundaries(
        lab_path=Path(item["lab_path"]),
        moras=text_info.moras,
        trim_start_sec=trim_start_sec,
        trim_duration_sec=ref_duration_sec,
    )
    f0_times, f0, _method = extract_f0(ref_y, sample_rate)
    meta = SentenceMeta(
        text=text_info.text,
        kana=text_info.kana,
        moras=text_info.moras,
        target_pitch=text_info.target_pitch,
        pitch_target_source=text_info.pitch_target_source,
        is_question=text_info.is_question,
        sr=sample_rate,
        ref_duration_sec=round(float(ref_duration_sec), 6),
        ref_mora_boundaries=boundaries,
        frontend_raw=run_frontend(item["target_text"]),
        accent_phrases=text_info.accent_phrases,
        reference_text=item["target_text"],
        reference_source="jvs_native_reference",
        ref_boundary_method="lab_phone_mora",
        reference_id=f"{speaker_id}:{utterance_id}",
        reference_provider="jvs",
        reference_voice=speaker_id,
        reference_language="ja-JP",
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_prefix.with_suffix(".npz"),
        ref_y=ref_y.astype(np.float32),
        ref_mfcc=_mfcc(ref_y, sample_rate).astype(np.float32),
        ref_f0_times=f0_times.astype(np.float32),
        ref_f0=f0.astype(np.float32),
    )
    sf.write(str(out_prefix.with_suffix(".ref.wav")), ref_y, sample_rate)
    cache = load_sentence_cache(out_prefix)
    if write_sidecar:
        write_prosody_reference_cache(
            cache,
            reference_audio_path=out_prefix.with_suffix(".ref.wav"),
            verified_reference=True,
        )
    cache_meta = {
        "jvs_root": str(jvs_root),
        "speaker_id": speaker_id,
        "utterance_id": utterance_id,
        "source_audio_path": item["audio_path"],
        "source_lab_path": item["lab_path"],
        "trim_start_sec": round(trim_start_sec, 6),
        **mapping_meta,
    }
    out_prefix.with_suffix(".build_meta.json").write_text(json.dumps(cache_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a test-only verified JVS prosody reference cache.")
    parser.add_argument("--jvs-root", default=str(ROOT.parent / "JVS"))
    parser.add_argument("--speaker-id", default="jvs001")
    parser.add_argument("--utterance-id", default="VOICEACTRESS100_001")
    parser.add_argument("--out-prefix", default="outputs/test_jvs_prosody_reference_cache/jvs001_VOICEACTRESS100_001")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--no-sidecar", action="store_true")
    args = parser.parse_args()

    cache = build_test_jvs_cache(
        jvs_root=Path(args.jvs_root),
        speaker_id=args.speaker_id,
        utterance_id=args.utterance_id,
        out_prefix=ROOT / args.out_prefix,
        sample_rate=args.sample_rate,
        write_sidecar=not args.no_sidecar,
    )
    print(f"cache_prefix={cache.prefix}")
    print(f"text={cache.meta.text}")
    print(f"mora_count={cache.mora_count}")
    print(f"reference_source={cache.meta.reference_source}")
    print(f"boundary_method={cache.meta.ref_boundary_method}")
    print(f"sidecar_written={not args.no_sidecar}")


if __name__ == "__main__":
    main()
