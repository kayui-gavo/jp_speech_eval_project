#!/usr/bin/env python3
"""Audit free-speech response audio at corpus level before v10 acceptance.

This is intentionally separate from runtime ``recording_quality.py``. Runtime
analyzability asks whether one recording can be processed. This audit asks
whether an entire development/held corpus is structurally trustworthy: files
exist and decode, response audio is not duplicated across splits, source
recordings do not leak across splits, and collection/device conditions are
visible before criterion claims are made.

Only structural corruption/leakage creates hard blockers. Signal-level findings
such as clipping, low RMS, stereo, mixed sample rates, or device imbalance are
review-only and never become learner-ability labels or automatic exclusions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import soundfile as sf

from validate_free_speech_sample_manifest import validate_manifest_file


AUDIT_SCHEMA = "free_speech_audio_dataset_audit_v11"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "data" / "research_eval" / "free_speech_v11_data_quality_protocol.json"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_csv(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_audio(audio_root: Path, audio_path: str) -> Path:
    path = Path(audio_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (audio_root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _audio_metrics(path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    info = sf.info(str(path))
    if int(info.frames) <= 0 or int(info.samplerate) <= 0:
        raise ValueError("audio has zero frames or invalid sample rate")
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if y.size == 0 or y.shape[0] <= 0:
        raise ValueError("decoded audio is empty")
    values = np.asarray(y, dtype=np.float32)
    abs_values = np.abs(values)
    peak = float(np.max(abs_values)) if abs_values.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))) if values.size else 0.0
    thresholds = protocol.get("review_thresholds") if isinstance(protocol.get("review_thresholds"), Mapping) else {}
    clip_level = float(thresholds.get("clipping_absolute_amplitude", 0.999))
    clipping_fraction = float(np.mean(abs_values >= clip_level)) if abs_values.size else 0.0
    duration = float(values.shape[0] / sr)
    return {
        "samplerate": int(sr),
        "channels": int(values.shape[1]),
        "frames": int(values.shape[0]),
        "duration_sec": duration,
        "format": _text(info.format),
        "subtype": _text(info.subtype),
        "peak_abs": peak,
        "rms": rms,
        "clipping_fraction": clipping_fraction,
    }


def _finding(kind: str, code: str, *, sample_ids: list[str] | None = None, detail: Any = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "code": code,
        "sample_ids": list(sample_ids or []),
        "detail": detail,
    }


def _speaker_split_leakage(rows: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    splits_by_speaker: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        speaker = _text(row.get("speaker_id"))
        group = _text(row.get("speaker_group"))
        split = _text(row.get("split"))
        if speaker and group in {"learner", "native"} and split:
            splits_by_speaker[speaker].add(split)
    return {
        speaker: sorted(splits)
        for speaker, splits in splits_by_speaker.items()
        if "development" in splits and "held" in splits
    }


def _device_review(
    manifest_rows: list[Mapping[str, Any]],
    speaker_metadata_csv: str | Path | None,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if speaker_metadata_csv is None:
        return {
            "metadata_supplied": False,
            "speaker_count": len({_text(row.get("speaker_id")) for row in manifest_rows if _text(row.get("speaker_id"))}),
            "device_metadata_coverage": None,
            "group_device_sets": {},
            "review_recommended": False,
            "reason": "speaker_metadata_not_supplied",
        }
    metadata_rows, fields = _read_csv(speaker_metadata_csv)
    if "speaker_id" not in fields or "recording_device" not in fields:
        return {
            "metadata_supplied": True,
            "speaker_count": 0,
            "device_metadata_coverage": 0.0,
            "group_device_sets": {},
            "review_recommended": True,
            "reason": "speaker_metadata_missing_required_device_columns",
        }
    metadata = {_text(row.get("speaker_id")): row for row in metadata_rows if _text(row.get("speaker_id"))}
    group_by_speaker: dict[str, str] = {}
    for row in manifest_rows:
        speaker = _text(row.get("speaker_id"))
        group = _text(row.get("speaker_group"))
        if speaker and group in {"learner", "native"}:
            old = group_by_speaker.get(speaker)
            if old and old != group:
                raise ValueError(f"speaker_group inconsistent for {speaker}: {old} vs {group}")
            group_by_speaker[speaker] = group
    speakers = sorted(group_by_speaker)
    with_device = [speaker for speaker in speakers if _text((metadata.get(speaker) or {}).get("recording_device"))]
    coverage = len(with_device) / len(speakers) if speakers else None
    group_devices: dict[str, set[str]] = {"learner": set(), "native": set()}
    group_speakers_with_device: Counter[str] = Counter()
    for speaker in with_device:
        group = group_by_speaker[speaker]
        device = _text((metadata.get(speaker) or {}).get("recording_device"))
        group_devices[group].add(device)
        group_speakers_with_device[group] += 1

    thresholds = protocol.get("review_thresholds") if isinstance(protocol.get("review_thresholds"), Mapping) else {}
    min_coverage = float(thresholds.get("device_metadata_min_coverage_for_confound_review", 0.8))
    min_group_speakers = int(thresholds.get("device_group_min_speakers_for_confound_review", 3))
    sets_disjoint = bool(group_devices["learner"] and group_devices["native"] and group_devices["learner"].isdisjoint(group_devices["native"]))
    eligible = (
        coverage is not None
        and coverage >= min_coverage
        and group_speakers_with_device["learner"] >= min_group_speakers
        and group_speakers_with_device["native"] >= min_group_speakers
    )
    review = bool(eligible and sets_disjoint and thresholds.get("device_group_disjoint_sets_review", True))
    reason = "learner_native_device_sets_disjoint" if review else "no_strong_device_group_confound_detected"
    if coverage is not None and coverage < min_coverage:
        reason = "device_metadata_coverage_too_low_for_confound_test"
    return {
        "metadata_supplied": True,
        "speaker_count": len(speakers),
        "device_metadata_coverage": coverage,
        "speaker_count_with_device": len(with_device),
        "group_speaker_count_with_device": dict(group_speakers_with_device),
        "group_device_sets": {key: sorted(value) for key, value in group_devices.items()},
        "disjoint_group_device_sets": sets_disjoint,
        "confound_test_eligible": eligible,
        "review_recommended": review,
        "reason": reason,
        "interpretation": "speaker_level_collection_confound_review_not_learner_ability",
    }


def audit(
    manifest_csv: str | Path,
    audio_root: str | Path,
    *,
    protocol_json: str | Path = DEFAULT_PROTOCOL,
    speaker_metadata_csv: str | Path | None = None,
) -> dict[str, Any]:
    protocol = _load_json(protocol_json)
    manifest_validation = validate_manifest_file(manifest_csv)
    rows, _ = _read_csv(manifest_csv)
    root = Path(audio_root).expanduser().resolve()

    hard: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    if not manifest_validation.get("ok"):
        hard.append(_finding("hard_blocker", "manifest_validation_failure", detail=manifest_validation.get("errors") or []))

    leakage = _speaker_split_leakage(rows)
    if leakage:
        hard.append(_finding("hard_blocker", "speaker_split_leakage", detail=leakage))

    sample_reports: list[dict[str, Any]] = []
    hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_ids_seen: set[str] = set()

    thresholds = protocol.get("review_thresholds") if isinstance(protocol.get("review_thresholds"), Mapping) else {}
    clip_review = float(thresholds.get("clipping_fraction_review", 0.001))
    low_rms_review = float(thresholds.get("low_rms_review", 0.003))
    short_review = float(thresholds.get("very_short_duration_sec_review", 0.35))
    long_review = float(thresholds.get("very_long_duration_sec_review", 45.0))

    for row in rows:
        sample_id = _text(row.get("sample_id"))
        audio_path = _text(row.get("audio_path"))
        split = _text(row.get("split"))
        source_recording_id = _text(row.get("source_recording_id"))
        channel_pair_id = _text(row.get("channel_pair_id"))
        if sample_id:
            sample_ids_seen.add(sample_id)
        report: dict[str, Any] = {
            "sample_id": sample_id,
            "speaker_id": _text(row.get("speaker_id")),
            "speaker_group": _text(row.get("speaker_group")),
            "split": split,
            "task_mode": _text(row.get("task_mode")),
            "channel_condition": _text(row.get("channel_condition")),
            "channel_pair_id": channel_pair_id,
            "source_recording_id": source_recording_id,
            "audio_path": audio_path,
            "exists": False,
            "decodable": False,
            "sha256": "",
            "metrics": {},
            "review_codes": [],
        }
        if source_recording_id:
            source_groups[source_recording_id].append(report)
        if not audio_path:
            hard.append(_finding("hard_blocker", "missing_response_audio", sample_ids=[sample_id], detail="audio_path empty"))
            sample_reports.append(report)
            continue
        path = _resolve_audio(root, audio_path)
        report["resolved_path"] = str(path)
        if not path.is_file():
            hard.append(_finding("hard_blocker", "missing_response_audio", sample_ids=[sample_id], detail=str(path)))
            sample_reports.append(report)
            continue
        report["exists"] = True
        try:
            digest = _sha256(path)
            report["sha256"] = digest
            hash_groups[digest].append(report)
            metrics = _audio_metrics(path, protocol)
        except Exception as exc:
            hard.append(_finding("hard_blocker", "undecodable_response_audio", sample_ids=[sample_id], detail=f"{type(exc).__name__}: {exc}"))
            sample_reports.append(report)
            continue
        report["decodable"] = True
        report["metrics"] = metrics
        if int(metrics.get("frames") or 0) <= 0 or float(metrics.get("duration_sec") or 0.0) <= 0:
            hard.append(_finding("hard_blocker", "empty_response_audio", sample_ids=[sample_id]))
        if bool(thresholds.get("non_mono_review", True)) and int(metrics["channels"]) != 1:
            report["review_codes"].append("non_mono_response_audio")
        if float(metrics["clipping_fraction"]) >= clip_review:
            report["review_codes"].append("clipping_fraction_high")
        if float(metrics["rms"]) < low_rms_review:
            report["review_codes"].append("low_rms")
        if float(metrics["duration_sec"]) < short_review:
            report["review_codes"].append("very_short_audio")
        if float(metrics["duration_sec"]) > long_review:
            report["review_codes"].append("very_long_audio")
        for code in report["review_codes"]:
            review.append(_finding("review_only", code, sample_ids=[sample_id], detail=metrics))
        sample_reports.append(report)

    for source_id, members in source_groups.items():
        splits = {_text(item.get("split")) for item in members}
        if "development" in splits and "held" in splits:
            hard.append(_finding(
                "hard_blocker",
                "development_held_source_recording_overlap",
                sample_ids=[_text(item.get("sample_id")) for item in members],
                detail={"source_recording_id": source_id, "splits": sorted(splits)},
            ))

    for digest, members in hash_groups.items():
        if len(members) < 2:
            continue
        sample_ids = [_text(item.get("sample_id")) for item in members]
        splits = {_text(item.get("split")) for item in members}
        sources = {_text(item.get("source_recording_id")) for item in members if _text(item.get("source_recording_id"))}
        pairs = {_text(item.get("channel_pair_id")) for item in members if _text(item.get("channel_pair_id"))}
        if "development" in splits and "held" in splits:
            hard.append(_finding(
                "hard_blocker",
                "development_held_exact_audio_hash_overlap",
                sample_ids=sample_ids,
                detail={"sha256": digest, "splits": sorted(splits)},
            ))
        elif len(sources) != 1 or not sources:
            hard.append(_finding(
                "hard_blocker",
                "undeclared_exact_duplicate_response",
                sample_ids=sample_ids,
                detail={"sha256": digest, "source_recording_ids": sorted(sources)},
            ))
        else:
            review.append(_finding(
                "review_only",
                "declared_shared_source_has_byte_identical_audio",
                sample_ids=sample_ids,
                detail={"sha256": digest, "source_recording_id": next(iter(sources)), "channel_pair_ids": sorted(pairs)},
            ))

    decodable = [item for item in sample_reports if item.get("decodable")]
    sample_rates = sorted({int((item.get("metrics") or {}).get("samplerate") or 0) for item in decodable if int((item.get("metrics") or {}).get("samplerate") or 0) > 0})
    subtypes = sorted({_text((item.get("metrics") or {}).get("subtype")) for item in decodable if _text((item.get("metrics") or {}).get("subtype"))})
    channel_counts = sorted({int((item.get("metrics") or {}).get("channels") or 0) for item in decodable if int((item.get("metrics") or {}).get("channels") or 0) > 0})
    if bool(thresholds.get("mixed_sample_rate_review", True)) and len(sample_rates) > 1:
        review.append(_finding("review_only", "mixed_sample_rates", detail=sample_rates))
    if bool(thresholds.get("mixed_audio_subtype_review", True)) and len(subtypes) > 1:
        review.append(_finding("review_only", "mixed_audio_subtypes", detail=subtypes))

    device = _device_review(rows, speaker_metadata_csv, protocol)
    if device.get("review_recommended"):
        review.append(_finding("review_only", "speaker_group_device_confound_possible", detail=device))

    hard_codes = Counter(_text(item.get("code")) for item in hard)
    review_codes = Counter(_text(item.get("code")) for item in review)
    if hard:
        decision = "collection_blocked"
    elif review:
        decision = "collection_ready_with_review"
    else:
        decision = "collection_ready"

    return {
        "schema": AUDIT_SCHEMA,
        "protocol_schema": protocol.get("schema_version"),
        "decision": decision,
        "manifest": str(manifest_csv),
        "audio_root": str(root),
        "sample_count": len(rows),
        "existing_audio_count": sum(bool(item.get("exists")) for item in sample_reports),
        "decodable_audio_count": len(decodable),
        "manifest_validation": manifest_validation,
        "hard_blocker_count": len(hard),
        "hard_blocker_code_counts": dict(sorted(hard_codes.items())),
        "hard_blockers": hard,
        "review_finding_count": len(review),
        "review_code_counts": dict(sorted(review_codes.items())),
        "review_findings": review,
        "format_summary": {
            "sample_rates": sample_rates,
            "channel_counts": channel_counts,
            "subtypes": subtypes,
        },
        "device_review": device,
        "sample_reports": sample_reports,
        "context_audio_hashed_for_response_leakage": False,
        "automatic_sample_exclusion_performed": False,
        "product_score_changed": False,
        "score_contract_changed": False,
        "interpretation": "corpus_integrity_and_collection_review_not_learner_ability_or_construct_validity",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_csv")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--speaker-metadata", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit(
        args.manifest_csv,
        args.audio_root,
        protocol_json=args.protocol,
        speaker_metadata_csv=args.speaker_metadata,
    )
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["decision"] != "collection_blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
