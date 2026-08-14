#!/usr/bin/env python3
"""Benchmark Japanese phone evidence on already-available corpora only.

The script never asks for new recordings. It scans existing audit manifests,
runs the pinned Japanese phone-CTC backend only for files that already exist on
disk, and reports missing paths as skipped. This is intended for local research
workspaces where JANON/JVS may be mounted outside Git.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.audio_features import load_audio  # noqa: E402
from jp_speech_eval.japanese_phoneme_gop import JapanesePhoneCtcBackend  # noqa: E402
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.segmentation_free_gop import evaluate_backend_fgop_sf_sd_shadow  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402


FIXED_MANIFEST = ROOT / "data" / "audit" / "fixed_reference_manifest_v0.csv"
NATIVE_BANK = ROOT / "data" / "audit" / "native_reference_bank.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "phone_gop_existing_data_benchmark_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Existing-data-only Japanese phone-GOP benchmark")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-sf-phones", type=int, default=18, help="Only run enumerated SD features for short targets")
    return parser.parse_args()


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _items() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    with FIXED_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            items.append({
                "sample_id": str(row.get("sample_id") or ""),
                "audio_path": str(row.get("audio_path") or ""),
                "target_text": str(row.get("target_text") or ""),
                "group": str(row.get("audio_type") or "fixed_manifest"),
                "source": "fixed_reference_manifest_v0",
            })
    with NATIVE_BANK.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            items.append({
                "sample_id": "native_" + str(row.get("reference_index") or "") + "_" + str(row.get("normalized_kana") or ""),
                "audio_path": str(row.get("wav_path") or ""),
                "target_text": str(row.get("target_text") or ""),
                "group": "janon_native_isolated",
                "source": "native_reference_bank",
            })
    return items


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "evaluated":
            by_group.setdefault(str(row.get("group") or "unknown"), []).append(row)
    output: dict[str, Any] = {}
    for group, values in sorted(by_group.items()):
        seq = [
            float(value)
            for row in values
            if (value := _finite(row.get("ctc_forward_logprob_per_frame"))) is not None
        ]
        sf = [
            float(value)
            for row in values
            if (value := _finite(row.get("sf_best_noncanonical_lpr_min"))) is not None
        ]
        output[group] = {
            "n": len(values),
            "ctc_forward_logprob_per_frame_mean": statistics.mean(seq) if seq else None,
            "ctc_forward_logprob_per_frame_median": statistics.median(seq) if seq else None,
            "sf_best_noncanonical_lpr_min_mean": statistics.mean(sf) if sf else None,
            "sf_evaluated_n": len(sf),
        }
    return output


def main() -> None:
    args = parse_args()
    backend = JapanesePhoneCtcBackend(
        device=args.device,
        local_files_only=not args.allow_download,
    )
    rows: list[dict[str, Any]] = []

    for item in _items():
        path = _resolve(item["audio_path"])
        record: dict[str, Any] = {**item, "resolved_audio_path": str(path)}
        if not path.exists():
            record["status"] = "skipped_missing_audio"
            rows.append(record)
            continue
        text = item["target_text"].strip()
        if not text:
            record["status"] = "skipped_missing_target_text"
            rows.append(record)
            continue
        try:
            target = build_japanese_target_evidence(text)
            audio = load_audio(str(path), sr=16000)
            speech, region = trim_to_speech(audio.y, audio.sr)
            result = backend.evaluate(speech, target.phones, sr=audio.sr)
            record.update({
                "status": "evaluated" if result.available else "gop_unavailable",
                "speech_region": region.to_dict(),
                "frontend_distribution": target.frontend_distribution,
                "phones": target.phones,
                "phone_count": len(target.phones),
                "gop_available": result.available,
                "gop_warnings": result.warnings,
                "ctc_forward_logprob_per_frame": result.summary.get("ctc_forward_logprob_per_frame"),
                "single_frame_support_ratio": result.summary.get("single_frame_support_ratio"),
            })
            clean_phone_count = len([p for p in target.phones if p not in {"pau", "sil"}])
            if result.available and clean_phone_count <= int(args.max_sf_phones):
                sf = evaluate_backend_fgop_sf_sd_shadow(backend, speech, target.phones, sr=audio.sr)
                record.update({
                    "sf_available": sf.available,
                    "sf_method": sf.method,
                    "sf_best_noncanonical_lpr_min": sf.summary.get("best_noncanonical_lpr_min"),
                    "sf_noncanonical_wins": sf.summary.get("phones_where_noncanonical_outscores_canonical"),
                    "sf_warnings": sf.warnings,
                })
            else:
                record["sf_status"] = "skipped_target_too_long" if clean_phone_count > int(args.max_sf_phones) else "skipped_gop_unavailable"
        except Exception as exc:
            record.update({
                "status": "evaluation_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        rows.append(record)

    payload = {
        "schema": "phone_gop_existing_data_benchmark_v1",
        "new_human_recordings_used": False,
        "human_recording_gate_changed": False,
        "score_mapped": False,
        "product_calibrated": False,
        "total_manifest_rows": len(rows),
        "evaluated_rows": sum(row.get("status") == "evaluated" for row in rows),
        "missing_audio_rows": sum(row.get("status") == "skipped_missing_audio" for row in rows),
        "error_rows": sum(row.get("status") == "evaluation_error" for row in rows),
        "group_summary": _group_summary(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    print(f"evaluated existing audio: {payload['evaluated_rows']} / {payload['total_manifest_rows']}")
    print(f"missing audio skipped: {payload['missing_audio_rows']}")
    print("HUMAN RECORDING GATE: UNCHANGED / BLOCKED")


if __name__ == "__main__":
    main()
