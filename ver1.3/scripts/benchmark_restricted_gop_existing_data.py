#!/usr/bin/env python3
"""Benchmark Japanese restricted-substitution GOP on already-present audio.

No corpus is downloaded and no new recording is requested.  Native/JANON group
labels are provenance only; non-native JANON speech has no inferred
pronunciation-quality label.  The output is intended to compare RPS behavior,
native false-alarm pressure and candidate-space reduction before any use in the
C-end clarity score.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_phone_gop_existing_data as existing  # noqa: E402
from jp_speech_eval.audio_features import load_audio  # noqa: E402
from jp_speech_eval.japanese_phoneme_gop import JapanesePhoneCtcBackend, sanitize_canonical_phones, segmental_competitor_ids  # noqa: E402
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.restricted_segmentation_free_gop import compute_restricted_fgop_sf_sd_features  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/restricted_gop_existing_data_v1.json")
    parser.add_argument("--allow-download", action="store_true", help="Allow pinned model download only; corpus audio is never downloaded")
    parser.add_argument("--device", default=None)
    parser.add_argument("--discover-janon-isolated", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _group_summary(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, list[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "evaluated":
            groups.setdefault(str(row.get("group") or "unknown"), []).append(row)
    output: Dict[str, Any] = {}
    for group, values in sorted(groups.items()):
        negative_rates = [
            float(value)
            for row in values
            if (value := _finite(row.get("noncanonical_outscore_rate"))) is not None
        ]
        gop_means = [
            float(value)
            for row in values
            if (value := _finite(row.get("restricted_gop_mean"))) is not None
        ]
        output[group] = {
            "n": len(values),
            "noncanonical_outscore_rate_mean": statistics.mean(negative_rates) if negative_rates else None,
            "noncanonical_outscore_rate_median": statistics.median(negative_rates) if negative_rates else None,
            "restricted_gop_mean_of_clip_means": statistics.mean(gop_means) if gop_means else None,
            "interpretation": "descriptive_group_distribution_not_pronunciation_quality",
        }
    return output


def main() -> None:
    args = parse_args()
    backend = JapanesePhoneCtcBackend(device=args.device, local_files_only=not args.allow_download)
    rows: list[Dict[str, Any]] = []
    for item in existing._items(discover_janon_isolated=bool(args.discover_janon_isolated)):
        path = existing._resolve(item["audio_path"])
        record: Dict[str, Any] = {**item, "resolved_audio_path": str(path)}
        if not path.exists():
            record["status"] = "skipped_missing_audio"
            rows.append(record)
            continue
        text = str(item.get("target_text") or "").strip()
        if not text:
            record["status"] = "skipped_missing_target_text"
            rows.append(record)
            continue
        try:
            target = build_japanese_target_evidence(text)
            phones, dropped = sanitize_canonical_phones(target.phones)
            audio = load_audio(str(path), sr=16000)
            speech, region = trim_to_speech(audio.y, audio.sr)
            logits, vocab, blank_id = existing._infer_logical_logits(backend, speech, sr=audio.sr)
            result = compute_restricted_fgop_sf_sd_features(
                logits,
                phones,
                vocab=vocab,
                blank_id=blank_id,
                model_id=str(backend.model_id),
                revision=str(backend.revision),
            )
            record.update(
                {
                    "status": "evaluated" if result.available else "restricted_gop_unavailable",
                    "speech_region": region.to_dict(),
                    "phones": phones,
                    "dropped_nonsegmental_target_tokens": dropped,
                    "restricted": result.to_dict(),
                }
            )
            if result.available:
                negative = [row for row in result.rows if row.best_noncanonical_lpr < 0.0]
                unrestricted_count = len(segmental_competitor_ids(vocab, blank_id=blank_id))
                record.update(
                    {
                        "noncanonical_outscore_count": len(negative),
                        "noncanonical_outscore_rate": len(negative) / len(result.rows),
                        "restricted_gop_mean": result.summary.get("gop_sf_sd_mean"),
                        "restricted_candidate_count_mean": result.summary.get("candidate_count_mean"),
                        "unrestricted_candidate_count": unrestricted_count,
                        "restricted_candidate_ratio_vs_unrestricted": (
                            float(result.summary["candidate_count_mean"]) / unrestricted_count
                            if unrestricted_count > 0 else None
                        ),
                    }
                )
        except Exception as exc:
            record.update({"status": "evaluation_error", "error_type": type(exc).__name__, "error": str(exc)})
        rows.append(record)

    payload = {
        "schema": "restricted_gop_existing_data_benchmark_v1",
        "model_id": str(backend.model_id),
        "revision": str(backend.revision),
        "new_human_recordings_used": False,
        "corpus_audio_downloaded": False,
        "learner_quality_labels_inferred": False,
        "negative_margin_is_pronunciation_error": False,
        "score_mapped": False,
        "product_calibrated": False,
        "product_score_changed": False,
        "human_recording_gate_changed": False,
        "evaluated_rows": sum(row.get("status") == "evaluated" for row in rows),
        "missing_audio_rows": sum(row.get("status") == "skipped_missing_audio" for row in rows),
        "error_rows": sum(row.get("status") == "evaluation_error" for row in rows),
        "group_summary": _group_summary(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"evaluated existing audio: {payload['evaluated_rows']}")
    print(f"missing audio skipped: {payload['missing_audio_rows']}")
    print("HUMAN RECORDING GATE: UNCHANGED / BLOCKED")
    print("PRODUCT SCORE: UNCHANGED / RESEARCH FEATURE ONLY")


if __name__ == "__main__":
    main()
