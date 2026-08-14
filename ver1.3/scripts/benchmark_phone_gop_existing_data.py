#!/usr/bin/env python3
"""Benchmark criterion-ready Japanese phone evidence on existing corpora only.

This script is deliberately **local-data only**: it never downloads JANON/JVS
and never asks for new recordings.  It scans existing audit manifests, discovers
matching JANON isolated-word recordings when a JANON corpus is already present
in the research workspace, runs the pinned Beatrice phone-CTC backend, and
emits the same strict criterion-ready `{LPP, LPR, graph GOP, Occ(i)}` bundles
used by Stage-0 preflights.

Missing corpus paths are skipped.  Non-native JANON items are not assigned a
pronunciation-quality label; they are only grouped as existing learner audio.
Nothing is mapped to `/100` and no product score changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.audio_features import load_audio  # noqa: E402
from jp_speech_eval.japanese_phoneme_gop import (  # noqa: E402
    JapanesePhoneCtcBackend,
    project_japanese_ctc_logits,
    sanitize_canonical_phones,
)
from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence  # noqa: E402
from jp_speech_eval.phone_criterion_features import build_phone_criterion_feature_bundle  # noqa: E402
from jp_speech_eval.segmentation_free_gop import compute_enumerated_fgop_sf_sd_features  # noqa: E402
from jp_speech_eval.segmentation_free_gop_norm import compute_segmentation_free_norm_features  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402


FIXED_MANIFEST = ROOT / "data" / "audit" / "fixed_reference_manifest_v0.csv"
NATIVE_BANK = ROOT / "data" / "audit" / "native_reference_bank.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "phone_gop_existing_data_benchmark_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Existing-data-only Japanese phone criterion benchmark")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--allow-download", action="store_true", help="Allow downloading the pinned acoustic model only; corpus audio is never downloaded")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-feature-phones", type=int, default=18, help="Only run enumerated criterion bundle for short targets")
    parser.add_argument("--discover-janon-isolated", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _native_speakers_and_targets() -> tuple[set[str], list[dict[str, str]]]:
    native_speakers: set[str] = set()
    targets: list[dict[str, str]] = []
    with NATIVE_BANK.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            wav_path = str(row.get("wav_path") or "")
            path = Path(wav_path)
            speaker = path.parent.parent.name if len(path.parts) >= 3 else ""
            if speaker:
                native_speakers.add(speaker)
            targets.append(
                {
                    "target_text": str(row.get("target_text") or ""),
                    "normalized_kana": str(row.get("normalized_kana") or ""),
                    "wav_path": wav_path,
                    "filename": path.name,
                }
            )
    # same target appears once per native reference; keep one filename pattern
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in targets:
        suffix = _janon_isolated_suffix(row["filename"])
        if suffix:
            unique[(row["target_text"], suffix)] = row
    return native_speakers, list(unique.values())


def _janon_isolated_suffix(filename: str) -> str | None:
    """Return stable JANON item suffix such as ``i63.wav``."""
    name = Path(str(filename)).name
    if "_i" not in name or not name.lower().endswith(".wav"):
        return None
    return "i" + name.rsplit("_i", 1)[1]


def _janon_root_candidates() -> list[Path]:
    candidates = [
        ROOT / "JANON",
        ROOT.parent / "JANON",
        ROOT.parent.parent / "JANON",
    ]
    seen: set[str] = set()
    output: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        key = str(resolved)
        if key not in seen:
            output.append(resolved)
            seen.add(key)
    return output


def _janon_group(speaker: str, native_speakers: set[str]) -> str:
    return "janon_native_isolated" if speaker in native_speakers else "janon_non_native_isolated_unlabeled_quality"


def _discover_janon_isolated_items(existing_paths: set[str]) -> list[dict[str, str]]:
    """Discover same-target JANON learner/native isolated words already on disk."""
    native_speakers, target_patterns = _native_speakers_and_targets()
    rows: list[dict[str, str]] = []
    for root in _janon_root_candidates():
        audio_root = root / "audio"
        if not audio_root.is_dir():
            continue
        for target in target_patterns:
            suffix = _janon_isolated_suffix(target["filename"])
            if not suffix:
                continue
            pattern = f"*_{{suffix}}".format(suffix=suffix)
            for wav in sorted(audio_root.glob(f"*/isolated/{pattern}")):
                resolved = str(wav.resolve())
                if resolved in existing_paths:
                    continue
                speaker = wav.parent.parent.name
                rows.append(
                    {
                        "sample_id": f"janon_discovered_{speaker}_{suffix[:-4]}",
                        "audio_path": str(wav.resolve()),
                        "target_text": target["target_text"],
                        "group": _janon_group(speaker, native_speakers),
                        "speaker": speaker,
                        "source": "local_JANON_isolated_discovery",
                        "criterion_label_status": (
                            "native_anchor_only" if speaker in native_speakers else "learner_quality_unlabeled"
                        ),
                    }
                )
                existing_paths.add(resolved)
    return rows


def _items(*, discover_janon_isolated: bool = True) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    with FIXED_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            items.append(
                {
                    "sample_id": str(row.get("sample_id") or ""),
                    "audio_path": str(row.get("audio_path") or ""),
                    "target_text": str(row.get("target_text") or ""),
                    "group": str(row.get("audio_type") or "fixed_manifest"),
                    "speaker": "",
                    "source": "fixed_reference_manifest_v0",
                    "criterion_label_status": "manifest_behavior_only_not_phone_correctness",
                }
            )
    with NATIVE_BANK.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            wav_path = str(row.get("wav_path") or "")
            path = Path(wav_path)
            items.append(
                {
                    "sample_id": "native_" + str(row.get("reference_index") or "") + "_" + str(row.get("normalized_kana") or ""),
                    "audio_path": wav_path,
                    "target_text": str(row.get("target_text") or ""),
                    "group": "janon_native_isolated",
                    "speaker": path.parent.parent.name if len(path.parts) >= 3 else "",
                    "source": "native_reference_bank",
                    "criterion_label_status": "native_anchor_only",
                }
            )
    if discover_janon_isolated:
        existing = {str(_resolve(item["audio_path"])) for item in items if item.get("audio_path")}
        items.extend(_discover_janon_isolated_items(existing))
    return items


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _infer_logical_logits(
    backend: JapanesePhoneCtcBackend,
    speech: np.ndarray,
    *,
    sr: int,
) -> tuple[np.ndarray, dict[str, int], int]:
    backend._load()  # pinned research backend; contract checked in _load
    waveform = np.asarray(speech, dtype=np.float32).reshape(-1)
    inputs = backend.processor(waveform, sampling_rate=sr, return_tensors="pt")
    model_inputs = {key: value.to(backend.device) for key, value in inputs.items()}
    with backend._torch.no_grad():
        output = backend.model(**model_inputs)
    raw_logits = output.logits.squeeze(0).detach().cpu().numpy()
    logical_logits, logical_vocab, _projection = project_japanese_ctc_logits(
        raw_logits, backend.vocabulary()
    )
    if "PAD" not in logical_vocab:
        raise RuntimeError("logical blank token missing from Beatrice projection")
    return logical_logits, logical_vocab, int(logical_vocab["PAD"])


def _criterion_bundle(
    backend: JapanesePhoneCtcBackend,
    speech: np.ndarray,
    phones: Iterable[str],
    *,
    sr: int,
) -> dict[str, Any]:
    clean_phones, dropped = sanitize_canonical_phones(list(phones))
    logits, vocab, blank_id = _infer_logical_logits(backend, speech, sr=sr)
    enumerated = compute_enumerated_fgop_sf_sd_features(
        logits,
        clean_phones,
        vocab=vocab,
        blank_id=blank_id,
        model_id=str(backend.model_id),
        revision=str(backend.revision),
    )
    normalized = compute_segmentation_free_norm_features(
        logits,
        clean_phones,
        vocab=vocab,
        blank_id=blank_id,
        model_id=str(backend.model_id),
        revision=str(backend.revision),
    )
    bundle = build_phone_criterion_feature_bundle(enumerated, normalized)
    payload = bundle.to_dict()
    payload["dropped_nonsegmental_target_tokens"] = dropped
    return payload


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
        norm = [
            float(value)
            for row in values
            if (value := _finite(row.get("criterion_normalized_graph_gop_mean"))) is not None
        ]
        output[group] = {
            "n": len(values),
            "criterion_bundle_n": len(norm),
            "ctc_forward_logprob_per_frame_mean": statistics.mean(seq) if seq else None,
            "ctc_forward_logprob_per_frame_median": statistics.median(seq) if seq else None,
            "normalized_graph_gop_mean_of_clip_means": statistics.mean(norm) if norm else None,
            "interpretation": "descriptive_unlabeled_group_distribution_not_pronunciation_quality",
        }
    return output


def main() -> None:
    args = parse_args()
    backend = JapanesePhoneCtcBackend(
        device=args.device,
        local_files_only=not args.allow_download,
    )
    rows: list[dict[str, Any]] = []

    for item in _items(discover_janon_isolated=bool(args.discover_janon_isolated)):
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
            base = backend.evaluate(speech, target.phones, sr=audio.sr)
            record.update(
                {
                    "status": "evaluated" if base.available else "gop_unavailable",
                    "speech_region": region.to_dict(),
                    "frontend_distribution": target.frontend_distribution,
                    "phones": target.phones,
                    "phone_count": len(target.phones),
                    "gop_available": base.available,
                    "gop_warnings": base.warnings,
                    "ctc_forward_logprob_per_frame": base.summary.get("ctc_forward_logprob_per_frame"),
                    "single_frame_support_ratio": base.summary.get("single_frame_support_ratio"),
                }
            )
            clean_phone_count = len([p for p in target.phones if p not in {"pau", "sil"}])
            if base.available and clean_phone_count <= int(args.max_feature_phones):
                bundle = _criterion_bundle(backend, speech, target.phones, sr=audio.sr)
                record["criterion_feature_bundle"] = bundle
                record["criterion_bundle_available"] = bool(bundle.get("available"))
                if bundle.get("available"):
                    norm_values = [
                        float(row["normalized_graph_gop_sf_sd"])
                        for row in bundle.get("rows", [])
                        if _finite(row.get("normalized_graph_gop_sf_sd")) is not None
                    ]
                    occ_values = [
                        float(row["occ_i"])
                        for row in bundle.get("rows", [])
                        if _finite(row.get("occ_i")) is not None
                    ]
                    record["criterion_normalized_graph_gop_mean"] = (
                        statistics.mean(norm_values) if norm_values else None
                    )
                    record["criterion_occ_i_mean"] = statistics.mean(occ_values) if occ_values else None
            else:
                record["criterion_bundle_status"] = (
                    "skipped_target_too_long" if clean_phone_count > int(args.max_feature_phones) else "skipped_gop_unavailable"
                )
        except Exception as exc:
            record.update(
                {
                    "status": "evaluation_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        rows.append(record)

    payload = {
        "schema": "phone_gop_existing_data_benchmark_v2",
        "model_id": str(backend.model_id),
        "revision": str(backend.revision),
        "new_human_recordings_used": False,
        "human_recording_gate_changed": False,
        "corpus_audio_downloaded": False,
        "learner_quality_labels_inferred": False,
        "score_mapped": False,
        "product_calibrated": False,
        "product_score_changed": False,
        "cross_model_raw_averaging_allowed": False,
        "total_manifest_rows": len(rows),
        "evaluated_rows": sum(row.get("status") == "evaluated" for row in rows),
        "criterion_bundle_rows": sum(bool(row.get("criterion_bundle_available")) for row in rows),
        "missing_audio_rows": sum(row.get("status") == "skipped_missing_audio" for row in rows),
        "error_rows": sum(row.get("status") == "evaluation_error" for row in rows),
        "discovered_janon_rows": sum(row.get("source") == "local_JANON_isolated_discovery" for row in rows),
        "group_summary": _group_summary(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    print(f"evaluated existing audio: {payload['evaluated_rows']} / {payload['total_manifest_rows']}")
    print(f"criterion bundles: {payload['criterion_bundle_rows']}")
    print(f"local JANON isolated discoveries: {payload['discovered_janon_rows']}")
    print(f"missing audio skipped: {payload['missing_audio_rows']}")
    print("HUMAN RECORDING GATE: UNCHANGED / BLOCKED")
    print("PRODUCT SCORE: UNCHANGED / RESEARCH FEATURES ONLY")


if __name__ == "__main__":
    main()
