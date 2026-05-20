#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.prosody_abx import AbxTrialResult, score_abx, summarize_abx
from jp_speech_eval.prosody_dataset import ProsodyPair, ProsodyTrial, iter_trials, load_prosody_dataset
from jp_speech_eval.representation_extractor import RepresentationExtractor, parse_layer_spec
from jp_speech_eval.tts_backends import synthesize_reference


def _resolve_audio(audio_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else audio_root / path


def _bootstrap_trial(pair: ProsodyPair, audio_root: Path, mode: str, tts_backend: str) -> ProsodyTrial:
    """Create deterministic TTS files only to smoke-test the pipeline."""

    out_dir = audio_root / "_bootstrap_tts" / pair.pair_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "out_of_context":
        text_a = pair.word_a
        text_b = pair.word_b
    else:
        text_a = pair.carrier_a
        text_b = pair.carrier_b

    paths = {
        "a": out_dir / f"{mode}_a.wav",
        "b": out_dir / f"{mode}_b.wav",
        "x": out_dir / f"{mode}_x_same_as_a.wav",
    }
    for key, text in (("a", text_a), ("b", text_b), ("x", text_a)):
        if not paths[key].exists():
            synth = synthesize_reference(text, sr=16000, backend=tts_backend)
            sf.write(str(paths[key]), synth.y, 16000)
    return ProsodyTrial(
        trial_id=f"{pair.pair_id}_{mode}_tts_bootstrap",
        a_audio=str(paths["a"].relative_to(audio_root)),
        b_audio=str(paths["b"].relative_to(audio_root)),
        x_audio=str(paths["x"].relative_to(audio_root)),
        x_label="A",
        context_mode=mode,
        reference_source="tts_bootstrap_sanity_check",
        notes="deterministic TTS bootstrap; not valid for scientific ranking",
    )


def _collect_trials(dataset_path: Path, audio_root: Path, mode: str, bootstrap_tts: bool, tts_backend: str) -> List[Tuple[ProsodyPair, ProsodyTrial]]:
    dataset = load_prosody_dataset(dataset_path)
    trials = list(iter_trials(dataset, mode=mode))
    explicit_pair_ids = {pair.pair_id for pair, _trial in trials}
    if bootstrap_tts:
        for pair in dataset.pairs:
            if pair.pair_id not in explicit_pair_ids:
                trials.append((pair, _bootstrap_trial(pair, audio_root, mode, tts_backend)))
    return trials


def _trial_payload(pair: ProsodyPair, trial: ProsodyTrial, model_name: str, layer_id: int, result: AbxTrialResult) -> Dict[str, Any]:
    payload = {
        "pair_id": pair.pair_id,
        "contrast_type": pair.contrast_type,
        "trial_id": trial.trial_id,
        "context_mode": trial.context_mode,
        "reference_source": trial.reference_source,
        "model_name": model_name,
        "layer_id": int(layer_id),
        "target_region": pair.target_region,
        "accent_a": pair.accent_a,
        "accent_b": pair.accent_b,
        "notes": trial.notes or pair.notes,
    }
    payload.update(result.to_dict())
    return payload


def _choose_recommendation(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    viable = [row for row in rows if row["n_trials"] > 0]
    if not viable:
        return {
            "recommended_model": None,
            "recommended_layer": None,
            "reason": "no_successful_trials",
            "fallback_model": "mfcc",
        }
    ranked = sorted(
        viable,
        key=lambda row: (
            -float(row["abx_accuracy"]),
            -float(row["mean_margin"]),
            float(row["std_margin"]),
            bool(row["low_confidence"]),
        ),
    )
    best = ranked[0]
    reason = (
        f"highest ABX accuracy ({best['abx_accuracy']:.3f}), "
        f"mean margin {best['mean_margin']:.4f}, std {best['std_margin']:.4f}"
    )
    if best["low_confidence"]:
        reason += "; low-confidence result because the current trial set is small or bootstrap-only"
    fallback = "mfcc" if best["model_name"] != "mfcc" else None
    return {
        "recommended_model": best["model_name"],
        "recommended_layer": best["layer_id"],
        "reason": reason,
        "fallback_model": fallback,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = Path(args.dataset)
    audio_root = Path(args.audio_root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    trials = _collect_trials(dataset_path, audio_root, args.mode, args.bootstrap_tts, args.tts_backend)
    if not trials:
        raise SystemExit("No trials found. Add native trials to the dataset or pass --bootstrap-tts.")

    extractor = RepresentationExtractor(device=args.device)
    per_trial: List[Dict[str, Any]] = []
    grouped: Dict[tuple[str, int, str], List[AbxTrialResult]] = defaultdict(list)
    group_sources: Dict[tuple[str, int, str], List[str]] = defaultdict(list)
    errors: List[Dict[str, Any]] = []

    for model_name in args.models:
        try:
            layers = parse_layer_spec(args.layers, extractor, model_name)
        except Exception as exc:
            errors.append({"model_name": model_name, "layer_id": None, "error": f"{type(exc).__name__}: {exc}"})
            continue
        for layer_id in layers:
            for pair, trial in trials:
                try:
                    a_path = _resolve_audio(audio_root, trial.a_audio)
                    b_path = _resolve_audio(audio_root, trial.b_audio)
                    x_path = _resolve_audio(audio_root, trial.x_audio)
                    a = extractor.extract(a_path, model_name, layer_id).values
                    b = extractor.extract(b_path, model_name, layer_id).values
                    x = extractor.extract(x_path, model_name, layer_id).values
                    if trial.x_label == "B":
                        a, b = b, a
                    result = score_abx(a, b, x, metric=args.metric)
                    per_trial.append(_trial_payload(pair, trial, model_name, layer_id, result))
                    key = (model_name, int(layer_id), pair.contrast_type)
                    grouped[key].append(result)
                    group_sources[key].append(trial.reference_source)
                except Exception as exc:
                    errors.append({
                        "model_name": model_name,
                        "layer_id": int(layer_id),
                        "pair_id": pair.pair_id,
                        "trial_id": trial.trial_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    aggregate_rows: List[Dict[str, Any]] = []
    for (model_name, layer_id, contrast_type), results in grouped.items():
        summary = summarize_abx(results)
        sources = group_sources[(model_name, layer_id, contrast_type)]
        low_confidence = len(results) < args.min_trials_for_confidence or any(source != "native" for source in sources)
        notes: List[str] = []
        if len(results) < args.min_trials_for_confidence:
            notes.append(f"n_trials<{args.min_trials_for_confidence}")
        if any(source != "native" for source in sources):
            notes.append("contains_non_native_or_bootstrap_audio")
        aggregate_rows.append({
            "model_name": model_name,
            "layer_id": int(layer_id),
            "contrast_type": contrast_type,
            **summary,
            "low_confidence": bool(low_confidence),
            "notes": ";".join(notes),
        })

    aggregate_rows.sort(key=lambda row: (row["model_name"], row["layer_id"], row["contrast_type"]))
    recommendation = _choose_recommendation(aggregate_rows)

    csv_path = results_dir / "prosody_abx_layer_selection.csv"
    json_path = results_dir / "prosody_abx_layer_selection.json"
    fieldnames = [
        "model_name",
        "layer_id",
        "contrast_type",
        "n_trials",
        "abx_error_rate",
        "abx_accuracy",
        "mean_margin",
        "std_margin",
        "low_confidence",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    payload = {
        "config": {
            "dataset": str(dataset_path),
            "audio_root": str(audio_root),
            "models": list(args.models),
            "layers": args.layers,
            "mode": args.mode,
            "metric": args.metric,
            "bootstrap_tts": bool(args.bootstrap_tts),
        },
        "recommendation": recommendation,
        "aggregate_results": aggregate_rows,
        "per_trial_results": per_trial,
        "errors": errors,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "recommendation": recommendation,
        "n_aggregate_rows": len(aggregate_rows),
        "n_trial_rows": len(per_trial),
        "n_errors": len(errors),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select prosody-sensitive representation layers with ABX diagnostics.")
    parser.add_argument("--dataset", default="data/prosody_minimal_pairs.json")
    parser.add_argument("--audio-root", default="data/prosody_audio")
    parser.add_argument("--models", nargs="+", default=["mfcc"])
    parser.add_argument("--layers", default="all", help="all, one layer id, or comma-separated ids")
    parser.add_argument("--mode", choices=["in_context", "out_of_context"], default="in_context")
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--bootstrap-tts", action="store_true")
    parser.add_argument("--tts-backend", default="pyopenjtalk")
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-trials-for-confidence", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
