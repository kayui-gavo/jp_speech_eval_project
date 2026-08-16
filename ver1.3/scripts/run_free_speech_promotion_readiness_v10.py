#!/usr/bin/env python3
"""Run the free-speech promotion-readiness workflow without changing ProductScore.

Stages:
1. validate the private sample manifest;
2. run the real product-condition v8 acceptance path (no gold transcript);
3. export frozen v5 candidates plus v10 speech-duration metadata;
4. preflight actual held learner short/long and task coverage before rating spend;
5. build a held-only blinded listener pack when rater ids are supplied;
6. when completed ratings are supplied, normalize them and execute v5 + v10 gates.

The runner is deliberately allowed to stop at ``collection_coverage_insufficient``
or ``awaiting_human_ratings``. Neither state is permission to promote from
machine-only evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from analyze_free_speech_promotion_v10 import analyze as analyze_v10
from assess_free_speech_machine_coverage_v10 import assess as assess_machine_coverage
from build_free_speech_listener_pack_v3 import build_listener_pack, _write_csv
from export_free_speech_v10_evidence import export_file as export_v10_evidence
from normalize_consumer_ratings_v3 import normalize_file
from run_free_speech_acceptance_v8 import run_acceptance
from validate_free_speech_sample_manifest import validate_manifest_file


RUN_SCHEMA = "free_speech_promotion_readiness_v10"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_PROTOCOL = ROOT / "data" / "research_eval" / "free_speech_v5_promotion_protocol.json"
DEFAULT_V10_PROTOCOL = ROOT / "data" / "research_eval" / "free_speech_v10_consumer_promotion_protocol.json"


def _read_csv(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _write_manifest(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_held_listener_pack(
    manifest_csv: str | Path,
    out_dir: Path,
    *,
    raters: list[str],
    ratings_per_presentation: int,
) -> dict[str, Any]:
    rows, fields = _read_csv(manifest_csv)
    held_rows = [row for row in rows if str(row.get("split") or "").strip() == "held"]
    held_manifest = out_dir / "private_held_manifest_v10.csv"
    _write_manifest(held_manifest, held_rows, fields)

    listener_rows, asset_rows, report = build_listener_pack(
        held_manifest,
        rater_ids=raters,
        ratings_per_presentation=ratings_per_presentation,
    )
    listener_out = out_dir / "listener_pack_held_v10.csv"
    asset_out = out_dir / "private_asset_map_held_v10.csv"
    listener_fields = list(listener_rows[0].keys()) if listener_rows else [
        "rater_id", "sample_id", "presentation_id", "presentation_variant", "task_mode",
        "audio_asset_id", "context_type", "context_id", "context_text", "context_audio_asset_id",
        "assigned_constructs", "analyzable_yes_no", "clarity_comprehensibility_1to7",
        "fluency_1to7", "rhythm_naturalness_1to7", "intonation_utterance_naturalness_1to7",
        "intonation_contextual_appropriateness_1to7", "context_presented_yes_no", "timestamp", "pack_schema",
    ]
    asset_fields = list(asset_rows[0].keys()) if asset_rows else [
        "sample_id", "audio_asset_id", "source_audio_path", "context_audio_asset_id",
        "source_context_audio_path", "private_only",
    ]
    _write_csv(listener_out, listener_rows, listener_fields)
    _write_csv(asset_out, asset_rows, asset_fields)
    return {
        **report,
        "split": "held",
        "held_manifest": str(held_manifest),
        "listener_csv": str(listener_out),
        "private_asset_map_csv": str(asset_out),
    }


def run(
    manifest_csv: str | Path,
    output_dir: str | Path,
    *,
    audio_root: str | Path | None = None,
    asr_model: str = "small",
    asr_provider: str = "auto",
    raters: list[str] | None = None,
    ratings_per_presentation: int = 5,
    completed_ratings_csv: str | Path | None = None,
    base_protocol_json: str | Path = DEFAULT_BASE_PROTOCOL,
    v10_protocol_json: str | Path = DEFAULT_V10_PROTOCOL,
    resume: bool = True,
) -> dict[str, Any]:
    validation = validate_manifest_file(manifest_csv)
    if not validation.get("ok"):
        raise ValueError("sample manifest failed validation: " + ";".join(validation.get("errors") or []))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acceptance_dir = out_dir / "machine_acceptance"
    acceptance = run_acceptance(
        manifest_csv,
        acceptance_dir,
        audio_root=audio_root,
        asr_model=asr_model,
        asr_provider=asr_provider,
        resume=resume,
    )
    evidence_csv = out_dir / "machine_evidence_v10.csv"
    evidence_export = export_v10_evidence(acceptance["batch_jsonl"], evidence_csv)

    machine_coverage = assess_machine_coverage(evidence_csv, v10_protocol_json)
    machine_coverage_json = out_dir / "machine_coverage_preflight_v10.json"
    machine_coverage_json.write_text(
        json.dumps(machine_coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    collection_ready = bool(machine_coverage.get("collection_structure_ready_for_final_listener_pack"))

    listener_pack = None
    if raters:
        listener_pack = _build_held_listener_pack(
            manifest_csv,
            out_dir,
            raters=raters,
            ratings_per_presentation=ratings_per_presentation,
        )
        listener_pack["machine_coverage_preflight"] = "ready" if collection_ready else "provisional_undercovered"
        listener_pack["final_freeze_ready"] = collection_ready

    normalized_ratings_csv = None
    promotion = None
    if completed_ratings_csv is not None:
        normalized_ratings_csv = out_dir / "normalized_human_ratings_v10.csv"
        normalize_file(completed_ratings_csv, manifest_csv, normalized_ratings_csv)
        promotion = analyze_v10(
            normalized_ratings_csv,
            evidence_csv,
            base_protocol_json,
            v10_protocol_json,
        )
        stage = "promotion_readiness_evaluated"
    else:
        stage = "awaiting_human_ratings" if collection_ready else "collection_coverage_insufficient"

    if promotion is not None:
        next_action = "only pass candidates may move to a separate score-changing A/B branch"
    elif collection_ready:
        next_action = "collect the blinded held-set ratings, then rerun with --ratings-csv"
    else:
        missing = machine_coverage.get("missing_or_undercovered_structure") or []
        next_action = (
            "collect or replace held learner recordings until the actual endpointed coverage passes before freezing the final listener pack"
            + (f"; undercovered: {','.join(str(item) for item in missing)}" if missing else "")
        )

    report = {
        "schema": RUN_SCHEMA,
        "stage": stage,
        "manifest_validation": validation,
        "machine_acceptance": acceptance,
        "evidence_export": evidence_export,
        "evidence_csv": str(evidence_csv),
        "machine_coverage_preflight": machine_coverage,
        "machine_coverage_preflight_json": str(machine_coverage_json),
        "listener_pack": listener_pack,
        "normalized_human_ratings_csv": None if normalized_ratings_csv is None else str(normalized_ratings_csv),
        "promotion_analysis": promotion,
        "product_score_changed": False,
        "score_contract_changed": False,
        "gold_transcript_used_for_scoring": False,
        "decision": None if promotion is None else promotion.get("overall_v10_promotion_readiness"),
        "next_action": next_action,
    }
    summary_path = out_dir / "promotion_readiness_summary_v10.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["summary_json"] = str(summary_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--asr-provider", default="auto")
    parser.add_argument("--raters", default="", help="comma-separated anonymized rater ids")
    parser.add_argument("--ratings-per-presentation", type=int, default=5)
    parser.add_argument("--ratings-csv", default=None, help="completed blinded listener ratings")
    parser.add_argument("--base-protocol", default=str(DEFAULT_BASE_PROTOCOL))
    parser.add_argument("--v10-protocol", default=str(DEFAULT_V10_PROTOCOL))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    report = run(
        args.manifest_csv,
        args.out_dir,
        audio_root=args.audio_root,
        asr_model=args.asr_model,
        asr_provider=args.asr_provider,
        raters=[item.strip() for item in args.raters.split(",") if item.strip()],
        ratings_per_presentation=args.ratings_per_presentation,
        completed_ratings_csv=args.ratings_csv,
        base_protocol_json=args.base_protocol,
        v10_protocol_json=args.v10_protocol,
        resume=not args.no_resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
