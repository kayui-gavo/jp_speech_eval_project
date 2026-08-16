#!/usr/bin/env python3
"""Run the frozen v8 free-speech acceptance pipeline end to end.

This is an execution wrapper, not a scorer. It intentionally keeps all model
and score policies in their existing modules and wires together:

1. manifest validation;
2. real product-condition evaluation with ``transcript=None``;
3. current ProductScore + shadow partial-evidence telemetry;
4. descriptive current-vs-candidate analysis;
5. routing/availability summary for valid Japanese and negative controls.

No thresholds are fitted here and no candidate is promoted to ProductScore.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from analyze_partial_evidence_aggregate import analyze_batch
from run_free_speech_validation_batch import run_batch
from validate_free_speech_sample_manifest import validate_manifest_file


ACCEPTANCE_SCHEMA = "free_speech_acceptance_v8"


def _latest_rows(path: Path) -> Dict[str, Mapping[str, Any]]:
    latest: Dict[str, Mapping[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "").strip()
            if sample_id:
                latest[sample_id] = row
    return latest


def _routing_summary(batch_jsonl: Path) -> Dict[str, Any]:
    rows = list(_latest_rows(batch_jsonl).values())
    groups: Dict[str, Dict[str, int]] = {}

    def bucket(row: Mapping[str, Any]) -> str:
        meta = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        expected = str(meta.get("expected_language") or "").strip().lower()
        if expected == "ja":
            return "expected_japanese"
        if expected in {"en", "zh", "other"}:
            return "expected_non_japanese_speech"
        if expected == "non_speech":
            return "expected_nonspeech_control"
        return "expected_language_unknown"

    for row in rows:
        key = bucket(row)
        stats = groups.setdefault(
            key,
            {
                "n": 0,
                "evaluation_ok": 0,
                "evaluation_error": 0,
                "product_score_available": 0,
                "product_no_score": 0,
                "candidate_available": 0,
            },
        )
        stats["n"] += 1
        if str(row.get("status") or "") != "ok":
            stats["evaluation_error"] += 1
            continue
        stats["evaluation_ok"] += 1
        user_score = row.get("user_score") if isinstance(row.get("user_score"), Mapping) else {}
        if bool(user_score.get("score_available")):
            stats["product_score_available"] += 1
        else:
            stats["product_no_score"] += 1
        candidates = row.get("score_candidates") if isinstance(row.get("score_candidates"), Mapping) else {}
        candidate = candidates.get("partial_evidence_aggregate") if isinstance(candidates.get("partial_evidence_aggregate"), Mapping) else {}
        if bool(candidate.get("available")):
            stats["candidate_available"] += 1

    for stats in groups.values():
        n = max(int(stats["n"]), 1)
        stats["product_score_available_rate"] = round(stats["product_score_available"] / n, 6)
        stats["product_no_score_rate"] = round(stats["product_no_score"] / n, 6)
        stats["candidate_available_rate"] = round(stats["candidate_available"] / n, 6)

    japanese = groups.get("expected_japanese") or {}
    non_japanese = groups.get("expected_non_japanese_speech") or {}
    nonspeech = groups.get("expected_nonspeech_control") or {}
    return {
        "latest_sample_count": len(rows),
        "groups": groups,
        "valid_japanese_false_no_score_count": int(japanese.get("product_no_score", 0) or 0),
        "non_japanese_speech_normal_score_count": int(non_japanese.get("product_score_available", 0) or 0),
        "nonspeech_control_normal_score_count": int(nonspeech.get("product_score_available", 0) or 0),
        "interpretation": (
            "language-routing and nonspeech-control failures are counted separately; nonsense/noise subtypes should "
            "remain identifiable in sample/source metadata rather than being pooled into a language error rate"
        ),
    }


def run_acceptance(
    manifest_csv: str | Path,
    output_dir: str | Path,
    *,
    audio_root: str | Path | None = None,
    asr_model: str = "small",
    asr_provider: str = "auto",
    resume: bool = True,
    evaluator: Callable[..., Mapping[str, Any]] | None = None,
    user_policy: Callable[..., Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    manifest_report = validate_manifest_file(manifest_csv)
    if not manifest_report.get("ok"):
        raise ValueError("sample manifest failed validation: " + ";".join(manifest_report.get("errors") or []))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_jsonl = out_dir / "free_speech_validation_v8.jsonl"
    analysis_json = out_dir / "partial_evidence_analysis_v8.json"
    acceptance_json = out_dir / "acceptance_summary_v8.json"

    batch_kwargs: Dict[str, Any] = {
        "audio_root": audio_root,
        "asr_model": asr_model,
        "asr_provider": asr_provider,
        "resume": resume,
    }
    if evaluator is not None:
        batch_kwargs["evaluator"] = evaluator
    if user_policy is not None:
        batch_kwargs["user_policy"] = user_policy

    batch_report = run_batch(manifest_csv, batch_jsonl, **batch_kwargs)
    aggregate_report = analyze_batch(batch_jsonl)
    analysis_json.write_text(
        json.dumps(aggregate_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    routing = _routing_summary(batch_jsonl)

    report = {
        "schema": ACCEPTANCE_SCHEMA,
        "manifest": str(manifest_csv),
        "audio_root": None if audio_root is None else str(audio_root),
        "output_dir": str(out_dir),
        "batch_jsonl": str(batch_jsonl),
        "partial_evidence_analysis_json": str(analysis_json),
        "manifest_validation": manifest_report,
        "batch": batch_report,
        "routing": routing,
        "partial_evidence_analysis": aggregate_report,
        "frozen_product_conditions": {
            "transcript": None,
            "scoring_used_gold_transcript": False,
            "partial_evidence_candidate_user_facing": False,
            "partial_evidence_candidate_product_score_changed": False,
            "threshold_tuning_allowed_on_this_run": False,
        },
        "decision": "collect_evidence_only_no_score_promotion",
        "next_gate": (
            "inspect held real-audio routing/dispersion/channel behavior, then join construct-matched listener ratings; "
            "do not promote from this descriptive report alone"
        ),
    }
    acceptance_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen v8 free-speech acceptance pipeline.")
    parser.add_argument("manifest_csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--asr-provider", default="auto")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    report = run_acceptance(
        args.manifest_csv,
        args.out_dir,
        audio_root=args.audio_root,
        asr_model=args.asr_model,
        asr_provider=args.asr_provider,
        resume=not args.no_resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

    # Per-sample evaluation errors are acceptance evidence and should not be
    # hidden, but a batch with such errors exits non-zero for local automation.
    return 0 if int(report["batch"].get("failed", 0) or 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
