#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.audio_features import median_f0_by_mora
from jp_speech_eval.content_match import _kana_similarity
from jp_speech_eval.scoring import score_weak_reference_native_likeness
from jp_speech_eval.sentence_cache import load_sentence_cache
from jp_speech_eval.text_frontend import build_text_info


PROVIDERS = (
    "current_pyopenjtalk",
    "pyopenjtalk_run_marine_if_available",
    "openai_gpt4o_mini_tts_candidate",
    "edge_or_other_tts_candidate",
    "human_native_reference_oracle",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _cache_metrics(prefix: Path, *, target_text: str) -> Dict[str, Any]:
    cache = load_sentence_cache(prefix)
    f0_by_mora = median_f0_by_mora(cache.ref_f0_times, cache.ref_f0, cache.meta.ref_mora_boundaries)
    _score, _feedback, weak = score_weak_reference_native_likeness(
        f0_by_mora=f0_by_mora,
        boundaries=cache.meta.ref_mora_boundaries,
        is_question=cache.meta.is_question,
    )
    target = build_text_info(target_text)
    wav_path = prefix.with_suffix(".ref.wav")
    timing_method = str(cache.meta.ref_boundary_method or "")
    timing_quality = "non_fallback" if "fallback" not in timing_method and "equal" not in timing_method else "approximate"
    return {
        "audio_readable": wav_path.exists(),
        "audio_path": str(wav_path) if wav_path.exists() else None,
        "sample_rate": cache.meta.sr,
        "duration_sec": round(float(cache.meta.ref_duration_sec), 4),
        "f0_coverage": weak.get("f0_coverage"),
        "mora_timing_quality": timing_quality,
        "mora_timing_source": timing_method,
        "pitch_range_log": weak.get("utterance_f0_range_log"),
        "local_pitch_movement": weak.get("local_pitch_movement"),
        "final_intonation_score": weak.get("phrase_final_intonation_score"),
        "g2p_kana_match": round(float(_kana_similarity(target.kana, cache.meta.kana)), 4),
        "openjtalk_full_context_available": bool(cache.meta.frontend_raw),
        "reference_source": cache.meta.reference_source,
    }


def _provider_row(item: dict[str, str], provider: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "case_id": item["case_id"],
        "target_text": item["target_text"],
        "focus": item["focus"],
        "provider_slot": provider,
        "external_provider_called": False,
        "result_source": "offline_inventory_or_missing_fixture",
        "fixture_available": False,
        "audio_readable": None,
        "audio_path": None,
        "sample_rate": None,
        "duration_sec": None,
        "f0_coverage": None,
        "mora_timing_quality": None,
        "mora_timing_source": None,
        "pitch_range_log": None,
        "local_pitch_movement": None,
        "final_intonation_score": None,
        "g2p_kana_match": None,
        "openjtalk_full_context_available": None,
        "provenance": "synthetic_tts" if provider != "human_native_reference_oracle" else "verified_human_oracle",
        "sidecar_reliability": "weak" if provider != "human_native_reference_oracle" else "reliable_if_verified_fixture_exists",
        "strong_pitch_reference_allowed": False,
        "user_facing_status": "weak_demo_reference" if provider != "human_native_reference_oracle" else "strict_reference_only_if_verified",
        "notes": item.get("notes"),
    }
    prefix_text = ""
    if provider == "current_pyopenjtalk":
        prefix_text = str(item.get("cache_prefix") or "")
    elif provider == "human_native_reference_oracle":
        prefix_text = str(item.get("human_oracle_prefix") or "")
    if prefix_text:
        prefix = ROOT / prefix_text
        try:
            base.update(_cache_metrics(prefix, target_text=item["target_text"]))
            base["fixture_available"] = True
            base["result_source"] = "existing_local_cache_replay"
            if provider == "human_native_reference_oracle":
                base["strong_pitch_reference_allowed"] = True
                base["sidecar_reliability"] = "reliable"
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            base["notes"] = f"{item.get('notes')}; fixture load failed: {type(exc).__name__}"
    if provider == "pyopenjtalk_run_marine_if_available":
        base["notes"] = f"{item.get('notes')}; run_marine candidate not invoked"
    elif provider in {"openai_gpt4o_mini_tts_candidate", "edge_or_other_tts_candidate"}:
        base["notes"] = f"{item.get('notes')}; external candidate output not supplied"
    return base


def audit_rows(plan_path: Path) -> list[dict[str, Any]]:
    return [_provider_row(item, provider) for item in _read(plan_path) for provider in PROVIDERS]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, Any]], plan_path: Path) -> None:
    available = [row for row in rows if row["fixture_available"]]
    lines = [
        "# TTS reference quality audit",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- plan: `{plan_path}`",
        f"- matrix rows: {len(rows)}",
        f"- local fixture rows measured: {len(available)}",
        "- external provider calls: 0",
        "- status: provider matrix and local-cache replay; missing external outputs remain explicitly unavailable.",
        "",
        "## Available local fixtures",
        "",
        "| case | provider | duration | F0 coverage | timing | pitch range | movement | kana match | reliability |",
        "|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in available:
        lines.append(
            f"| {row['case_id']} | {row['provider_slot']} | {row['duration_sec']} | {row['f0_coverage']} | "
            f"{row['mora_timing_quality']} | {row['pitch_range_log']} | {row['local_pitch_movement']} | "
            f"{row['g2p_kana_match']} | {row['sidecar_reliability']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Better TTS may improve reference-audio naturalness, shadowing UX, phrase timing, and how easy the demo is to imitate.",
        "- TTS output always remains synthetic/unverified and weak for scoring provenance, regardless of subjective quality.",
        "- Existing verified JVS human audio is an oracle comparator, not a TTS provider result.",
        "- Better TTS alone cannot create human/native pitch ground truth, calibrate learner scores, or validate wrong accent drops.",
        "- Real A/B requires generated WAV files for the same text/voice/style conditions, followed by the offline metrics in this matrix and human listening review.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local/replayed TTS reference quality without provider calls.")
    parser.add_argument("--plan", default="data/tts_benchmark_plan.csv")
    parser.add_argument("--out-csv", default="results/calibration/tts_reference_quality_audit.csv")
    parser.add_argument("--out-report", default="reports/tts_reference_quality_audit.md")
    args = parser.parse_args()
    plan = ROOT / args.plan
    rows = audit_rows(plan)
    _write_csv(ROOT / args.out_csv, rows)
    _write_report(ROOT / args.out_report, rows, plan)
    print(f"wrote {ROOT / args.out_csv}")
    print(f"wrote {ROOT / args.out_report}")


if __name__ == "__main__":
    main()
