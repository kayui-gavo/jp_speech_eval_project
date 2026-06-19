#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_cross_speaker_prosody_reference import (  # noqa: E402
    _jvs_by_utterance,
    _lab_f0,
    flat_f0,
    f0_coverage,
    low_f0_coverage,
    shuffled_f0,
    wrong_drop_f0,
)
from analyze_janon import _pick_rows, _read_rows, _resolve_audio_path  # noqa: E402
from jp_speech_eval.evaluator import evaluate_utterance  # noqa: E402
from jp_speech_eval.scoring import score_prosody, score_weak_reference_native_likeness  # noqa: E402
from jp_speech_eval.sentence_cache import build_sentence_cache  # noqa: E402
from jp_speech_eval.text_frontend import build_text_info  # noqa: E402


def _safe(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _safe(row.get(key)) for key in fieldnames})


def _summary(values: Iterable[Any]) -> Dict[str, Any]:
    nums: List[float] = []
    for value in values:
        if value in {None, ""}:
            continue
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            continue
    if not nums:
        return {"n": 0, "mean": None, "min": None, "p50": None, "max": None}
    nums = sorted(nums)
    return {
        "n": len(nums),
        "mean": round(statistics.mean(nums), 4),
        "min": round(nums[0], 4),
        "p50": round(statistics.median(nums), 4),
        "max": round(nums[-1], 4),
    }


def _score_weak_row(
    *,
    dataset: str,
    case_name: str,
    utterance_id: str,
    target_text: str,
    speaker: str,
    f0_values: Sequence[float],
    moras: Sequence[str],
    notes: str,
    strict_openjtalk_score: int | None = None,
) -> Dict[str, Any]:
    weak_score, _feedback, weak_details = score_weak_reference_native_likeness(list(f0_values))
    return {
        "dataset": dataset,
        "case_name": case_name,
        "utterance_id": utterance_id,
        "target_text": target_text,
        "speaker": speaker,
        "mora_count": len(moras),
        "weak_prosody_naturalness_score": weak_score,
        "weak_available": weak_details.get("available"),
        "weak_unavailable_reason": weak_details.get("unavailable_reason"),
        "f0_coverage": weak_details.get("f0_coverage"),
        "voiced_mora_count": weak_details.get("voiced_mora_count"),
        "valid_f0_mora_count": weak_details.get("valid_f0_mora_count"),
        "utterance_f0_range_log": weak_details.get("utterance_f0_range_log"),
        "local_pitch_movement": weak_details.get("local_pitch_movement"),
        "transition_smoothness": weak_details.get("transition_smoothness"),
        "flatness_penalty": weak_details.get("flatness_penalty"),
        "strict_openjtalk_prosody_score": strict_openjtalk_score,
        "score_type": "weak_reference_native_likeness",
        "strict_reference_available": False,
        "tone_score_in_core_four": "no",
        "notes": notes,
    }


def audit_jvs(jvs_root: Path, *, max_pairs: int, max_speakers: int | None, sample_rate: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped = _jvs_by_utterance(jvs_root, max_speakers=max_speakers)
    pair_count = 0
    for utterance_id in sorted(grouped):
        items = sorted(grouped[utterance_id], key=lambda item: item["speaker_id"])
        if len(items) < 2:
            continue
        user = items[1]
        text_info = build_text_info(user["target_text"])
        user_f0, meta = _lab_f0(user, text_info.moras, sample_rate=sample_rate)
        if user_f0 is None:
            continue
        strict_score, _strict_fb, _strict_details = score_prosody(
            moras=text_info.moras,
            target_pattern=text_info.target_pitch,
            f0_by_mora=user_f0,
            pitch_target_source=text_info.pitch_target_source,
            is_question=text_info.is_question,
            accent_phrases=text_info.accent_phrases,
        )
        rows.append(_score_weak_row(
            dataset="JVS",
            case_name="jvs_native",
            utterance_id=utterance_id,
            target_text=user["target_text"],
            speaker=user["speaker_id"],
            f0_values=user_f0,
            moras=text_info.moras,
            strict_openjtalk_score=strict_score,
            notes="native audio scored without strict reference contour",
        ))
        rows.append(_score_weak_row(
            dataset="JVS",
            case_name="flat_pitch_control",
            utterance_id=utterance_id,
            target_text=user["target_text"],
            speaker=user["speaker_id"],
            f0_values=flat_f0(user_f0),
            moras=text_info.moras,
            strict_openjtalk_score=None,
            notes="same content/timing with flattened F0",
        ))
        rows.append(_score_weak_row(
            dataset="JVS",
            case_name="shuffled_random_pitch_control",
            utterance_id=utterance_id,
            target_text=user["target_text"],
            speaker=user["speaker_id"],
            f0_values=shuffled_f0(user_f0, seed=sum(ord(ch) for ch in utterance_id)),
            moras=text_info.moras,
            strict_openjtalk_score=None,
            notes="same content/timing with shuffled F0",
        ))
        rows.append(_score_weak_row(
            dataset="JVS",
            case_name="wrong_accent_drop_control",
            utterance_id=utterance_id,
            target_text=user["target_text"],
            speaker=user["speaker_id"],
            f0_values=wrong_drop_f0(user_f0, user_f0, text_info.accent_phrases),
            moras=text_info.moras,
            strict_openjtalk_score=None,
            notes="rough wrong-drop control; known limitation if weakly separated",
        ))
        rows.append(_score_weak_row(
            dataset="JVS",
            case_name="low_f0_coverage_control",
            utterance_id=utterance_id,
            target_text=user["target_text"],
            speaker=user["speaker_id"],
            f0_values=low_f0_coverage(user_f0),
            moras=text_info.moras,
            strict_openjtalk_score=None,
            notes="same content/timing with insufficient F0 coverage",
        ))
        pair_count += 1
        if pair_count >= max_pairs:
            break
    return rows


def audit_janon(janon_root: Path, *, max_rows_per_group: int, sample_rate: int) -> List[Dict[str, Any]]:
    manifest = janon_root / "data.csv"
    if not manifest.exists():
        return [{
            "dataset": "JANON",
            "case_name": "janon_unavailable",
            "notes": "JANON data.csv not found",
        }]
    rows: List[Dict[str, Any]] = []
    all_rows = _read_rows(manifest)
    selected = (
        _pick_rows(all_rows, stimulus_type="isolated", native_language="Japanese", max_rows=max_rows_per_group)
        + _pick_rows(all_rows, stimulus_type="isolated", native_language="English", max_rows=max_rows_per_group)
    )
    for idx, row in enumerate(selected, start=1):
        text = row.get("Stmiulus") or ""
        wav_path = _resolve_audio_path(janon_root, row.get("Path") or "")
        if not text or not wav_path.exists():
            continue
        try:
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
            cache_prefix = ROOT / "outputs" / "weak_reference_native_likeness_janon_cache" / f"txt_{digest}"
            if not cache_prefix.with_suffix(".json").exists() or not cache_prefix.with_suffix(".npz").exists():
                build_sentence_cache(text, cache_prefix, sr=sample_rate, save_reference_wav=False)
            result = evaluate_utterance(
                wav_path=wav_path,
                cache_path=cache_prefix,
                alignment_mode="cached_dtw",
                sample_rate=sample_rate,
                use_content_match=False,
            ).to_dict()
            weak = (result.get("details") or {}).get("weak_reference_native_likeness") or {}
            rows.append({
                "dataset": "JANON",
                "case_name": f"janon_{row.get('Native Language', 'unknown').lower()}",
                "utterance_id": row.get("Path"),
                "target_text": text,
                "speaker": row.get("Speaker"),
                "native_language": row.get("Native Language"),
                "mora_count": len(result.get("moras") or []),
                "weak_prosody_naturalness_score": weak.get("weak_prosody_naturalness_score"),
                "weak_overall_practice_score": weak.get("weak_overall_practice_score"),
                "weak_available": weak.get("available"),
                "weak_unavailable_reason": weak.get("unavailable_reason"),
                "f0_coverage": weak.get("f0_coverage"),
                "utterance_f0_range_log": weak.get("utterance_f0_range_log"),
                "local_pitch_movement": weak.get("local_pitch_movement"),
                "transition_smoothness": weak.get("transition_smoothness"),
                "alignment_mode": result.get("alignment_mode"),
                "strict_openjtalk_prosody_score": result.get("prosody_score"),
                "score_type": weak.get("score_type") or result.get("score_type"),
                "strict_reference_available": result.get("strict_reference_available"),
                "tone_score_in_core_four": "no",
                "notes": "small JANON trend sanity; not calibration",
            })
        except Exception as exc:
            rows.append({
                "dataset": "JANON",
                "case_name": "janon_error",
                "target_text": text,
                "speaker": row.get("Speaker"),
                "native_language": row.get("Native Language"),
                "notes": f"{type(exc).__name__}: {exc}",
            })
    return rows


def _by_case_summary(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Any]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("case_name") or "")].append(row.get(field))
    return {case: _summary(values) for case, values in sorted(grouped.items())}


def write_report(path: Path, rows: Sequence[Mapping[str, Any]], *, jvs_root: Path, janon_root: Path) -> None:
    weak_summary = _by_case_summary(rows, "weak_prosody_naturalness_score")
    overall_summary = _by_case_summary(rows, "weak_overall_practice_score")
    strict_summary = _by_case_summary(rows, "strict_openjtalk_prosody_score")
    lines = [
        "# Weak-reference native-likeness audit",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- jvs_root: `{jvs_root}`",
        f"- janon_root: `{janon_root}`",
        "- scope: diagnostic/practice scoring only; no strict pitch calibration is activated.",
        "",
        "## Weak Prosody Naturalness",
        "",
        "| case | n | mean | min | p50 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case, stats in weak_summary.items():
        lines.append(f"| {case} | {stats['n']} | {stats['mean']} | {stats['min']} | {stats['p50']} | {stats['max']} |")
    lines.extend([
        "",
        "## Weak Overall Practice Score",
        "",
        "| case | n | mean | min | p50 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for case, stats in overall_summary.items():
        lines.append(f"| {case} | {stats['n']} | {stats['mean']} | {stats['min']} | {stats['p50']} | {stats['max']} |")
    lines.extend([
        "",
        "## OpenJTalk Strict Target Comparison",
        "",
        "| case | n | mean | min | p50 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for case, stats in strict_summary.items():
        lines.append(f"| {case} | {stats['n']} | {stats['mean']} | {stats['min']} | {stats['p50']} | {stats['max']} |")
    jvs_native = weak_summary.get("jvs_native", {})
    flat = weak_summary.get("flat_pitch_control", {})
    random = weak_summary.get("shuffled_random_pitch_control", {})
    wrong = weak_summary.get("wrong_accent_drop_control", {})
    low = [row for row in rows if row.get("case_name") == "low_f0_coverage_control"]
    low_unavailable = sum(1 for row in low if row.get("weak_available") in {False, "False", "false"} or row.get("weak_unavailable_reason"))
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- JVS native weak prosody mean: {jvs_native.get('mean')}.",
        f"- Flat control mean: {flat.get('mean')}; random/shuffled control mean: {random.get('mean')}.",
        "- Native is no longer evaluated by strict OpenJTalk contour mismatch in weak-reference practice mode.",
        "- Flat/random controls are not lifted together with native when the weak score is used.",
        f"- Low-F0 unavailable rows: {low_unavailable}/{len(low)}.",
        f"- Wrong-drop mean: {wrong.get('mean')}; this remains a known limitation and is not treated as strict accent correctness.",
        "- OpenJTalk is used only as kana/mora/accent hint in arbitrary-sentence practice, not as reliable pitch target.",
        "- JANON rows are a small trend sanity only, not calibration.",
        "",
        "## Product Policy",
        "",
        "- Arbitrary confirmed text can receive weak-reference native-likeness practice scores.",
        "- Pitch feedback should say 音高变化参考 / may be flat / sentence-final intonation may be unclear.",
        "- It must not claim teacher-grade pitch accent correctness without reliable reference audio.",
        "- Verified fixed-reference scoring remains separate and unchanged.",
        "",
        "## Calibration Readiness",
        "",
        "Not ready. This establishes a safer arbitrary-sentence practice path, but does not calibrate strict pitch accent correctness.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit weak-reference native-likeness practice scoring.")
    parser.add_argument("--jvs-root", default=str(ROOT.parent / "JVS"))
    parser.add_argument("--janon-root", default=str(ROOT.parent / "JANON"))
    parser.add_argument("--max-jvs-pairs", type=int, default=24)
    parser.add_argument("--max-jvs-speakers", type=int, default=3)
    parser.add_argument("--max-janon-per-group", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--out-csv", default="results/calibration/weak_reference_native_likeness_audit.csv")
    parser.add_argument("--out-report", default="reports/weak_reference_native_likeness_audit.md")
    args = parser.parse_args()

    jvs_root = Path(args.jvs_root)
    janon_root = Path(args.janon_root)
    rows = audit_jvs(
        jvs_root,
        max_pairs=args.max_jvs_pairs,
        max_speakers=args.max_jvs_speakers,
        sample_rate=args.sample_rate,
    )
    rows.extend(audit_janon(
        janon_root,
        max_rows_per_group=args.max_janon_per_group,
        sample_rate=args.sample_rate,
    ))
    out_csv = ROOT / args.out_csv
    out_report = ROOT / args.out_report
    _write_csv(out_csv, rows)
    write_report(out_report, rows, jvs_root=jvs_root, janon_root=janon_root)
    print(f"wrote {out_csv}")
    print(f"wrote {out_report}")


if __name__ == "__main__":
    main()
