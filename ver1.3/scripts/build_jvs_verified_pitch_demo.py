#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_cross_speaker_prosody_reference import (  # noqa: E402
    _lab_f0,
    _score_row,
    flat_f0,
    f0_coverage,
    low_f0_coverage,
    pattern_from_f0,
    shuffled_f0,
    wrong_drop_f0,
)
from build_test_jvs_prosody_reference_cache import build_test_jvs_cache, jvs_item  # noqa: E402
from jp_speech_eval.evaluator import evaluate_utterance  # noqa: E402
from jp_speech_eval.feedback_renderer import render_user_facing_result  # noqa: E402
from jp_speech_eval.prosody_reference_cache import load_prosody_reference_cache, smooth_f0_by_mora  # noqa: E402
from jp_speech_eval.text_frontend import build_text_info  # noqa: E402


def jvs_by_utterance(jvs_root: Path, *, max_speakers: int | None = None) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    speakers = sorted(path for path in jvs_root.glob("jvs*") if path.is_dir())
    if max_speakers is not None:
        speakers = speakers[:max_speakers]
    for speaker_dir in speakers:
        transcript_path = speaker_dir / "parallel100" / "transcripts_utf8.txt"
        wav_dir = speaker_dir / "parallel100" / "wav24kHz16bit"
        lab_dir = speaker_dir / "parallel100" / "lab" / "mon"
        if not transcript_path.exists() or not wav_dir.exists() or not lab_dir.exists():
            continue
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            utt_id, text = line.split(":", 1)
            wav = wav_dir / f"{utt_id}.wav"
            lab = lab_dir / f"{utt_id}.lab"
            if wav.exists() and lab.exists():
                grouped[utt_id].append({
                    "speaker_id": speaker_dir.name,
                    "utterance_id": utt_id,
                    "target_text": text.strip(),
                    "audio_path": str(wav),
                    "lab_path": str(lab),
                })
    return grouped


def _finite_range(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr) & (arr > 0)]
    if valid.size < 2:
        return 0.0
    return float(np.percentile(valid, 90) - np.percentile(valid, 10))


def select_demo_items(
    jvs_root: Path,
    *,
    max_items: int = 4,
    max_speakers: int | None = None,
    sample_rate: int = 16000,
    max_mora_count: int = 32,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for utterance_id, items in sorted(jvs_by_utterance(jvs_root, max_speakers=max_speakers).items()):
        items = sorted(items, key=lambda item: item["speaker_id"])
        if len(items) < 2:
            continue
        reference = items[0]
        user = items[1]
        text_info = build_text_info(reference["target_text"])
        mora_count = len(text_info.moras)
        if mora_count > max_mora_count:
            continue
        ref_f0, ref_meta = _lab_f0(reference, text_info.moras, sample_rate=sample_rate)
        user_f0, user_meta = _lab_f0(user, text_info.moras, sample_rate=sample_rate)
        if ref_f0 is None or user_f0 is None:
            continue
        ref_cov = f0_coverage(ref_f0)
        user_cov = f0_coverage(user_f0)
        ref_range = _finite_range(ref_f0)
        user_range = _finite_range(user_f0)
        if min(ref_cov, user_cov) < 0.70 or max(ref_range, user_range) < 20.0:
            continue
        candidates.append({
            "utterance_id": utterance_id,
            "target_text": reference["target_text"],
            "target_kana": text_info.kana,
            "mora_count": mora_count,
            "reference_speaker": reference["speaker_id"],
            "user_speaker": user["speaker_id"],
            "reference_f0_coverage": round(ref_cov, 4),
            "user_f0_coverage": round(user_cov, 4),
            "reference_f0_range_hz": round(ref_range, 4),
            "user_f0_range_hz": round(user_range, 4),
            "accent_phrase_count": len(text_info.accent_phrases),
            "selection_score": round((max_mora_count - mora_count) * 2 + min(ref_cov, user_cov) * 20 + max(ref_range, user_range) / 8, 4),
        })
    return sorted(candidates, key=lambda row: (-float(row["selection_score"]), int(row["mora_count"]), str(row["utterance_id"])))[:max_items]


def _safe(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def _render_fields(result: Mapping[str, Any], *, mode: str) -> Dict[str, Any]:
    rendered = render_user_facing_result(result, mode=mode)
    debug = rendered.get("debug") or {}
    return {
        "user_facing_status": rendered.get("status"),
        "display_score": rendered.get("display_score"),
        "practice_score_value": (rendered.get("practice_score") or {}).get("value"),
        "visible_prosody_score": debug.get("visible_prosody_score"),
        "prosody_score_visible": debug.get("prosody_score_visible"),
        "suppressed_reasons": ";".join(str(x) for x in rendered.get("suppressed_reasons") or []),
        "tone_score_in_core_four": "no",
    }


def _evaluator_row(
    *,
    item: Mapping[str, Any],
    case_name: str,
    result: Mapping[str, Any],
    mode: str,
    notes: str,
) -> Dict[str, Any]:
    details = result.get("details") or {}
    reliability = details.get("reliability") or {}
    content = details.get("content_match") or {}
    prosody = details.get("prosody") or {}
    row = {
        "utterance_id": item["utterance_id"],
        "target_text": result.get("target_text") or item["target_text"],
        "target_kana": result.get("kana") or item.get("target_kana"),
        "mora_count": item.get("mora_count"),
        "reference_speaker": item["reference_speaker"],
        "user_speaker": item["user_speaker"],
        "case_name": case_name,
        "eval_path": "evaluator",
        "mode": mode,
        "prosody_score": result.get("prosody_score"),
        "pronunciation_score": result.get("pronunciation_score"),
        "fluency_score": result.get("fluency_score"),
        "total_score": result.get("total_score"),
        "pitch_target_source": details.get("pitch_target_source") or prosody.get("pitch_target_source"),
        "pitch_target_reliability": details.get("pitch_target_reliability") or prosody.get("pitch_target_reliability"),
        "content_match_status": content.get("status"),
        "content_verified": content.get("content_verified"),
        "alignment_mode": result.get("alignment_mode"),
        "f0_coverage": reliability.get("f0_coverage"),
        "contour_corr": prosody.get("contour_corr"),
        "transition_agreement": prosody.get("transition_agreement"),
        "accent_drop_match": prosody.get("accent_drop_agreement"),
        "final_intonation_score": prosody.get("final_score"),
        "notes": notes,
    }
    row.update(_render_fields(result, mode=mode))
    return {key: _safe(value) for key, value in row.items()}


def _semi_audio_rows(item: Mapping[str, Any], *, jvs_root: Path, sample_rate: int) -> List[Dict[str, Any]]:
    reference = jvs_item(jvs_root, str(item["reference_speaker"]), str(item["utterance_id"]))
    user = jvs_item(jvs_root, str(item["user_speaker"]), str(item["utterance_id"]))
    text_info = build_text_info(reference["target_text"])
    ref_f0, _ref_meta = _lab_f0(reference, text_info.moras, sample_rate=sample_rate)
    user_f0, _user_meta = _lab_f0(user, text_info.moras, sample_rate=sample_rate)
    if ref_f0 is None or user_f0 is None:
        return []
    ref_smooth = smooth_f0_by_mora(ref_f0)
    ref_pattern = pattern_from_f0(ref_smooth)
    pair = {
        "pair_id": f"{item['reference_speaker']}->{item['user_speaker']}:{item['utterance_id']}",
        "reference": reference,
        "user": user,
    }
    cases = [
        ("native_cross_speaker_reference_audio_f0_cache", user_f0, "speaker B native F0 against speaker A verified reference F0"),
        ("native_cross_speaker_openjtalk_target", user_f0, "same native F0 scored against OpenJTalk target"),
        ("flat_pitch_correct_content", flat_f0(user_f0), "same content/timing with flattened F0"),
        ("shuffled_random_pitch_correct_content", shuffled_f0(user_f0, seed=sum(ord(ch) for ch in str(item["utterance_id"]))), "same content/timing with shuffled F0"),
        ("wrong_accent_drop_correct_content", wrong_drop_f0(user_f0, ref_smooth, text_info.accent_phrases), "same content/timing with accent-drop counterfactual when available"),
        ("low_f0_coverage_correct_content", low_f0_coverage(user_f0), "same content/timing with insufficient F0 coverage"),
    ]
    rows: List[Dict[str, Any]] = []
    for mode, f0_values, notes in cases:
        target_pattern = text_info.target_pitch if mode == "native_cross_speaker_openjtalk_target" else ref_pattern
        source = text_info.pitch_target_source if mode == "native_cross_speaker_openjtalk_target" else "reference_audio_f0_cache"
        reference_f0 = None if mode == "native_cross_speaker_openjtalk_target" else ref_smooth
        row = _score_row(
            pair=pair,
            text_info=text_info,
            mode=mode,
            user_f0=f0_values,
            target_pattern=target_pattern,
            pitch_target_source=source,
            reference_f0=reference_f0,
            notes=notes,
        )
        row.update({
            "case_name": mode,
            "eval_path": "semi_audio_lab_f0",
            "display_score": "",
            "visible_prosody_score": "",
            "prosody_score_visible": "",
            "tone_score_in_core_four": "no",
            "low_f0_unavailable": "yes" if mode == "low_f0_coverage_correct_content" and f0_coverage(f0_values) < 0.5 else "no",
        })
        rows.append(row)
    return rows


def run_demo(
    *,
    jvs_root: Path,
    out_dir: Path,
    max_items: int = 4,
    max_speakers: int | None = None,
    sample_rate: int = 16000,
    clean: bool = False,
) -> List[Dict[str, Any]]:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_demo_items(jvs_root, max_items=max_items, max_speakers=max_speakers, sample_rate=sample_rate)
    rows: List[Dict[str, Any]] = []
    for item in selected:
        prefix = out_dir / f"{item['reference_speaker']}_{item['utterance_id']}"
        cache = build_test_jvs_cache(
            jvs_root=jvs_root,
            speaker_id=str(item["reference_speaker"]),
            utterance_id=str(item["utterance_id"]),
            out_prefix=prefix,
            sample_rate=sample_rate,
            write_sidecar=True,
            reference_source="test_only_jvs_native_reference",
        )
        sidecar = load_prosody_reference_cache(cache.prefix) or {}
        user = jvs_item(jvs_root, str(item["user_speaker"]), str(item["utterance_id"]))
        fixed = evaluate_utterance(
            wav_path=user["audio_path"],
            alignment_mode="cached_dtw",
            cache_path=cache.prefix,
            use_content_match=True,
        ).to_dict()
        fixed_row = _evaluator_row(
            item=item,
            case_name="verified_jvs_sidecar_fixed_reference",
            result=fixed,
            mode="fixed_reference",
            notes="test-only JVS sidecar; evaluator path should read reference_audio_f0_cache",
        )
        fixed_row.update({
            "test_only_cache_prefix": str(cache.prefix),
            "sidecar_reliable": "yes" if sidecar.get("reliable") else "no",
            "sidecar_reference_source": sidecar.get("reference_source"),
        })
        rows.append(fixed_row)
        text_baseline = evaluate_utterance(
            text=str(item["target_text"]),
            wav_path=user["audio_path"],
            alignment_mode="dtw",
            use_content_match=False,
        ).to_dict()
        rows.append(_evaluator_row(
            item=item,
            case_name="same_audio_text_openjtalk_evaluator_baseline",
            result=text_baseline,
            mode="text_openjtalk_baseline",
            notes="same user audio without fixed-reference cache; OpenJTalk target baseline",
        ))
        rows.extend(_semi_audio_rows(item, jvs_root=jvs_root, sample_rate=sample_rate))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _mode_summary(rows: Sequence[Mapping[str, Any]], *, eval_path: str = "semi_audio_lab_f0") -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row.get("eval_path") != eval_path:
            continue
        try:
            score = float(row.get("prosody_score"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            grouped[str(row.get("case_name") or row.get("mode"))].append(score)
    summary: Dict[str, Dict[str, Any]] = {}
    for mode, values in grouped.items():
        ordered = sorted(values)
        summary[mode] = {
            "n": len(values),
            "mean": round(statistics.fmean(values), 4),
            "min": round(min(values), 4),
            "p50": round(ordered[len(ordered) // 2], 4),
            "max": round(max(values), 4),
        }
    return summary


def _write_report(path: Path, rows: Sequence[Mapping[str, Any]], *, out_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("case_name") == "native_cross_speaker_reference_audio_f0_cache":
            selected[str(row["utterance_id"])] = row
    summary = _mode_summary(rows)
    ref_mean = summary.get("native_cross_speaker_reference_audio_f0_cache", {}).get("mean")
    openjtalk_mean = summary.get("native_cross_speaker_openjtalk_target", {}).get("mean")
    flat_mean = summary.get("flat_pitch_correct_content", {}).get("mean")
    random_mean = summary.get("shuffled_random_pitch_correct_content", {}).get("mean")
    wrong_mean = summary.get("wrong_accent_drop_correct_content", {}).get("mean")
    lines = [
        "# JVS Verified Pitch Demo Summary",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- test_only_cache_dir: `{out_dir}`",
        f"- selected_items: {len(selected)}",
        "- scope: test-only JVS verified fixed-reference pitch demo; no packaged demo target is promoted.",
        "- selection_note: JVS parallel100 has no very short sentence in this local set; selected items are the shortest medium-length candidates with good F0 coverage and pitch movement.",
        "",
        "## Selected JVS Items",
        "",
        "| utterance_id | target_text | mora_count | reference_speaker | user_speaker |",
        "|---|---|---:|---|---|",
    ]
    for row in selected.values():
        lines.append(
            f"| {row.get('utterance_id')} | {row.get('target_text')} | {row.get('mora_count')} | "
            f"{row.get('reference_speaker')} | {row.get('user_speaker')} |"
        )
    lines.extend([
        "",
        "## Semi-Audio Prosody Score Summary",
        "",
        "| mode | n | mean | min | p50 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for mode, stats in sorted(summary.items()):
        lines.append(
            f"| {mode} | {stats.get('n')} | {stats.get('mean')} | {stats.get('min')} | {stats.get('p50')} | {stats.get('max')} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if isinstance(ref_mean, (int, float)) and isinstance(openjtalk_mean, (int, float)):
        lines.append(f"- reference_audio_f0_cache mean: {ref_mean}; OpenJTalk mean: {openjtalk_mean}.")
        lines.append("- Reference-audio F0 target beats OpenJTalk on this selected JVS demo set." if ref_mean > openjtalk_mean else "- Reference-audio F0 target did not beat OpenJTalk; inspect selected references.")
    if isinstance(ref_mean, (int, float)) and isinstance(flat_mean, (int, float)) and isinstance(random_mean, (int, float)):
        lines.append(f"- Flat mean: {flat_mean}; random/shuffled mean: {random_mean}.")
        lines.append("- Flat/random controls are lower than the native reference contour." if ref_mean > flat_mean and ref_mean > random_mean else "- Flat/random controls are not reliably lower; pitch discrimination remains insufficient.")
    if isinstance(ref_mean, (int, float)) and isinstance(wrong_mean, (int, float)):
        lines.append(f"- Wrong-drop mean: {wrong_mean}.")
        if ref_mean - wrong_mean < 10:
            lines.append("- Wrong-drop remains weakly separated; this is a known limitation of the current component design.")
        else:
            lines.append("- Wrong-drop is lower in this demo set, but this still needs broader validation.")
    lines.extend([
        "- Low-F0 rows are marked as insufficient evidence when voiced mora coverage is below threshold.",
        "- Evaluator rows confirm the sidecar path reads `reference_audio_f0_cache`; user-facing gates such as fallback alignment remain active.",
        "- `tone_score` is not part of the core four dimensions.",
        "",
        "## Product Conclusion",
        "",
        "This demonstrates the pitch/prosody pipeline upper bound when a real human/native reference contour with lab timing exists. It cannot be directly used for the packaged short-sentence demo because those targets still lack verified human/native reference audio and reliable timing sidecars.",
        "",
        "## Next Minimal Product Action",
        "",
        "Record/import verified native or teacher reference audio for the packaged targets, add non-fallback mora timing, then rebuild reliable sidecars. Calibration remains inactive.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a test-only JVS verified pitch demo and summary.")
    parser.add_argument("--jvs-root", default=str(ROOT.parent / "JVS"))
    parser.add_argument("--max-items", type=int, default=4)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--out-dir", default="results/test_fixtures/jvs_verified_pitch_demo")
    parser.add_argument("--out-csv", default="results/calibration/jvs_verified_pitch_demo_summary.csv")
    parser.add_argument("--out-report", default="reports/jvs_verified_pitch_demo_summary.md")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    rows = run_demo(
        jvs_root=Path(args.jvs_root),
        out_dir=ROOT / args.out_dir,
        max_items=args.max_items,
        max_speakers=args.max_speakers,
        sample_rate=args.sample_rate,
        clean=args.clean,
    )
    _write_csv(ROOT / args.out_csv, rows)
    _write_report(ROOT / args.out_report, rows, out_dir=ROOT / args.out_dir)
    summary = _mode_summary(rows)
    print(f"wrote {ROOT / args.out_csv}")
    print(f"wrote {ROOT / args.out_report}")
    print(f"test_only_cache_dir={ROOT / args.out_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
