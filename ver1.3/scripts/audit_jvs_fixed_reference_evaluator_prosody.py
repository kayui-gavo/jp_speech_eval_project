#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from jp_speech_eval.evaluator import evaluate_utterance  # noqa: E402
from jp_speech_eval.feedback_renderer import render_user_facing_result  # noqa: E402
from jp_speech_eval.text_frontend import build_text_info  # noqa: E402

from audit_cross_speaker_prosody_reference import (  # noqa: E402
    _lab_f0,
    _score_row,
    flat_f0,
    low_f0_coverage,
    pattern_from_f0,
    shuffled_f0,
    wrong_drop_f0,
)
from build_test_jvs_prosody_reference_cache import build_test_jvs_cache, jvs_item  # noqa: E402
from jp_speech_eval.prosody_reference_cache import smooth_f0_by_mora  # noqa: E402


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
    case_name: str,
    result: Mapping[str, Any],
    mode: str,
    reference_speaker: str,
    user_speaker: str,
    notes: str,
) -> Dict[str, Any]:
    details = result.get("details") or {}
    reliability = details.get("reliability") or {}
    content = details.get("content_match") or {}
    prosody = details.get("prosody") or {}
    row = {
        "case_name": case_name,
        "eval_path": "evaluator",
        "mode": mode,
        "target_text": result.get("target_text"),
        "reference_speaker": reference_speaker,
        "user_speaker": user_speaker,
        "prosody_score": result.get("prosody_score"),
        "pronunciation_score": result.get("pronunciation_score"),
        "fluency_score": result.get("fluency_score"),
        "total_score": result.get("total_score"),
        "pitch_target_source": details.get("pitch_target_source") or prosody.get("pitch_target_source"),
        "pitch_target_reliability": details.get("pitch_target_reliability") or prosody.get("pitch_target_reliability"),
        "content_match_status": content.get("status"),
        "content_verified": content.get("content_verified"),
        "content_match_note": content.get("note"),
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


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "p50": round(ordered[len(ordered) // 2], 4),
        "max": round(max(values), 4),
    }


def run_audit(
    *,
    jvs_root: Path,
    reference_speaker: str,
    user_speaker: str,
    utterance_id: str,
    cache_prefix: Path,
    sample_rate: int,
) -> List[Dict[str, Any]]:
    cache = build_test_jvs_cache(
        jvs_root=jvs_root,
        speaker_id=reference_speaker,
        utterance_id=utterance_id,
        out_prefix=cache_prefix,
        sample_rate=sample_rate,
        write_sidecar=True,
    )
    user = jvs_item(jvs_root, user_speaker, utterance_id)
    reference = jvs_item(jvs_root, reference_speaker, utterance_id)
    rows: List[Dict[str, Any]] = []

    fixed_result = evaluate_utterance(
        wav_path=user["audio_path"],
        alignment_mode="cached_dtw",
        cache_path=cache.prefix,
        use_content_match=True,
    ).to_dict()
    rows.append(_evaluator_row(
        case_name="verified_jvs_sidecar_fixed_reference",
        result=fixed_result,
        mode="fixed_reference",
        reference_speaker=reference_speaker,
        user_speaker=user_speaker,
        notes="test-only JVS cache with verified reference_audio_f0_cache sidecar",
    ))

    text_result = evaluate_utterance(
        text=reference["target_text"],
        wav_path=user["audio_path"],
        alignment_mode="dtw",
        use_content_match=False,
    ).to_dict()
    rows.append(_evaluator_row(
        case_name="same_audio_text_openjtalk_evaluator_baseline",
        result=text_result,
        mode="text_openjtalk_baseline",
        reference_speaker=reference_speaker,
        user_speaker=user_speaker,
        notes="evaluator baseline without fixed reference cache; uses text/OpenJTalk target and no content gate",
    ))

    text_info = build_text_info(reference["target_text"])
    ref_f0, _ref_meta = _lab_f0(reference, text_info.moras, sample_rate=sample_rate)
    user_f0, _user_meta = _lab_f0(user, text_info.moras, sample_rate=sample_rate)
    if ref_f0 and user_f0:
        ref_smooth = smooth_f0_by_mora(ref_f0)
        ref_pattern = pattern_from_f0(ref_smooth)
        pair = {
            "pair_id": f"{reference_speaker}->{user_speaker}:{utterance_id}",
            "reference": reference,
            "user": user,
        }
        semi_audio_cases = [
            ("native_cross_speaker_reference_audio_f0_cache", user_f0, "speaker B native F0 against speaker A verified reference F0"),
            ("flat_pitch_correct_content", flat_f0(user_f0), "same content/timing, flattened F0"),
            ("shuffled_random_pitch_correct_content", shuffled_f0(user_f0, seed=sum(ord(ch) for ch in utterance_id)), "same content/timing, shuffled F0"),
            ("wrong_accent_drop_correct_content", wrong_drop_f0(user_f0, ref_smooth, text_info.accent_phrases), "same content/timing, accent-drop transition contradicted when available"),
            ("low_f0_coverage_correct_content", low_f0_coverage(user_f0), "same content/timing, insufficient F0 coverage"),
        ]
        for case_name, f0_values, notes in semi_audio_cases:
            row = _score_row(
                pair=pair,
                text_info=text_info,
                mode=case_name,
                user_f0=f0_values,
                target_pattern=ref_pattern,
                pitch_target_source="reference_audio_f0_cache",
                reference_f0=ref_smooth,
                notes=notes,
            )
            row.update({
                "case_name": case_name,
                "eval_path": "semi_audio_lab_f0",
                "user_facing_status": "",
                "display_score": "",
                "visible_prosody_score": "",
                "prosody_score_visible": "",
                "tone_score_in_core_four": "no",
            })
            rows.append(row)
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


def _write_report(path: Path, rows: Sequence[Mapping[str, Any]], *, cache_prefix: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_mode: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        try:
            value = float(row.get("prosody_score"))
        except (TypeError, ValueError):
            continue
        by_mode[str(row.get("case_name") or row.get("mode"))].append(value)
    sidecar = next((row for row in rows if row.get("case_name") == "verified_jvs_sidecar_fixed_reference"), {})
    openjtalk = next((row for row in rows if row.get("case_name") == "same_audio_text_openjtalk_evaluator_baseline"), {})
    lines = [
        "# JVS fixed-reference evaluator prosody sanity",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- test_cache_prefix: `{cache_prefix}`",
        "- scope: test-only verified native reference path; no packaged demo target is promoted.",
        "",
        "## Evaluator Path Checks",
        "",
        "| case | prosody | pitch_target_source | reliability | content_status | alignment | display_score | visible_prosody |",
        "|---|---:|---|---|---|---|---:|---:|",
    ]
    for row in [sidecar, openjtalk]:
        lines.append(
            f"| {row.get('case_name')} | {row.get('prosody_score')} | {row.get('pitch_target_source')} | "
            f"{row.get('pitch_target_reliability')} | {row.get('content_match_status')} | {row.get('alignment_mode')} | "
            f"{row.get('display_score')} | {row.get('visible_prosody_score')} |"
        )
    lines.extend([
        "",
        "## Prosody Score Summary",
        "",
        "| case | n | mean | min | p50 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for mode, values in sorted(by_mode.items()):
        summary = _summary(values)
        lines.append(
            f"| {mode} | {summary.get('n', 0)} | {summary.get('mean', '')} | {summary.get('min', '')} | "
            f"{summary.get('p50', '')} | {summary.get('max', '')} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if sidecar:
        if sidecar.get("pitch_target_source") == "reference_audio_f0_cache" and sidecar.get("pitch_target_reliability") == "reliable":
            lines.append("- Evaluator successfully reads the test-only verified JVS sidecar as `reference_audio_f0_cache`.")
        else:
            lines.append("- Evaluator did not use the verified sidecar as expected; inspect cache selection.")
        if sidecar.get("content_match_status") == "pass":
            lines.append("- Content gate passes for this cross-speaker JVS fixed-reference sample.")
        else:
            lines.append("- Content gate did not pass; do not treat this as a product-ready score.")
    lines.extend([
        "- The OpenJTalk row is an evaluator-level text baseline, not a fixed-reference cache path.",
        "- Flat/random/low-F0 rows are semi-audio lab-F0 controls, not waveform-manipulated audio.",
        "- This audit does not enable calibration or change runtime scoring.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluator-level sanity for a test-only JVS verified reference sidecar.")
    parser.add_argument("--jvs-root", default=str(ROOT.parent / "JVS"))
    parser.add_argument("--reference-speaker", default="jvs001")
    parser.add_argument("--user-speaker", default="jvs002")
    parser.add_argument("--utterance-id", default="VOICEACTRESS100_001")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--cache-prefix", default="outputs/test_jvs_prosody_reference_cache/jvs001_VOICEACTRESS100_001")
    parser.add_argument("--out-csv", default="results/calibration/jvs_fixed_reference_evaluator_prosody_sanity.csv")
    parser.add_argument("--out-report", default="reports/jvs_fixed_reference_evaluator_prosody_sanity.md")
    parser.add_argument("--clean-cache", action="store_true")
    args = parser.parse_args()

    cache_prefix = ROOT / args.cache_prefix
    if args.clean_cache and cache_prefix.parent.exists():
        shutil.rmtree(cache_prefix.parent)
    rows = run_audit(
        jvs_root=Path(args.jvs_root),
        reference_speaker=args.reference_speaker,
        user_speaker=args.user_speaker,
        utterance_id=args.utterance_id,
        cache_prefix=cache_prefix,
        sample_rate=args.sample_rate,
    )
    _write_csv(ROOT / args.out_csv, rows)
    _write_report(ROOT / args.out_report, rows, cache_prefix=cache_prefix)
    print(f"wrote {ROOT / args.out_csv}")
    print(f"wrote {ROOT / args.out_report}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
