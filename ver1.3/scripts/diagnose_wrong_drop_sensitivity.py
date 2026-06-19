#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
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

from jp_speech_eval.prosody_reference_cache import smooth_f0_by_mora  # noqa: E402
from jp_speech_eval.scoring import score_prosody  # noqa: E402
from jp_speech_eval.text_frontend import build_text_info  # noqa: E402

from audit_cross_speaker_prosody_reference import (  # noqa: E402
    _lab_f0,
    _select_pairs,
    pattern_from_f0,
    wrong_drop_f0,
)


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "min": round(min(values), 4),
        "p50": round(ordered[len(ordered) // 2], 4),
        "max": round(max(values), 4),
    }


def _score_details(text_info: Any, user_f0: Sequence[float], ref_f0: Sequence[float]) -> tuple[int, Dict[str, Any]]:
    score, _feedback, details = score_prosody(
        moras=text_info.moras,
        target_pattern=pattern_from_f0(ref_f0),
        f0_by_mora=list(user_f0),
        reference_f0_by_mora=list(ref_f0),
        pitch_target_source="reference_audio_f0_cache",
        reference_f0_target_source="reference_audio_f0_cache",
        is_question=bool(text_info.is_question),
        accent_phrases=text_info.accent_phrases,
    )
    return score, details


def diagnose(
    jvs_root: Path,
    *,
    max_speakers: int | None,
    max_pairs: int,
    sample_rate: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pair in _select_pairs(jvs_root, max_speakers=max_speakers, max_pairs=max_pairs):
        text_info = build_text_info(pair["reference"]["target_text"])
        ref_f0, _ref_meta = _lab_f0(pair["reference"], text_info.moras, sample_rate=sample_rate)
        user_f0, _user_meta = _lab_f0(pair["user"], text_info.moras, sample_rate=sample_rate)
        if ref_f0 is None or user_f0 is None:
            continue
        ref_smooth = smooth_f0_by_mora(ref_f0)
        wrong_f0 = wrong_drop_f0(user_f0, ref_smooth, text_info.accent_phrases)
        native_score, native_details = _score_details(text_info, user_f0, ref_smooth)
        wrong_score, wrong_details = _score_details(text_info, wrong_f0, ref_smooth)
        native_components = native_details.get("prosody_score_components") or {}
        wrong_components = wrong_details.get("prosody_score_components") or {}
        weights = native_components.get("weights") or {}
        drop_indices = [
            idx
            for idx, role in enumerate(native_details.get("transition_roles") or [])
            if role == "accent_nucleus_drop"
        ]
        changed_indices = [
            idx
            for idx, (a, b) in enumerate(zip(user_f0, wrong_f0))
            if (math.isfinite(float(a)) or math.isfinite(float(b))) and abs(float(a) - float(b)) > 1e-6
        ]
        contour_weight = float(weights.get("contour", 0.0) or 0.0)
        transition_weight = float(weights.get("transition", 0.0) or 0.0)
        native_contour = _num(native_components.get("contour")) or 0.0
        wrong_contour = _num(wrong_components.get("contour")) or 0.0
        native_transition = _num(native_components.get("transition")) or 0.0
        wrong_transition = _num(wrong_components.get("transition")) or 0.0
        rows.append({
            "pair_id": pair["pair_id"],
            "utterance_id": pair["reference"]["utterance_id"],
            "target_text": pair["reference"]["target_text"],
            "mora_count": len(text_info.moras),
            "accent_phrase_count": len(text_info.accent_phrases or []),
            "accent_drop_target_count": len(drop_indices),
            "drop_indices": " ".join(str(i) for i in drop_indices),
            "changed_indices": " ".join(str(i) for i in changed_indices),
            "changed_drop_overlap_count": len(set(drop_indices) & set(changed_indices)),
            "native_score": native_score,
            "wrong_drop_score": wrong_score,
            "score_delta_native_minus_wrong": native_score - wrong_score,
            "native_accent_drop_match": native_details.get("accent_drop_agreement"),
            "wrong_accent_drop_match": wrong_details.get("accent_drop_agreement"),
            "native_transition": native_transition,
            "wrong_transition": wrong_transition,
            "transition_delta": native_transition - wrong_transition,
            "transition_weight": transition_weight,
            "approx_transition_score_loss": round(100.0 * transition_weight * (native_transition - wrong_transition), 4),
            "native_contour": native_contour,
            "wrong_contour": wrong_contour,
            "contour_delta": native_contour - wrong_contour,
            "contour_weight": contour_weight,
            "approx_contour_score_loss": round(100.0 * contour_weight * (native_contour - wrong_contour), 4),
            "native_hl": native_components.get("hl"),
            "wrong_hl": wrong_components.get("hl"),
            "final_delta": (_num(native_components.get("final")) or 0.0) - (_num(wrong_components.get("final")) or 0.0),
            "diagnosis": "drop_changed_but_no_explicit_drop_component" if drop_indices else "no_explicit_accent_drop_target",
        })
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


def _write_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values_by_field: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for key in (
            "score_delta_native_minus_wrong",
            "accent_drop_target_count",
            "changed_drop_overlap_count",
            "transition_delta",
            "approx_transition_score_loss",
            "contour_delta",
            "approx_contour_score_loss",
        ):
            value = _num(row.get(key))
            if value is not None:
                values_by_field[key].append(value)
    zero_or_negative = [
        row for row in rows
        if (_num(row.get("score_delta_native_minus_wrong")) or 0.0) <= 0.0
    ]
    lines = [
        "# Prosody wrong-drop sensitivity diagnostics",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- pairs: {len(rows)}",
        "- scope: diagnostic only; no prosody weights or scoring formulas are changed.",
        "",
        "## Summary",
        "",
        "| field | n | mean | min | p50 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for field, values in values_by_field.items():
        summary = _summary(values)
        lines.append(
            f"| {field} | {summary.get('n', 0)} | {summary.get('mean', '')} | {summary.get('min', '')} | "
            f"{summary.get('p50', '')} | {summary.get('max', '')} |"
        )
    lines.extend([
        "",
        "## Weak-Separation Cases",
        "",
        f"- wrong_drop_score_not_below_native_count: {len(zero_or_negative)}/{len(rows)}",
        "",
        "## Interpretation",
        "",
        "- The wrong-drop counterfactual usually changes the intended drop mora, but the total score only changes modestly.",
        "- Current scoring has contour, transition, final, and H/L components; accent-drop agreement is logged and used for feedback, but it is not an independent weighted score component.",
        "- Wrong-drop therefore only affects the aggregate indirectly through transition direction and contour similarity.",
        "- In many sentences, changing one accent-drop transition leaves the overall contour correlation high, so the score remains close to native.",
        "- OpenJTalk-derived accent phrase/drop targets are still heuristic; drop-specific feedback should require reliable targets.",
        "",
        "## Candidate Fixes For Later",
        "",
        "- Add a separate accent-drop subscore after target confidence is reliable.",
        "- Detect phrase-level drop windows instead of only adjacent mora transitions.",
        "- Keep accent-drop feedback unavailable when target confidence is weak.",
        "- Validate with real or waveform-level wrong-accent audio before enabling calibration.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose weak wrong-accent-drop separation.")
    parser.add_argument("--jvs-root", default=str(ROOT.parent / "JVS"))
    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--out-csv", default="results/calibration/prosody_wrong_drop_sensitivity_diagnostics.csv")
    parser.add_argument("--out-report", default="reports/prosody_wrong_drop_sensitivity_diagnostics.md")
    args = parser.parse_args()

    rows = diagnose(
        Path(args.jvs_root),
        max_speakers=args.max_speakers,
        max_pairs=args.max_pairs,
        sample_rate=args.sample_rate,
    )
    _write_csv(ROOT / args.out_csv, rows)
    _write_report(ROOT / args.out_report, rows)
    print(f"wrote {ROOT / args.out_csv}")
    print(f"wrote {ROOT / args.out_report}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
