#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.evaluator import evaluate_utterance
from jp_speech_eval.feedback_renderer import render_user_facing_result


def _jvs_items(jvs_root: Path, *, speakers: int, utterances_per_speaker: int, limit: int | None) -> Iterable[Dict[str, str]]:
    emitted = 0
    for speaker_dir in sorted(jvs_root.glob("jvs*"))[:speakers]:
        transcript_path = speaker_dir / "parallel100" / "transcripts_utf8.txt"
        wav_dir = speaker_dir / "parallel100" / "wav24kHz16bit"
        if not transcript_path.exists() or not wav_dir.exists():
            continue
        count = 0
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            utt_id, text = line.split(":", 1)
            wav = wav_dir / f"{utt_id}.wav"
            if not wav.exists():
                continue
            yield {
                "speaker_id": speaker_dir.name,
                "utterance_id": utt_id,
                "audio_path": str(wav),
                "target_text": text.strip(),
            }
            emitted += 1
            count += 1
            if limit is not None and emitted >= limit:
                return
            if count >= utterances_per_speaker:
                break


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _pct(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 4)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return round(float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac), 4)


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(float(statistics.fmean(values)), 4),
        "min": round(float(min(values)), 4),
        "p10": _pct(values, 0.10),
        "p50": _pct(values, 0.50),
        "p90": _pct(values, 0.90),
        "max": round(float(max(values)), 4),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _component(row: Dict[str, Any], key: str) -> float | None:
    return _num(row.get(f"component_{key}"))


def _component_loss(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    weight_keys = ("contour", "transition", "final", "hl")
    out = []
    for key in weight_keys:
        values = [_component(row, key) for row in rows]
        values = [v for v in values if v is not None]
        weights = [_num(row.get(f"weight_{key}")) for row in rows]
        weights = [v for v in weights if v is not None]
        if not values:
            continue
        mean_value = float(statistics.fmean(values))
        mean_weight = float(statistics.fmean(weights)) if weights else 0.0
        out.append({
            "component": key,
            "mean_component": round(mean_value, 4),
            "mean_weight": round(mean_weight, 4),
            "approx_score_loss": round(100.0 * mean_weight * (1.0 - mean_value), 2),
        })
    return sorted(out, key=lambda row: float(row["approx_score_loss"]), reverse=True)


def _diagnose_item(item: Dict[str, str], *, sample_rate: int, alignment_mode: str, config: str | None) -> Dict[str, Any]:
    result = evaluate_utterance(
        item["target_text"],
        item["audio_path"],
        alignment_mode=alignment_mode,
        sample_rate=sample_rate,
        scoring_config_path=config,
        use_content_match=False,
    )
    data = result.to_dict()
    user_facing = render_user_facing_result(data)
    details = data.get("details") or {}
    reliability = details.get("reliability") or {}
    prosody = details.get("prosody") or {}
    components = prosody.get("prosody_score_components") or {}
    weights = components.get("weights") or {}
    content = details.get("content_match") or {}
    alignment = details.get("alignment") or {}
    moras = data.get("moras") or []
    valid_count = int(prosody.get("valid_mora_count") or 0)
    mora_count = int(prosody.get("mora_count") or len(moras))
    gate = ((user_facing.get("debug") or {}).get("reliability_gate") or {})
    hidden_reasons = user_facing.get("suppressed_reasons") or gate.get("reasons") or []
    alignment_mode_out = str(data.get("alignment_mode") or alignment.get("mode") or "")
    pitch_source = str(prosody.get("pitch_target_source") or "")
    hl_source = str(prosody.get("hl_target_source") or "")
    note = str(prosody.get("note") or "")
    unavailable_reason = note or (";".join(hidden_reasons) if hidden_reasons else "")

    return {
        "sample_id": f"{item['speaker_id']}:{item['utterance_id']}",
        "speaker_id": item["speaker_id"],
        "utterance_id": item["utterance_id"],
        "target_text": item["target_text"],
        "target_kana": data.get("kana"),
        "target_mora_count": len(moras),
        "audio_path": item["audio_path"],
        "prosody_score": data.get("prosody_score"),
        "raw_prosody_score": data.get("prosody_score"),
        "visible_prosody_score": (user_facing.get("debug") or {}).get("visible_prosody_score"),
        "display_score": user_facing.get("display_score"),
        "f0_coverage": reliability.get("f0_coverage"),
        "pitch_target_source": pitch_source,
        "hl_target_source": hl_source,
        "pitch_target_consistency": prosody.get("pitch_target_consistency"),
        "alignment_mode": alignment_mode_out,
        "alignment_fallback": "fallback" in alignment_mode_out,
        "content_match_status": content.get("status"),
        "content_match_ok": str(content.get("status") or "unknown") in {"pass", "unknown"},
        "contour_corr": prosody.get("contour_corr"),
        "contour_rmse": prosody.get("contour_rmse"),
        "transition_agreement": prosody.get("transition_agreement"),
        "hl_match": prosody.get("hl_match_rate", prosody.get("hl_match")),
        "accent_drop_match": prosody.get("accent_drop_agreement"),
        "final_intonation_match": prosody.get("final_intonation_match"),
        "final_intonation_score": prosody.get("final_score"),
        "voiced_mora_count": valid_count,
        "valid_f0_mora_count": valid_count,
        "mora_count": mora_count,
        "unavailable_reason": unavailable_reason,
        "gate_reason": ";".join(str(item) for item in hidden_reasons),
        "component_contour": components.get("contour"),
        "component_transition": components.get("transition"),
        "component_final": components.get("final"),
        "component_hl": components.get("hl"),
        "component_dynamics": components.get("dynamics"),
        "weight_contour": weights.get("contour"),
        "weight_transition": weights.get("transition"),
        "weight_final": weights.get("final"),
        "weight_hl": weights.get("hl"),
        "feedback": " | ".join(str(item) for item in data.get("feedback") or []),
    }


def _write_report(path: Path, rows: Sequence[Dict[str, Any]], *, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric_fields = [
        "prosody_score",
        "f0_coverage",
        "contour_corr",
        "transition_agreement",
        "hl_match",
        "accent_drop_match",
        "final_intonation_score",
        "component_contour",
        "component_transition",
        "component_final",
        "component_hl",
    ]
    summaries = {
        key: _summary([v for v in (_num(row.get(key)) for row in rows) if v is not None])
        for key in numeric_fields
    }
    losses = _component_loss(rows)
    fallback_count = sum(1 for row in rows if str(row.get("alignment_fallback")) == "True" or row.get("alignment_fallback") is True)
    low_f0_count = sum(1 for row in rows if (_num(row.get("f0_coverage")) or 0.0) < 0.50)
    pitch_sources = Counter(str(row.get("pitch_target_source") or "unknown") for row in rows)
    hl_sources = Counter(str(row.get("hl_target_source") or "unknown") for row in rows)
    low_rows = sorted(rows, key=lambda row: _num(row.get("prosody_score")) if _num(row.get("prosody_score")) is not None else 999)[:5]

    likely_causes = []
    if rows and fallback_count == 0:
        likely_causes.append("alignment fallback is unlikely to explain the low native prosody scores in this run")
    if rows and low_f0_count == 0:
        likely_causes.append("F0 coverage is generally sufficient, so missing F0 is not the primary explanation")
    if any(source in {"auto_pyopenjtalk", "heuristic", "unknown", ""} or source.startswith("openjtalk") for source in pitch_sources):
        likely_causes.append("pitch target/reference quality is a likely factor because targets are tool-generated or heuristic")
    if summaries.get("component_contour", {}).get("mean", 1.0) < 0.75:
        likely_causes.append("contour similarity is a major score limiter")
    if summaries.get("component_transition", {}).get("mean", 1.0) < 0.75:
        likely_causes.append("mora-to-mora transition agreement is a major score limiter")
    if summaries.get("prosody_score", {}).get("mean", 100.0) < 80.0:
        likely_causes.append("raw score scale/calibration is not ready for user-facing pitch scoring")

    lines = [
        "# Prosody component diagnostics: JVS native",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- jvs_root: `{args.jvs_root}`",
        f"- samples: {len(rows)}",
        f"- speakers: {args.jvs_speakers}",
        f"- utterances_per_speaker: {args.jvs_utterances_per_speaker}",
        f"- alignment_mode: `{args.alignment_mode}`",
        "",
        "## Score and Component Summary",
        "",
        "| field | n | mean | min | p10 | p50 | p90 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in numeric_fields:
        s = summaries[key]
        lines.append(
            f"| {key} | {s.get('n', 0)} | {s.get('mean', '')} | {s.get('min', '')} | {s.get('p10', '')} | {s.get('p50', '')} | {s.get('p90', '')} | {s.get('max', '')} |"
        )
    lines.extend([
        "",
        "## Approximate Component Loss",
        "",
        "| component | mean_component | mean_weight | approx_score_loss |",
        "|---|---:|---:|---:|",
    ])
    for row in losses:
        lines.append(
            f"| {row['component']} | {row['mean_component']} | {row['mean_weight']} | {row['approx_score_loss']} |"
        )
    lines.extend([
        "",
        "## Evidence and Gate Summary",
        "",
        f"- alignment fallback count: {fallback_count}/{len(rows)}",
        f"- low F0 coverage count (<0.50): {low_f0_count}/{len(rows)}",
        f"- pitch_target_source counts: `{dict(pitch_sources)}`",
        f"- hl_target_source counts: `{dict(hl_sources)}`",
        "",
        "## Lowest Prosody Samples",
        "",
        "| sample_id | prosody_score | f0_coverage | contour_corr | transition_agreement | final_intonation_score | pitch_target_source | gate_reason |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in low_rows:
        lines.append(
            f"| {row.get('sample_id')} | {row.get('prosody_score')} | {row.get('f0_coverage')} | {row.get('contour_corr')} | {row.get('transition_agreement')} | {row.get('final_intonation_score')} | {row.get('pitch_target_source')} | {row.get('gate_reason')} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    if likely_causes:
        for item in likely_causes:
            lines.append(f"- {item}.")
    else:
        lines.append("- No dominant cause was identified from this diagnostic run.")
    lines.extend([
        "- This report is diagnostic only; it does not change runtime scoring.",
        "- Native high-score calibration is not justified yet without flat/random/wrong-accent negative controls.",
        "",
        "## Calibration Readiness",
        "",
        "Not ready. This diagnostic identifies component behavior on native audio, but calibration should wait until counterfactual or real negative controls prove that native-correct contours rank above flat/random/wrong-accent contours.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rows = [
        _diagnose_item(item, sample_rate=args.sr, alignment_mode=args.alignment_mode, config=args.config)
        for item in _jvs_items(
            Path(args.jvs_root),
            speakers=args.jvs_speakers,
            utterances_per_speaker=args.jvs_utterances_per_speaker,
            limit=args.jvs_limit,
        )
    ]
    csv_path = Path(args.output_csv)
    report_path = Path(args.report)
    _write_csv(csv_path, rows)
    _write_report(report_path, rows, args=args)
    return {
        "rows": len(rows),
        "output_csv": str(csv_path),
        "report": str(report_path),
        "prosody_summary": _summary([v for v in (_num(row.get("prosody_score")) for row in rows) if v is not None]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose pitch/prosody components on a small JVS native sample.")
    parser.add_argument("--jvs-root", default=str(PROJECT_ROOT / "JVS"))
    parser.add_argument("--jvs-speakers", type=int, default=1)
    parser.add_argument("--jvs-utterances-per-speaker", type=int, default=17)
    parser.add_argument("--jvs-limit", type=int, default=17)
    parser.add_argument("--alignment-mode", default="equal", choices=["equal", "dtw", "cached_dtw"])
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-csv", default=str(ROOT / "results" / "calibration" / "prosody_component_diagnostics_jvs_native.csv"))
    parser.add_argument("--report", default=str(ROOT / "reports" / "prosody_component_diagnostics_jvs_native.md"))
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
