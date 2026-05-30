#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.evaluator import evaluate_utterance
from jp_speech_eval.feedback_renderer import render_user_facing_result
from jp_speech_eval.phonology import classify_mora_sequence


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def _mean(values: Sequence[float]) -> float | None:
    return round(float(statistics.fmean(values)), 4) if values else None


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


def _jvs_items(jvs_root: Path, *, speakers: int, utterances_per_speaker: int, limit: int | None = None) -> Iterable[Dict[str, str]]:
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
                "dataset": "jvs",
                "split": "native",
                "speaker_id": speaker_dir.name,
                "utterance_id": utt_id,
                "audio_path": str(wav),
                "text": text.strip(),
                "stimulus_type": "sentence",
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return
            count += 1
            if count >= utterances_per_speaker:
                break


def _janon_items(janon_root: Path, *, limit: int) -> Iterable[Dict[str, str]]:
    data_path = janon_root / "data.csv"
    if not data_path.exists():
        return
    emitted = 0
    with data_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = row.get("Path") or row.get("path") or ""
            text = row.get("Stmiulus") or row.get("Stimulus") or ""
            wav = janon_root / rel_path
            if not text or not wav.exists():
                continue
            yield {
                "dataset": "janon",
                "split": "l2",
                "speaker_id": row.get("Speaker") or "",
                "native_language": row.get("Native Language") or "",
                "utterance_id": wav.stem,
                "audio_path": str(wav),
                "text": text.strip(),
                "stimulus_type": row.get("Stimulus Type") or "",
            }
            emitted += 1
            if emitted >= limit:
                break


def _evaluate_item(item: Dict[str, str], *, sample_rate: int, config: str | None) -> Dict[str, Any]:
    result = evaluate_utterance(
        item["text"],
        Path(item["audio_path"]),
        sample_rate=sample_rate,
        alignment_mode="equal",
        scoring_config_path=config,
        use_content_match=False,
    )
    data = result.to_dict()
    user_facing = render_user_facing_result(data)
    details = data.get("details", {})
    reliability = details.get("reliability", {})
    fluency = details.get("fluency", {})
    pronunciation = details.get("pronunciation", {})
    evidence = details.get("mora_evidence_summary", {})
    recording = details.get("recording_quality", {})
    special_count = int(evidence.get("strong_special_mora_count", evidence.get("special_mora_count", 0)) or 0)
    special_judged = int(evidence.get("strong_special_mora_judgement_available_count", evidence.get("special_mora_judgement_available_count", 0)) or 0)
    moras = data.get("moras") or []
    mora_count = len(moras)
    length_bucket = "short" if mora_count <= 5 else "medium" if mora_count <= 18 else "long"
    mora_text = "".join(str(mora) for mora in moras)
    phonology = classify_mora_sequence([str(mora) for mora in moras])
    has_long_vowel = any(row.mora_type == "explicit_long_vowel" for row in phonology)
    has_sokuon = any(row.mora_type == "sokuon" for row in phonology)
    has_moraic_nasal = any(row.mora_type == "nasal" for row in phonology)
    has_yoon = any(ch in mora_text for ch in "ゃゅょャュョ")
    rhythm_timing_score = fluency.get("rhythm_timing_score", data.get("fluency_score", 0))
    delivery_fluency_score = fluency.get("delivery_fluency_score", data.get("fluency_score", 0))
    return {
        **item,
        "mora_count": mora_count,
        "moras": " ".join(str(mora) for mora in moras),
        "length_bucket": length_bucket,
        "has_long_vowel": has_long_vowel,
        "has_sokuon": has_sokuon,
        "has_moraic_nasal": has_moraic_nasal,
        "has_yoon": has_yoon,
        "alignment_success": not str(data.get("alignment_mode", "")).endswith("fallback_equal"),
        "alignment_method": data.get("alignment_mode"),
        "alignment_confidence": _float(reliability.get("alignment")),
        "fallback_used": str(data.get("alignment_mode", "")).endswith("fallback_equal"),
        "recording_quality": _float(recording.get("score")),
        "f0_coverage": _float(reliability.get("f0_coverage")),
        "speech_rate_mora_per_sec": _float(fluency.get("speech_rate_mora_per_sec")),
        "avg_mora_duration_sec": _float(fluency.get("avg_mora_duration_sec")),
        "mora_duration_cv": _float(pronunciation.get("mora_duration_cv")),
        "special_mora_count": special_count,
        "special_mora_judgement_available_count": special_judged,
        "special_mora_penalty": _float(pronunciation.get("special_mora_penalty")),
        "pause_ratio": _float((data.get("pause_info") or {}).get("pause_ratio")),
        "score_total_raw": int(data.get("total_score", 0) or 0),
        "score_display": user_facing.get("display_score"),
        "score_pronunciation": int(data.get("pronunciation_score", 0) or 0),
        "score_prosody": int(data.get("prosody_score", 0) or 0),
        "score_fluency": int(data.get("fluency_score", 0) or 0),
        "rhythm_timing_score": _float(rhythm_timing_score),
        "delivery_fluency_score": _float(delivery_fluency_score),
        "practice_check_result": user_facing.get("practice_check_result"),
        "user_reliability": user_facing.get("reliability"),
        "feedback_suppressed_by_gate": " | ".join((user_facing.get("debug", {}).get("reliability_gate", {}) or {}).get("blocked_categories", [])),
        "failure_reason": " | ".join((user_facing.get("debug", {}).get("reliability_gate", {}) or {}).get("reasons", [])),
        "user_feedback": " | ".join(user_facing.get("user_messages") or []),
    }


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    display_scores = [_float(r.get("score_display")) for r in rows if r.get("score_display") not in {None, ""}]
    pronunciation = [_float(r.get("score_pronunciation")) for r in rows]
    fluency = [_float(r.get("score_fluency")) for r in rows]
    rhythm = [_float(r.get("rhythm_timing_score")) for r in rows]
    delivery = [_float(r.get("delivery_fluency_score")) for r in rows]
    retry = [r for r in rows if r.get("practice_check_result") == "retry"]
    unscorable = [r for r in rows if r.get("practice_check_result") in {"retry", "unscorable"}]
    special_rows = [r for r in rows if int(r.get("special_mora_count") or 0) > 0]
    special_false_alarm = [
        r for r in special_rows
        if _float(r.get("special_mora_penalty")) > 0.0 or _float(r.get("score_pronunciation")) < 90.0
    ]
    return {
        "n": len(rows),
        "display_mean": _mean(display_scores),
        "display_p10": _pct(display_scores, 0.10),
        "display_p50": _pct(display_scores, 0.50),
        "pronunciation_mean": _mean(pronunciation),
        "fluency_mean": _mean(fluency),
        "rhythm_timing_mean": _mean(rhythm),
        "delivery_fluency_mean": _mean(delivery),
        "retry_rate": round(len(retry) / len(rows), 4),
        "unscorable_rate": round(len(unscorable) / len(rows), 4),
        "alignment_fallback_rate": round(sum(1 for r in rows if r.get("fallback_used")) / len(rows), 4),
        "special_mora_rows": len(special_rows),
        "special_mora_false_alarm_rate_proxy": round(len(special_false_alarm) / len(special_rows), 4) if special_rows else None,
    }


def _feature_distribution(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fields = [
        "score_display",
        "score_pronunciation",
        "score_prosody",
        "score_fluency",
        "rhythm_timing_score",
        "delivery_fluency_score",
        "alignment_confidence",
        "f0_coverage",
        "speech_rate_mora_per_sec",
        "avg_mora_duration_sec",
        "mora_duration_cv",
        "special_mora_penalty",
        "pause_ratio",
    ]
    out: List[Dict[str, Any]] = []
    for field in fields:
        values = [_float(r.get(field)) for r in rows if r.get(field) not in {None, ""}]
        out.append({
            "feature": field,
            "n": len(values),
            "mean": _mean(values),
            "p05": _pct(values, 0.05),
            "p10": _pct(values, 0.10),
            "p50": _pct(values, 0.50),
            "p90": _pct(values, 0.90),
            "p95": _pct(values, 0.95),
        })
    return out


def _false_alarm_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    checks = [
        ("display_score_below_85", lambda r: r.get("score_display") not in {None, ""} and _float(r.get("score_display")) < 85),
        ("pronunciation_below_90", lambda r: _float(r.get("score_pronunciation")) < 90),
        ("retry_or_unscorable", lambda r: r.get("practice_check_result") in {"retry", "unscorable"}),
        ("alignment_fallback", lambda r: bool(r.get("fallback_used"))),
        ("special_mora_penalty_nonzero", lambda r: int(r.get("special_mora_count") or 0) > 0 and _float(r.get("special_mora_penalty")) > 0),
        ("f0_coverage_below_50pct", lambda r: _float(r.get("f0_coverage")) < 0.50),
    ]
    out: List[Dict[str, Any]] = []
    for name, predicate in checks:
        flagged = [r for r in rows if predicate(r)]
        out.append({
            "feature": name,
            "n_flagged": len(flagged),
            "rate": round(len(flagged) / len(rows), 4) if rows else 0.0,
            "example_utterances": " | ".join(str(r.get("utterance_id")) for r in flagged[:5]),
        })
    return out


def _percentile_table(rows: Sequence[Dict[str, Any]], *, group_field: str = "length_bucket") -> Dict[str, Any]:
    fields = [
        "speech_rate_mora_per_sec",
        "pause_ratio",
        "mora_duration_cv",
        "avg_mora_duration_sec",
        "f0_coverage",
        "score_fluency",
        "rhythm_timing_score",
        "delivery_fluency_score",
    ]
    group_names = ["all"] + sorted({str(r.get(group_field) or "unknown") for r in rows})
    out: Dict[str, Any] = {
        "note": "Small-sample native percentile snapshot. Use as threshold-audit evidence, not as a final norm.",
        "group_field": group_field,
        "groups": {},
    }
    for group in group_names:
        group_rows = rows if group == "all" else [r for r in rows if str(r.get(group_field) or "unknown") == group]
        stats: Dict[str, Any] = {"n": len(group_rows)}
        for field in fields:
            values = [_float(r.get(field)) for r in group_rows if r.get(field) not in {None, ""}]
            stats[field] = {
                "n": len(values),
                "p01": _pct(values, 0.01),
                "p05": _pct(values, 0.05),
                "p10": _pct(values, 0.10),
                "p50": _pct(values, 0.50),
                "p90": _pct(values, 0.90),
                "p95": _pct(values, 0.95),
                "p99": _pct(values, 0.99),
            }
        out["groups"][group] = stats
    return out


def _coverage_summary(group_name: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"group": group_name, "total_samples": 0}
    return {
        "group": group_name,
        "total_samples": len(rows),
        "speaker_count": len({r.get("speaker_id") for r in rows if r.get("speaker_id")}),
        "utterance_count": len({r.get("utterance_id") for r in rows if r.get("utterance_id")}),
        "samples_with_long_vowel": sum(1 for r in rows if r.get("has_long_vowel")),
        "samples_with_sokuon": sum(1 for r in rows if r.get("has_sokuon")),
        "samples_with_moraic_nasal": sum(1 for r in rows if r.get("has_moraic_nasal")),
        "samples_with_yoon": sum(1 for r in rows if r.get("has_yoon")),
        "short_count": sum(1 for r in rows if r.get("length_bucket") == "short"),
        "medium_count": sum(1 for r in rows if r.get("length_bucket") == "medium"),
        "long_count": sum(1 for r in rows if r.get("length_bucket") == "long"),
        "alignment_success_rate": round(sum(1 for r in rows if r.get("alignment_success")) / len(rows), 4),
        "fallback_rate": round(sum(1 for r in rows if r.get("fallback_used")) / len(rows), 4),
        "f0_coverage_mean": _mean([_float(r.get("f0_coverage")) for r in rows]),
    }


def _feature_coverage(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [_coverage_summary("all", rows)]
    for bucket in sorted({str(r.get("length_bucket") or "unknown") for r in rows}):
        out.append(_coverage_summary(f"length_bucket:{bucket}", [r for r in rows if str(r.get("length_bucket") or "unknown") == bucket]))
    for stimulus_type in sorted({str(r.get("stimulus_type") or "unknown") for r in rows}):
        out.append(_coverage_summary(f"stimulus_type:{stimulus_type}", [r for r in rows if str(r.get("stimulus_type") or "unknown") == stimulus_type]))
    return out


def _write_jvs_report(path: Path, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    verdicts = [
        ("mora_clarity native mean >= 90", _float(summary.get("pronunciation_mean")) >= 90.0),
        ("rhythm timing native mean >= 85", _float(summary.get("rhythm_timing_mean")) >= 85.0),
        ("delivery fluency native mean >= 85", _float(summary.get("delivery_fluency_mean")) >= 85.0),
        ("unscorable rate <= 5%", _float(summary.get("unscorable_rate")) <= 0.05),
        ("special mora false alarm proxy <= 5%", (summary.get("special_mora_false_alarm_rate_proxy") is None) or _float(summary.get("special_mora_false_alarm_rate_proxy")) <= 0.05),
    ]
    ceiling_warning = "- Ceiling warning: many native pronunciation scores are exactly 100, so this proxy may be too blunt." if sum(1 for r in rows if _float(r.get("score_pronunciation")) >= 100) >= max(1, len(rows) // 2) else "- No strong ceiling-effect warning in this sample."
    lines = [
        "# JVS native sanity check",
        "",
        "This report is a native false-alarm audit. It does not prove scoring validity, but it checks whether native speakers are being punished by current engineering thresholds.",
        "",
        "## Summary",
        f"- n: {summary.get('n')}",
        f"- display mean: {summary.get('display_mean')}",
        f"- pronunciation mean: {summary.get('pronunciation_mean')}",
        f"- fluency mean: {summary.get('fluency_mean')}",
        f"- rhythm timing mean: {summary.get('rhythm_timing_mean')}",
        f"- delivery fluency mean: {summary.get('delivery_fluency_mean')}",
        f"- retry rate: {summary.get('retry_rate')}",
        f"- unscorable rate: {summary.get('unscorable_rate')}",
        f"- alignment fallback rate: {summary.get('alignment_fallback_rate')}",
        f"- special mora false alarm proxy: {summary.get('special_mora_false_alarm_rate_proxy')}",
        "",
        "## Acceptance checks",
    ]
    lines.extend(f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in verdicts)
    lines.extend([
        "",
        "## Interpretation",
        "- `score_pronunciation` is still a mora-timing/acoustic proxy, not phoneme correctness.",
        ceiling_warning,
        "- Special mora false-alarm checks are meaningful only if the calibration sample includes enough long vowels, sokuon, moraic nasals, and yoon.",
        "- F0/pitch failures should usually suppress pitch feedback instead of lowering native ability claims.",
        "- If alignment fallback is frequent, detailed mora/special-mora feedback must stay gated.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_janon_report(path: Path, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any], native_dist: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    native_ranges = {r["feature"]: r for r in native_dist}
    def outside_native(field: str) -> str:
        ref = native_ranges.get(field)
        if not ref or ref.get("p05") is None or ref.get("p95") is None:
            return "n/a"
        lo = _float(ref["p05"])
        hi = _float(ref["p95"])
        vals = [_float(r.get(field)) for r in rows if r.get(field) not in {None, ""}]
        if not vals:
            return "n/a"
        rate = sum(1 for v in vals if v < lo or v > hi) / len(vals)
        return f"{rate:.4f}"
    lines = [
        "# JANON L2 trend audit",
        "",
        "JANON is learner speech, not ground truth. This report only identifies trends relative to the current JVS native-range snapshot.",
        "",
        "## Summary",
        f"- n: {summary.get('n')}",
        f"- display mean: {summary.get('display_mean')}",
        f"- pronunciation mean: {summary.get('pronunciation_mean')}",
        f"- fluency mean: {summary.get('fluency_mean')}",
        f"- rhythm timing mean: {summary.get('rhythm_timing_mean')}",
        f"- delivery fluency mean: {summary.get('delivery_fluency_mean')}",
        f"- retry rate: {summary.get('retry_rate')}",
        f"- unscorable rate: {summary.get('unscorable_rate')}",
        f"- alignment fallback rate: {summary.get('alignment_fallback_rate')}",
        "",
        "## Outside current JVS native range",
        f"- speech_rate_mora_per_sec outside native P05-P95: {outside_native('speech_rate_mora_per_sec')}",
        f"- avg_mora_duration_sec outside native P05-P95: {outside_native('avg_mora_duration_sec')}",
        f"- mora_duration_cv outside native P05-P95: {outside_native('mora_duration_cv')}",
        f"- f0_coverage outside native P05-P95: {outside_native('f0_coverage')}",
        f"- pause_ratio outside native P05-P95: {outside_native('pause_ratio')}",
        f"- rhythm_timing_score outside native P05-P95: {outside_native('rhythm_timing_score')}",
        f"- delivery_fluency_score outside native P05-P95: {outside_native('delivery_fluency_score')}",
        "",
        "## Interpretation",
        "- Do not conclude that lower score equals lower Japanese ability without teacher/listener ratings.",
        "- Use this to find which feedback rules are too sensitive or too weak.",
        "- Isolated words and sentences should be analyzed separately in the next iteration.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_coverage_report(path: Path, coverage_rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_row = next((r for r in coverage_rows if r.get("group") == "all"), {})
    lines = [
        "# Calibration coverage",
        "",
        "This report checks whether the native calibration sample covers the features that the product claims to evaluate.",
        "",
        "## Overall",
        f"- total samples: {all_row.get('total_samples')}",
        f"- speakers: {all_row.get('speaker_count')}",
        f"- utterances: {all_row.get('utterance_count')}",
        f"- long vowel samples: {all_row.get('samples_with_long_vowel')}",
        f"- sokuon samples: {all_row.get('samples_with_sokuon')}",
        f"- moraic nasal samples: {all_row.get('samples_with_moraic_nasal')}",
        f"- yoon samples: {all_row.get('samples_with_yoon')}",
        "",
        "## Notes",
        "- If a feature count is small, related user feedback should remain conservative or `uncertain`.",
        "- This is coverage accounting, not a validation study.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_threshold_report(path: Path, percentiles: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_stats = (percentiles.get("groups") or {}).get("all") or {}
    lines = [
        "# Rhythm and fluency threshold update",
        "",
        "Current recommendation: use JVS native percentiles as a guardrail for threshold tuning, but do not automatically overwrite runtime thresholds from this small snapshot.",
        "",
        "## Native guardrails",
    ]
    for field in ["speech_rate_mora_per_sec", "pause_ratio", "mora_duration_cv", "rhythm_timing_score", "delivery_fluency_score"]:
        stats = all_stats.get(field) or {}
        lines.append(f"- {field}: P05={stats.get('p05')}, P50={stats.get('p50')}, P95={stats.get('p95')} (n={stats.get('n')})")
    lines.extend([
        "",
        "## Product rule",
        "- Penalize native-like timing less aggressively.",
        "- Split rhythm timing from delivery fluency so a fast but clear utterance is not forced into the same bucket as a choppy utterance.",
        "- Keep pitch-accent diagnosis out of the display score unless the target is verified and the signal evidence is strong.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    jvs_rows = [
        _evaluate_item(item, sample_rate=args.sr, config=args.config)
        for item in _jvs_items(
            Path(args.jvs_root),
            speakers=args.jvs_speakers,
            utterances_per_speaker=args.jvs_utterances_per_speaker,
            limit=args.jvs_limit,
        )
    ]
    janon_rows = [
        _evaluate_item(item, sample_rate=args.sr, config=args.config)
        for item in _janon_items(Path(args.janon_root), limit=args.janon_limit)
    ]

    jvs_dist = _feature_distribution(jvs_rows)
    jvs_percentiles = _percentile_table(jvs_rows)
    jvs_coverage = _feature_coverage(jvs_rows)
    jvs_summary = _summary(jvs_rows)
    janon_summary = _summary(janon_rows)

    paths = {
        "jvs_metrics": out_dir / "jvs_native_metrics.csv",
        "jvs_feature_coverage": out_dir / "jvs_feature_coverage.csv",
        "jvs_distribution": out_dir / "jvs_score_distribution.csv",
        "jvs_percentiles": out_dir / "jvs_native_percentiles.json",
        "jvs_false_alarm": out_dir / "jvs_false_alarm_by_feature.csv",
        "janon_metrics": out_dir / "janon_l2_metrics.csv",
        "jvs_report": report_dir / "jvs_native_sanity_check.md",
        "coverage_report": report_dir / "calibration_coverage.md",
        "threshold_report": report_dir / "rhythm_fluency_threshold_update.md",
        "janon_report": report_dir / "janon_l2_trend_report.md",
    }
    _write_csv(paths["jvs_metrics"], jvs_rows)
    _write_csv(paths["jvs_feature_coverage"], jvs_coverage)
    _write_csv(paths["jvs_distribution"], jvs_dist)
    paths["jvs_percentiles"].write_text(json.dumps(jvs_percentiles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(paths["jvs_false_alarm"], _false_alarm_rows(jvs_rows))
    _write_csv(paths["janon_metrics"], janon_rows)
    _write_jvs_report(paths["jvs_report"], jvs_rows, jvs_summary)
    _write_coverage_report(paths["coverage_report"], jvs_coverage)
    _write_threshold_report(paths["threshold_report"], jvs_percentiles)
    _write_janon_report(paths["janon_report"], janon_rows, janon_summary, jvs_dist)
    return {
        "jvs_summary": jvs_summary,
        "janon_summary": janon_summary,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small native/L2 calibration snapshot without training models.")
    parser.add_argument("--jvs-root", default=str(PROJECT_ROOT / "JVS"))
    parser.add_argument("--janon-root", default=str(PROJECT_ROOT / "JANON"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "calibration"))
    parser.add_argument("--report-dir", default=str(ROOT / "reports"))
    parser.add_argument("--jvs-speakers", type=int, default=2)
    parser.add_argument("--jvs-utterances-per-speaker", type=int, default=5)
    parser.add_argument("--jvs-limit", type=int, default=None)
    parser.add_argument("--janon-limit", type=int, default=10)
    parser.add_argument("--stratify-by-feature", action="store_true", help="Accepted for calibration runs; feature strata are always included in coverage output.")
    parser.add_argument("--include-feature-coverage-report", action="store_true", help="Accepted for calibration runs; coverage reports are always written.")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--config", default=None)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
