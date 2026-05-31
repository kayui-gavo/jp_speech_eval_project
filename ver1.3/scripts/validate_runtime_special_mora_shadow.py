#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from jp_speech_eval.special_mora_scorer import (
    decide_special_mora_feature_value,
    load_runtime_special_mora_thresholds,
)


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_USER_TYPES = {"long_vowel", "moraic_nasal"}
PERTURB_FACTORS = [1.0, 0.8, 0.6, 0.4, 0.25]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: Any) -> Optional[float]:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _feature_value(row: Mapping[str, Any], special_type: str) -> Optional[float]:
    if special_type == "long_vowel":
        return _as_float(row.get("long_vowel_ratio_to_avg_mora") or row.get("ratio_to_avg_mora") or row.get("ratio_to_avg"))
    if special_type == "moraic_nasal":
        return _as_float(row.get("nasal_ratio_to_avg_mora") or row.get("ratio_to_avg_mora") or row.get("ratio_to_avg"))
    if special_type == "sokuon":
        return _as_float(row.get("closure_ratio_to_neighbor_mora") or row.get("ratio_to_neighbor_mora") or row.get("ratio_to_avg"))
    if special_type == "yoon":
        return _as_float(row.get("ratio_to_avg_mora") or row.get("ratio_to_avg"))
    return _as_float(row.get("ratio_to_avg_mora") or row.get("ratio_to_avg"))


def _shadow_decision_rows(sample_rows: List[Dict[str, str]], thresholds: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in sample_rows:
        special_type = str(row.get("special_mora_type") or row.get("special_type") or "")
        th = thresholds.get(special_type, {})
        feature = _feature_value(row, special_type)
        decision = decide_special_mora_feature_value(th, feature)
        evidence_conf = _as_float(row.get("evidence_confidence")) or _as_float(row.get("alignment_confidence")) or 0.0
        mapping_success = _as_bool(row.get("mapping_success", True))
        alignment_fallback = _as_bool(row.get("alignment_fallback", False)) or str(row.get("alignment_method")) in {"equal_fallback", "cached_dtw_fallback_equal", "mfcc_dtw"}
        threshold_status = str(th.get("status") or "invalid")
        user_feedback_allowed = bool(
            special_type in ACTIVE_USER_TYPES
            and threshold_status == "active"
            and decision in {"too_short", "too_long"}
            and evidence_conf >= 0.75
            and mapping_success
            and not alignment_fallback
        )
        suppression = ""
        if not user_feedback_allowed:
            if threshold_status == "insufficient":
                suppression = "insufficient_native_evidence"
            elif threshold_status == "debug_only":
                suppression = "debug_only_threshold"
            elif threshold_status != "active":
                suppression = "missing_or_invalid_threshold_metadata"
            elif decision == "ok":
                suppression = "no_correction_needed"
            elif alignment_fallback:
                suppression = "fallback_or_non_phone_alignment"
            elif evidence_conf < 0.75:
                suppression = "evidence_confidence_low"
            elif not mapping_success:
                suppression = "mapping_failed"
            else:
                suppression = "not_user_facing_type"
        out.append({
            "dataset": row.get("dataset"),
            "speaker_id": row.get("speaker_id"),
            "utterance_id": row.get("utterance_id"),
            "audio_path": row.get("audio_path"),
            "transcript": row.get("transcript") or row.get("text"),
            "special_mora_type": special_type,
            "surface_mora": row.get("surface_mora") or row.get("mora"),
            "mora_index": row.get("mora_index"),
            "feature_value": None if feature is None else round(float(feature), 4),
            "threshold_low": th.get("low_ratio"),
            "threshold_high": th.get("high_ratio"),
            "threshold_status": threshold_status,
            "decision": decision,
            "evidence_confidence": round(float(evidence_conf), 4),
            "alignment_method": row.get("alignment_method"),
            "alignment_fallback": alignment_fallback,
            "mapping_success": mapping_success,
            "mapping_warning_flags": row.get("mapping_warning_flags", ""),
            "user_feedback_allowed": user_feedback_allowed,
            "false_alarm_proxy": user_feedback_allowed and decision in {"too_short", "too_long"},
            "suppression_reason": suppression,
            "special_mora_score_available": threshold_status == "active" and decision in {"ok", "too_short", "too_long"} and evidence_conf >= 0.45,
        })
    return out


def _false_alarm_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("special_mora_type"))].append(row)
    out: List[Dict[str, Any]] = []
    for special_type, group in sorted(by_type.items()):
        active = [r for r in group if r.get("threshold_status") == "active"]
        allowed = [r for r in group if r.get("user_feedback_allowed")]
        false_alarm = [r for r in group if r.get("false_alarm_proxy")]
        out.append({
            "special_mora_type": special_type,
            "instances": len(group),
            "active_instances": len(active),
            "user_feedback_allowed": len(allowed),
            "false_alarm_proxy_count": len(false_alarm),
            "false_alarm_proxy_rate_all": round(len(false_alarm) / len(group), 4) if group else None,
            "false_alarm_proxy_rate_active": round(len(false_alarm) / len(active), 4) if active else None,
            "threshold_statuses": "|".join(sorted({str(r.get("threshold_status")) for r in group})),
        })
    return out


def _counterfactual_rows(jvs_rows: List[Dict[str, str]], thresholds: Mapping[str, Mapping[str, Any]], limit: int = 80) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    candidates = [
        row for row in jvs_rows
        if str(row.get("special_mora_type")) in {"long_vowel", "moraic_nasal"}
        and str(thresholds.get(str(row.get("special_mora_type")), {}).get("status")) == "active"
        and _as_bool(row.get("mapping_success", True))
        and (_as_float(row.get("evidence_confidence")) or 0.0) >= 0.75
    ][:limit]
    for row in candidates:
        special_type = str(row.get("special_mora_type"))
        th = thresholds.get(special_type, {})
        original = _feature_value(row, special_type)
        if original is None:
            continue
        previous_rank = -1
        for factor in PERTURB_FACTORS:
            value = original * factor
            decision = decide_special_mora_feature_value(th, value)
            rank = 1 if decision == "too_short" else 0
            rows.append({
                "dataset": row.get("dataset"),
                "speaker_id": row.get("speaker_id"),
                "utterance_id": row.get("utterance_id"),
                "special_mora_type": special_type,
                "surface_mora": row.get("surface_mora") or row.get("mora"),
                "original_feature_value": round(float(original), 4),
                "factor": factor,
                "counterfactual_feature_value": round(float(value), 4),
                "threshold_low": th.get("low_ratio"),
                "threshold_high": th.get("high_ratio"),
                "decision": decision,
                "too_short": decision == "too_short",
                "monotonic_step_ok": rank >= previous_rank if previous_rank >= 0 else True,
            })
            previous_rank = max(previous_rank, rank)
    return rows


def _counterfactual_summary(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for special_type in sorted({str(r.get("special_mora_type")) for r in rows}):
        group = [r for r in rows if r.get("special_mora_type") == special_type]
        by_factor = {}
        for factor in PERTURB_FACTORS:
            sub = [r for r in group if float(r.get("factor")) == factor]
            by_factor[str(factor)] = round(sum(1 for r in sub if r.get("too_short")) / len(sub), 4) if sub else None
        summary[special_type] = {
            "n_instances": len({(r.get("speaker_id"), r.get("utterance_id"), r.get("surface_mora"), r.get("original_feature_value")) for r in group}),
            "too_short_detection_rate_by_factor": by_factor,
            "monotonicity_ok": all(bool(r.get("monotonic_step_ok")) for r in group),
        }
    return summary


def _janon_trend_rows(rows: List[Dict[str, str]], thresholds: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    shadow = _shadow_decision_rows(rows, thresholds)
    for row in shadow:
        row["trend_only"] = True
    return shadow


def _rate_line(rows: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for row in rows:
        lines.append(
            f"- {row['special_mora_type']}: instances={row['instances']}, active={row['active_instances']}, "
            f"user_feedback_allowed={row['user_feedback_allowed']}, false_alarm_proxy={row['false_alarm_proxy_count']} "
            f"(all={row['false_alarm_proxy_rate_all']}, active={row['false_alarm_proxy_rate_active']}), statuses={row['threshold_statuses']}"
        )
    return lines


def _write_jvs_report(path: Path, decision_rows: List[Dict[str, Any]], false_alarm_rows: List[Dict[str, Any]], thresholds: Mapping[str, Mapping[str, Any]]) -> None:
    utterances = len({str(r.get("audio_path") or r.get("utterance_id")) for r in decision_rows})
    suppression = Counter(str(r.get("suppression_reason") or "allowed") for r in decision_rows)
    fallback_rate = round(sum(1 for r in decision_rows if r.get("alignment_fallback")) / len(decision_rows), 4) if decision_rows else 0.0
    low_evidence_rate = round(sum(1 for r in decision_rows if float(r.get("evidence_confidence") or 0.0) < 0.75) / len(decision_rows), 4) if decision_rows else 0.0
    missing_threshold_rate = round(sum(1 for r in decision_rows if r.get("threshold_status") == "invalid") / len(decision_rows), 4) if decision_rows else 0.0
    leakage = sum(1 for r in decision_rows if r.get("special_mora_type") in {"sokuon", "yoon"} and r.get("user_feedback_allowed"))
    lines = [
        "# Runtime special mora JVS validation",
        "",
        "JVS is treated as native speech for false-alarm sanity checks. This reduces false-alarm risk but does not prove user-facing effectiveness.",
        "",
        f"- total utterances: {utterances}",
        f"- total special mora instances: {len(decision_rows)}",
        f"- equal/non-phone fallback rate: {fallback_rate}",
        f"- low evidence rate: {low_evidence_rate}",
        f"- missing threshold rate: {missing_threshold_rate}",
        f"- sokuon/yoon user-facing leakage: {leakage}",
        f"- display_score impact check: safe; shadow validation does not modify display_score",
        "",
        "## Threshold status",
    ]
    for key, th in sorted(thresholds.items()):
        lines.append(f"- {key}: status={th.get('status')}, sample_count={th.get('sample_count')}, source={th.get('source_dataset')}")
    lines.extend(["", "## False alarm proxy by type", *_rate_line(false_alarm_rows), "", "## Suppression reasons"])
    for reason, count in sorted(suppression.items()):
        lines.append(f"- {reason}: {count}")
    lines.extend([
        "",
        "## Answer",
        "- 長音 native false alarm should be judged from the table above; <= 5% is the rollout target.",
        "- 撥音 native false alarm should be judged from the table above; <= 5% is the rollout target.",
        "- sokuon/yoon must remain zero leakage before any user-facing release.",
        "- Concentrated transcript/mapping errors should be inspected in `jvs_shadow_decisions.csv`.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_counterfactual_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Runtime special mora counterfactual sensitivity",
        "",
        "Counterfactual feature perturbation validates threshold decision sensitivity only; it does not prove real learner scoring validity.",
        "",
    ]
    for special_type, item in summary.items():
        lines.append(f"## {special_type}")
        lines.append(f"- n_instances: {item.get('n_instances')}")
        lines.append(f"- monotonicity_ok: {item.get('monotonicity_ok')}")
        rates = item.get("too_short_detection_rate_by_factor") or {}
        for factor in PERTURB_FACTORS:
            lines.append(f"- shortened_{int(factor * 100)}_percent_detection_rate: {rates.get(str(factor))}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_janon_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    by_type = Counter(str(r.get("special_mora_type")) for r in rows)
    decisions = Counter(str(r.get("decision")) for r in rows)
    suppression = Counter(str(r.get("suppression_reason") or "allowed") for r in rows)
    uncertain_rate = round(sum(1 for r in rows if r.get("decision") == "uncertain") / len(rows), 4) if rows else 0.0
    failure_rate = round(sum(1 for r in rows if not r.get("mapping_success")) / len(rows), 4) if rows else 0.0
    lines = [
        "# Runtime special mora JANON shadow trend",
        "",
        "JANON has no teacher/native listener rating and no phone labels; these results are trend-only and cannot prove scoring correctness.",
        "",
        f"- rows: {len(rows)}",
        f"- by_type: {dict(by_type)}",
        f"- decisions: {dict(decisions)}",
        f"- uncertain_rate: {uncertain_rate}",
        f"- alignment/mapping failure rate proxy: {failure_rate}",
        f"- suppression reasons: {dict(suppression)}",
        "",
        "Interpretation: use this only to spot suspicious trends, not as ground truth.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _readiness_status(special_type: str, false_alarm_rows: List[Dict[str, Any]], counter_summary: Mapping[str, Any], thresholds: Mapping[str, Mapping[str, Any]]) -> Dict[str, str]:
    th = thresholds.get(special_type, {})
    status = str(th.get("status") or "invalid")
    row = next((r for r in false_alarm_rows if r.get("special_mora_type") == special_type), {})
    fa_rate = row.get("false_alarm_proxy_rate_all")
    cf = counter_summary.get(special_type, {})
    rates = cf.get("too_short_detection_rate_by_factor", {}) if isinstance(cf, Mapping) else {}
    detect_40 = rates.get("0.4")
    if special_type == "sokuon":
        return {"status": "blocked", "reason": "insufficient native reliable count", "next_step": "collect/search more JVS sokuon or improve closure evidence"}
    if special_type == "yoon":
        return {"status": "blocked/debug_only", "reason": "duration threshold inappropriate", "next_step": "design mora_count_consistency evidence"}
    if status != "active":
        return {"status": "keep_shadow", "reason": f"threshold status is {status}", "next_step": "fix threshold metadata"}
    ready = fa_rate is not None and float(fa_rate) <= 0.05 and detect_40 is not None and float(detect_40) > 0.0
    return {
        "status": "ready_for_limited_user_facing" if ready else "keep_shadow",
        "reason": f"native_false_alarm_proxy_all={fa_rate}, shortened_40_detection_rate={detect_40}",
        "required_flag": "enable_user_facing_calibrated_special_mora=True",
        "allowed_modes": "fixed-reference strong target only; weak-reference mild candidate only",
        "feedback_wording": "short, non-accusatory length/nasal-hold suggestion",
    }


def _write_readiness_report(path: Path, false_alarm_rows: List[Dict[str, Any]], counter_summary: Mapping[str, Any], thresholds: Mapping[str, Mapping[str, Any]]) -> None:
    lines = [
        "# Special mora user-facing readiness",
        "",
        "This report recommends rollout state only. It does not automatically enable user-facing calibrated special mora feedback.",
        "",
    ]
    for special_type in ["long_vowel", "moraic_nasal", "sokuon", "yoon"]:
        item = _readiness_status(special_type, false_alarm_rows, counter_summary, thresholds)
        lines.append(f"## {special_type}")
        for key, value in item.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend([
        "## Required limitations",
        "- counterfactual feature perturbation is not human validation",
        "- JANON has no teacher/native listener rating",
        "- pronunciation_score ceiling effect still unresolved",
        "- sokuon threshold insufficient",
        "- yoon duration threshold debug_only",
        "- JVS native calibration reduces false alarm risk but does not prove user-facing effectiveness",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    thresholds = load_runtime_special_mora_thresholds(args.threshold_path)
    jvs_rows_all = _read_csv(args.jvs_sample_audit)
    janon_rows_all = _read_csv(args.janon_sample_audit)
    type_filter = {x.strip() for x in str(args.special_mora_types).split(",") if x.strip()}
    if type_filter:
        jvs_rows_all = [r for r in jvs_rows_all if (r.get("special_mora_type") or r.get("special_type")) in type_filter]
        janon_rows_all = [r for r in janon_rows_all if (r.get("special_mora_type") or r.get("special_type")) in type_filter]
    if args.jvs_limit:
        jvs_rows_all = jvs_rows_all[: int(args.jvs_limit)]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = getattr(args, "report_dir", ROOT / "reports")

    jvs_decisions = _shadow_decision_rows(jvs_rows_all, thresholds)
    false_alarm = _false_alarm_rows(jvs_decisions)
    counter = _counterfactual_rows(jvs_rows_all, thresholds)
    counter_summary = _counterfactual_summary(counter)
    janon_trend = _janon_trend_rows(janon_rows_all, thresholds)

    _write_csv(out_dir / "jvs_shadow_decisions.csv", jvs_decisions)
    _write_csv(out_dir / "jvs_false_alarm_by_type.csv", false_alarm)
    _write_csv(out_dir / "counterfactual_feature_sensitivity.csv", counter)
    _write_csv(out_dir / "janon_shadow_trend.csv", janon_trend)
    _write_jvs_report(report_dir / "runtime_special_mora_jvs_validation.md", jvs_decisions, false_alarm, thresholds)
    _write_counterfactual_report(report_dir / "runtime_special_mora_counterfactual_sensitivity.md", counter_summary)
    _write_janon_report(report_dir / "runtime_special_mora_janon_shadow_trend.md", janon_trend)
    _write_readiness_report(report_dir / "special_mora_user_facing_readiness.md", false_alarm, counter_summary, thresholds)
    return {
        "jvs_decisions": len(jvs_decisions),
        "janon_trend": len(janon_trend),
        "counterfactual_rows": len(counter),
        "false_alarm_by_type": false_alarm,
        "counterfactual_summary": counter_summary,
        "output_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate runtime special-mora shadow decisions with JVS/JANON audit rows")
    parser.add_argument("--jvs-speakers", type=int, default=0, help="Accepted for compatibility; use calibration snapshot to regenerate sample audit.")
    parser.add_argument("--jvs-utterances-per-speaker", type=int, default=0, help="Accepted for compatibility; use calibration snapshot to regenerate sample audit.")
    parser.add_argument("--jvs-limit", type=int, default=0)
    parser.add_argument("--focus-special-mora", action="store_true")
    parser.add_argument("--special-mora-types", default="long_vowel,moraic_nasal,sokuon,yoon")
    parser.add_argument("--alignment-backend", default="auto")
    parser.add_argument("--threshold-path", type=Path, default=ROOT / "results" / "calibration" / "special_mora_thresholds.json")
    parser.add_argument("--jvs-sample-audit", type=Path, default=ROOT / "results" / "calibration" / "special_mora_sample_audit.csv")
    parser.add_argument("--janon-sample-audit", type=Path, default=ROOT / "results" / "calibration" / "janon_special_mora_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "runtime_special_mora_validation")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
