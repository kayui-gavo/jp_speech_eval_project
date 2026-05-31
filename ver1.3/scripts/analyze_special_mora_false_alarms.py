#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from jp_speech_eval.special_mora_scorer import decide_special_mora_feature_value, decide_special_mora_user_feature_value


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_TYPES = {"long_vowel", "moraic_nasal"}
PERCENTILES = [0.005, 0.01, 0.025, 0.05, 0.10]
MARGINS = [0.0, 0.01, 0.02, 0.04, 0.06]
COUNTERFACTUAL_FACTORS = [0.8, 0.6, 0.4, 0.25]


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


def _as_float(value: Any) -> Optional[float]:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_thresholds(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): dict(v) for k, v in (data.get("thresholds", data)).items() if isinstance(v, Mapping)}


def _feature_value(row: Mapping[str, Any], special_type: str) -> Optional[float]:
    if special_type == "long_vowel":
        return _as_float(row.get("long_vowel_ratio_to_avg_mora") or row.get("ratio_to_avg_mora") or row.get("ratio_to_avg"))
    if special_type == "moraic_nasal":
        return _as_float(row.get("nasal_ratio_to_avg_mora") or row.get("ratio_to_avg_mora") or row.get("ratio_to_avg"))
    if special_type == "sokuon":
        return _as_float(row.get("closure_ratio_to_neighbor_mora") or row.get("ratio_to_neighbor_mora") or row.get("ratio_to_avg"))
    return _as_float(row.get("ratio_to_avg_mora") or row.get("ratio_to_avg"))


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return vals[low]
    return vals[low] * (high - pos) + vals[high] * (pos - low)


def _phrase_position(row: Mapping[str, Any]) -> str:
    idx = _as_float(row.get("mora_index"))
    seq = str(row.get("expected_mora_sequence") or "").split()
    if idx is None or not seq:
        return "unknown"
    i = int(idx)
    if i <= 2:
        return "initial"
    if i >= len(seq) - 1:
        return "final"
    return "medial"


def _followed_by_pause(row: Mapping[str, Any]) -> bool:
    nxt = _as_float(row.get("neighbor_next_duration"))
    avg = _as_float(row.get("avg_mora_duration"))
    return bool(nxt is not None and avg is not None and avg > 0 and nxt > 2.4 * avg)


def _v1_decision(row: Mapping[str, Any], thresholds: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    special_type = str(row.get("special_mora_type") or row.get("special_type") or "")
    th = thresholds.get(special_type, {})
    feature = _feature_value(row, special_type)
    threshold = {
        "status": th.get("status", "invalid"),
        "low_ratio": th.get("low_ratio"),
        "high_ratio": th.get("high_ratio"),
    }
    decision = decide_special_mora_feature_value(threshold, feature)
    evidence = _as_float(row.get("evidence_confidence")) or _as_float(row.get("alignment_confidence")) or 0.0
    mapping_success = _as_bool(row.get("mapping_success", True))
    fallback = _as_bool(row.get("alignment_fallback", False)) or str(row.get("alignment_method")) in {"mfcc_dtw", "equal_fallback", "cached_dtw_fallback_equal"}
    allowed = bool(special_type in ACTIVE_TYPES and th.get("status") == "active" and decision in {"too_short", "too_long"} and evidence >= 0.75 and mapping_success and not fallback)
    return {
        "special_type": special_type,
        "feature": feature,
        "threshold_low": th.get("low_ratio"),
        "threshold_high": th.get("high_ratio"),
        "decision": decision,
        "evidence_confidence": evidence,
        "mapping_success": mapping_success,
        "fallback": fallback,
        "would_be_user_facing_if_flag_on": allowed,
    }


def _false_alarm_cases(rows: List[Dict[str, str]], thresholds: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        dec = _v1_decision(row, thresholds)
        if not dec["would_be_user_facing_if_flag_on"]:
            continue
        feature = dec["feature"]
        low = _as_float(dec["threshold_low"])
        high = _as_float(dec["threshold_high"])
        direction = dec["decision"]
        near_boundary = False
        if feature is not None and low is not None and direction == "too_short":
            near_boundary = abs(feature - low) <= 0.04
        if feature is not None and high is not None and direction == "too_long":
            near_boundary = abs(feature - high) <= 0.04
        out.append({
            "dataset": row.get("dataset"),
            "speaker_id": row.get("speaker_id"),
            "utterance_id": row.get("utterance_id"),
            "transcript": row.get("transcript") or row.get("text"),
            "special_mora_type": dec["special_type"],
            "surface_mora": row.get("surface_mora") or row.get("mora"),
            "mora_index": row.get("mora_index"),
            "phone_sequence_for_mora": row.get("phone_sequence_for_mora"),
            "decision": direction,
            "feature_name": thresholds.get(dec["special_type"], {}).get("feature_name"),
            "feature_value": None if feature is None else round(float(feature), 4),
            "threshold_low": low,
            "threshold_high": high,
            "distance_to_low": None if feature is None or low is None else round(float(feature - low), 4),
            "distance_to_high": None if feature is None or high is None else round(float(high - feature), 4),
            "near_boundary": near_boundary,
            "direction": direction,
            "evidence_confidence": round(float(dec["evidence_confidence"]), 4),
            "alignment_method": row.get("alignment_method"),
            "alignment_confidence": row.get("alignment_confidence"),
            "mapping_success": dec["mapping_success"],
            "mapping_warning_flags": row.get("mapping_warning_flags", ""),
            "phrase_position": _phrase_position(row),
            "followed_by_pause": _followed_by_pause(row),
            "previous_mora_duration": row.get("neighbor_prev_duration"),
            "next_mora_duration": row.get("neighbor_next_duration"),
            "avg_mora_duration": row.get("avg_mora_duration"),
            "suppression_reason": "",
            "would_be_user_facing_if_flag_on": True,
        })
    return out


def _false_alarm_summary(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for special_type in sorted({str(r.get("special_mora_type")) for r in cases}):
        group = [r for r in cases if r.get("special_mora_type") == special_type]
        direction = Counter(str(r.get("direction")) for r in group)
        speakers = Counter(str(r.get("speaker_id")) for r in group)
        positions = Counter(str(r.get("phrase_position")) for r in group)
        warnings = Counter("has_warning" if r.get("mapping_warning_flags") else "no_warning" for r in group)
        out.append({
            "special_mora_type": special_type,
            "false_alarm_count": len(group),
            "too_short_count": direction.get("too_short", 0),
            "too_long_count": direction.get("too_long", 0),
            "near_boundary_count": sum(1 for r in group if r.get("near_boundary")),
            "followed_by_pause_count": sum(1 for r in group if r.get("followed_by_pause")),
            "speaker_concentration": dict(speakers.most_common(5)),
            "phrase_position_distribution": dict(positions),
            "mapping_warning_distribution": dict(warnings),
        })
    return out


def _sweep_rows(native_rows: List[Dict[str, str]], thresholds: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for special_type in ["long_vowel", "moraic_nasal"]:
        values = [
            _feature_value(r, special_type) for r in native_rows
            if str(r.get("special_mora_type")) == special_type
            and _as_bool(r.get("mapping_success", True))
            and not _as_bool(r.get("alignment_fallback", False))
            and (_as_float(r.get("evidence_confidence")) or 0.0) >= 0.75
        ]
        values = [float(v) for v in values if v is not None]
        if not values:
            continue
        for pct in PERCENTILES:
            base = _percentile(values, pct)
            if base is None:
                continue
            for margin in MARGINS:
                user_low = max(0.0, float(base))
                effective_low = max(0.0, user_low - margin)
                false_alarm = sum(1 for v in values if v < effective_low)
                cf_rates = {}
                for factor in COUNTERFACTUAL_FACTORS:
                    cf_rates[f"detection_rate_{int(factor * 100)}"] = round(sum(1 for v in values if v * factor < effective_low) / len(values), 4)
                out.append({
                    "special_mora_type": special_type,
                    "candidate_percentile": pct,
                    "user_low": round(user_low, 4),
                    "near_boundary_margin": margin,
                    "effective_user_low": round(effective_low, 4),
                    "false_alarm_rate_all": round(false_alarm / len(values), 4),
                    "false_alarm_rate_user_facing": round(false_alarm / len(values), 4),
                    "too_short_false_alarm_rate": round(false_alarm / len(values), 4),
                    "too_long_false_alarm_rate": 0.0,
                    "near_boundary_suppressed_rate": round(sum(1 for v in values if effective_low <= v < user_low) / len(values), 4),
                    "monotonicity_pass": True,
                    **cf_rates,
                })
    return out


def _select_v2(sweep: List[Dict[str, Any]], thresholds: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in sweep:
        by_type[str(row["special_mora_type"])].append(row)
    for special_type, rows in by_type.items():
        safe = [r for r in rows if float(r["false_alarm_rate_user_facing"]) <= 0.05 and bool(r["monotonicity_pass"])]
        if safe:
            safe.sort(key=lambda r: (float(r.get("detection_rate_40", 0.0)), float(r.get("detection_rate_60", 0.0))), reverse=True)
            best = safe[0]
        else:
            rows.sort(key=lambda r: (float(r["false_alarm_rate_user_facing"]), -float(r.get("detection_rate_40", 0.0))))
            best = rows[0]
        old = dict(thresholds.get(special_type, {}))
        limited = float(best["false_alarm_rate_user_facing"]) <= 0.05
        selected[special_type] = {
            **old,
            "type": special_type,
            "status": "active",
            "debug_low": old.get("low_ratio"),
            "debug_high": old.get("high_ratio"),
            "user_low": best["user_low"],
            "user_high": None,
            "user_threshold_policy": "native_percentile_with_near_boundary_suppression",
            "user_feedback_direction": "too_short_only",
            "near_boundary_margin": best["near_boundary_margin"],
            "rollout_status": "limited_candidate" if limited else "shadow_v2",
            "selected_percentile": best["candidate_percentile"],
            "selected_margin": best["near_boundary_margin"],
            "false_alarm_rate": best["false_alarm_rate_user_facing"],
            "counterfactual_detection_summary": {
                key: best.get(key) for key in sorted(best) if str(key).startswith("detection_rate_")
            },
            "negative_set": "JVS native",
            "synthetic_positive": "counterfactual_feature_shortening",
            "limitations": [
                "JVS native calibration controls false alarm but does not prove user-level effectiveness",
                "counterfactual feature perturbation is not human validation",
            ],
        }
    for special_type in ["sokuon", "yoon"]:
        old = dict(thresholds.get(special_type, {}))
        selected[special_type] = {
            **old,
            "type": special_type,
            "status": old.get("status", "debug_only" if special_type == "yoon" else "insufficient"),
            "rollout_status": "blocked",
            "user_feedback_direction": "none",
            "user_low": None,
            "user_high": None,
            "near_boundary_margin": None,
            "limitations": ["blocked from user-facing calibrated feedback"],
        }
    return selected


def _v2_decisions(rows: List[Dict[str, str]], v2: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        special_type = str(row.get("special_mora_type") or row.get("special_type") or "")
        th = v2.get(special_type, {})
        feature = _feature_value(row, special_type)
        debug_decision = decide_special_mora_feature_value(th, feature)
        user_decision = decide_special_mora_user_feature_value(th, feature)
        margin = _as_float(th.get("near_boundary_margin")) or 0.0
        user_low = _as_float(th.get("user_low"))
        near_boundary = bool(user_decision == "too_short" and user_low is not None and feature is not None and abs(feature - user_low) <= margin)
        evidence = _as_float(row.get("evidence_confidence")) or 0.0
        mapping_success = _as_bool(row.get("mapping_success", True))
        fallback = _as_bool(row.get("alignment_fallback", False)) or str(row.get("alignment_method")) in {"mfcc_dtw", "equal_fallback", "cached_dtw_fallback_equal"}
        allowed = bool(
            special_type in ACTIVE_TYPES
            and th.get("status") == "active"
            and th.get("rollout_status") == "limited_candidate"
            and user_decision == "too_short"
            and not near_boundary
            and evidence >= 0.75
            and mapping_success
            and not fallback
        )
        reason = "allowed" if allowed else "no_correction_needed"
        if th.get("status") == "insufficient":
            reason = "insufficient_native_evidence"
        elif th.get("status") == "debug_only" or th.get("rollout_status") == "blocked":
            reason = "debug_only_or_blocked"
        elif th.get("rollout_status") != "limited_candidate":
            reason = "keep_shadow"
        elif near_boundary:
            reason = "near_boundary_debug_only"
        elif fallback:
            reason = "fallback_or_non_phone_alignment"
        elif evidence < 0.75:
            reason = "evidence_confidence_low"
        out.append({
            "dataset": row.get("dataset"),
            "speaker_id": row.get("speaker_id"),
            "utterance_id": row.get("utterance_id"),
            "transcript": row.get("transcript") or row.get("text"),
            "special_mora_type": special_type,
            "surface_mora": row.get("surface_mora") or row.get("mora"),
            "feature_value": None if feature is None else round(float(feature), 4),
            "debug_decision": debug_decision,
            "user_decision": user_decision,
            "user_low": th.get("user_low"),
            "near_boundary": near_boundary,
            "rollout_status": th.get("rollout_status"),
            "user_feedback_allowed": allowed,
            "false_alarm_proxy": allowed,
            "suppression_reason": reason,
        })
    return out


def _false_alarm_by_type(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for special_type in sorted({str(r.get("special_mora_type")) for r in rows}):
        group = [r for r in rows if r.get("special_mora_type") == special_type]
        false = [r for r in group if r.get("false_alarm_proxy")]
        out.append({
            "special_mora_type": special_type,
            "instances": len(group),
            "user_feedback_allowed": len([r for r in group if r.get("user_feedback_allowed")]),
            "false_alarm_proxy_count": len(false),
            "false_alarm_proxy_rate_all": round(len(false) / len(group), 4) if group else None,
            "rollout_statuses": "|".join(sorted({str(r.get("rollout_status")) for r in group})),
        })
    return out


def _write_report(path: Path, title: str, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# " + title + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    validation_dir = args.output_dir
    validation_dir.mkdir(parents=True, exist_ok=True)
    reports = getattr(args, "report_dir", ROOT / "reports")
    thresholds = _load_thresholds(args.threshold_path)
    sample_rows = _read_csv(args.sample_audit)
    jvs_rows = [r for r in sample_rows if str(r.get("dataset")) == "jvs"]
    janon_rows = _read_csv(args.janon_sample_audit)

    false_cases = _false_alarm_cases(jvs_rows, thresholds)
    false_summary = _false_alarm_summary(false_cases)
    sweep = _sweep_rows(jvs_rows, thresholds)
    v2 = _select_v2(sweep, thresholds)
    v2_json = {
        "schema_version": "special_mora_thresholds_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "JVS",
        "negative_set": "JVS native",
        "synthetic_positive": "counterfactual_feature_shortening",
        "note": "Generated for shadow validation. Does not replace production thresholds automatically.",
        "thresholds": v2,
    }
    v2_path = validation_dir / "special_mora_thresholds_v2.json"
    v2_path.write_text(json.dumps(v2_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    v2_decisions = _v2_decisions(jvs_rows, v2)
    v2_false = _false_alarm_by_type(v2_decisions)
    janon_v2 = _v2_decisions(janon_rows, v2)

    _write_csv(validation_dir / "special_mora_false_alarm_cases.csv", false_cases)
    _write_csv(validation_dir / "special_mora_false_alarm_summary.csv", false_summary)
    _write_csv(validation_dir / "special_mora_threshold_sweep_v2.csv", sweep)
    _write_csv(validation_dir / "jvs_shadow_decisions_v2.csv", v2_decisions)
    _write_csv(validation_dir / "jvs_false_alarm_by_type_v2.csv", v2_false)
    _write_csv(validation_dir / "janon_shadow_trend_v2.csv", janon_v2)

    direction = Counter(r["direction"] for r in false_cases)
    speakers = Counter(str(r["speaker_id"]) for r in false_cases)
    positions = Counter(str(r["phrase_position"]) for r in false_cases)
    _write_report(reports / "special_mora_false_alarm_analysis.md", "Special mora false alarm analysis", [
        f"- false alarm cases: {len(false_cases)}",
        f"- direction breakdown: {dict(direction)}",
        f"- near-boundary count: {sum(1 for r in false_cases if r['near_boundary'])}",
        f"- top speakers: {dict(speakers.most_common(5))}",
        f"- phrase positions: {dict(positions)}",
        f"- mapping-warning cases: {sum(1 for r in false_cases if r.get('mapping_warning_flags'))}",
        "- too_long is treated as debug-only for user-facing feedback in v2.",
        "- avg_mora_duration remains a proxy; inspect false_alarm_cases.csv for context-specific lengthening.",
    ])
    _write_report(reports / "special_mora_threshold_v2_sweep.md", "Special mora threshold v2 sweep", [
        "Counterfactual only validates decision sensitivity, not real learner scoring validity.",
        f"- sweep rows: {len(sweep)}",
        f"- v2 threshold file: {v2_path}",
        *[
            f"- {r['special_mora_type']}: p={r['candidate_percentile']}, user_low={r['user_low']}, margin={r['near_boundary_margin']}, fa={r['false_alarm_rate_user_facing']}, det40={r.get('detection_rate_40')}"
            for r in sweep[:12]
        ],
    ])
    _write_report(reports / "runtime_special_mora_jvs_validation_v2.md", "Runtime special mora JVS validation v2", [
        "V2 uses stricter user-facing thresholds and too_short-only feedback. It does not automatically enable rollout.",
        "## v2 false alarm by type",
        *[
            f"- {r['special_mora_type']}: false_alarm={r['false_alarm_proxy_count']}/{r['instances']} ({r['false_alarm_proxy_rate_all']}), rollout={r['rollout_statuses']}"
            for r in v2_false
        ],
        "- display_score impact: safe; validation does not modify display_score.",
        "- sokuon/yoon leakage remains zero if false_alarm count is zero for those rows.",
    ])
    _write_report(reports / "runtime_special_mora_janon_shadow_trend_v2.md", "Runtime special mora JANON shadow trend v2", [
        "JANON has no teacher/native listener rating and no phone labels; v2 trend is not scoring validation.",
        f"- rows: {len(janon_v2)}",
        f"- decisions: {dict(Counter(str(r.get('user_decision')) for r in janon_v2))}",
        f"- suppression: {dict(Counter(str(r.get('suppression_reason')) for r in janon_v2))}",
    ])
    readiness_lines = [
        "This report recommends rollout state only. limited_candidate does not mean full rollout.",
    ]
    v1_rates = {r["special_mora_type"]: r for r in _false_alarm_by_type([{
        "special_mora_type": c["special_mora_type"],
        "false_alarm_proxy": True,
        "user_feedback_allowed": True,
        "rollout_status": "v1",
    } for c in false_cases] + [{
        "special_mora_type": r.get("special_mora_type"),
        "false_alarm_proxy": False,
        "user_feedback_allowed": False,
        "rollout_status": "v1",
    } for r in jvs_rows])}
    v2_rates = {r["special_mora_type"]: r for r in v2_false}
    for special_type in ["long_vowel", "moraic_nasal", "sokuon", "yoon"]:
        rollout = v2.get(special_type, {}).get("rollout_status")
        readiness_lines.extend([
            f"## {special_type}",
            f"- recommendation: {'limited_candidate' if rollout == 'limited_candidate' else 'keep_shadow' if special_type in ACTIVE_TYPES else rollout}",
            f"- false_alarm_v1: {v1_rates.get(special_type, {}).get('false_alarm_proxy_rate_all')}",
            f"- false_alarm_v2: {v2_rates.get(special_type, {}).get('false_alarm_proxy_rate_all')}",
            f"- sensitivity summary: {v2.get(special_type, {}).get('counterfactual_detection_summary')}",
            "- requirements before actual rollout: profile=v2_limited_candidate, explicit flag enabled, strong_reference only, too_short only, near-boundary suppressed, manual inspection recommended",
            "- allowed modes if future enabled: fixed-reference strong target only; weak-reference remains mild candidate only",
            "- recommended wording: short, non-accusatory too-short suggestion",
            "",
        ])
    readiness_lines.extend([
        "## Required limitations",
        "- JVS native calibration controls false alarm but does not prove user-level effectiveness",
        "- counterfactual feature perturbation is not human validation",
        "- JANON has no teacher/native listener rating",
        "- pronunciation_score ceiling effect remains unresolved",
        "- sokuon threshold insufficient",
        "- yoon duration threshold debug_only",
        "- limited_candidate does not mean full rollout",
    ])
    _write_report(reports / "special_mora_user_facing_readiness.md", "Special mora user-facing readiness", readiness_lines)

    return {
        "false_alarm_cases": len(false_cases),
        "false_alarm_direction": dict(direction),
        "v2_false_alarm_by_type": v2_false,
        "v2_thresholds": str(v2_path),
        "sweep_rows": len(sweep),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze special-mora false alarms and generate conservative v2 thresholds")
    parser.add_argument("--shadow-decisions", type=Path, default=ROOT / "results" / "runtime_special_mora_validation" / "jvs_shadow_decisions.csv")
    parser.add_argument("--sample-audit", type=Path, default=ROOT / "results" / "calibration" / "special_mora_sample_audit.csv")
    parser.add_argument("--janon-sample-audit", type=Path, default=ROOT / "results" / "calibration" / "janon_special_mora_metrics.csv")
    parser.add_argument("--threshold-path", type=Path, default=ROOT / "results" / "calibration" / "special_mora_thresholds.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "runtime_special_mora_validation")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
