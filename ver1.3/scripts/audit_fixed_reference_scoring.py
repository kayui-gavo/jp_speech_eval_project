from __future__ import annotations

import argparse
import csv
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "jp_speech_eval_matplotlib"))

from jp_speech_eval.eval_modes import evaluate_mode
from jp_speech_eval.feedback_renderer import render_user_facing_result


FIELDNAMES = [
    "sample_id",
    "target_text",
    "audio_type",
    "reference_type",
    "expected_behavior",
    "recording_gate",
    "content_gate",
    "alignment_gate",
    "pronunciation_evidence_gate",
    "score_available",
    "display_score",
    "display_score_before_cap",
    "display_score_after_cap",
    "display_cap_applied",
    "display_cap_reason",
    "display_cap_reduction",
    "pronunciation_score",
    "raw_total_score",
    "raw_prosody_score",
    "dtw_distance",
    "alignment_confidence",
    "f0_coverage",
    "pitch_feedback_allowed",
    "pitch_suppression_reason",
    "pitch_text_leakage_warning",
    "pitch_text_leakage_terms",
    "special_mora_user_facing_count",
    "special_mora_evidence_level",
    "special_mora_suppression_reason",
    "special_mora_suppressed",
    "rhythm_timing_penalty_reason",
    "warning_codes",
    "suppressed_reasons",
    "user_message_type",
    "user_facing_message",
    "status",
    "error",
]


def _read_manifest(path: str | Path) -> List[Dict[str, str]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Expected CSV columns include sample_id,audio_path,target_text."
        )
    with manifest_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick(mapping: Mapping[str, Any], *path: str, default: Any = "") -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _gate_flag(user_facing: Mapping[str, Any], reason: str) -> str:
    reasons = set(str(item) for item in user_facing.get("suppressed_reasons") or [])
    warnings = set(str(item) for item in user_facing.get("score_policy_warnings") or [])
    return "blocked" if reason in reasons or reason in warnings else "ok"


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ok"}


def _float_or_none(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(values: Iterable[bool]) -> float:
    data = list(values)
    if not data:
        return 0.0
    return round(sum(1 for item in data if item) / len(data), 4)


def _summary_stats(values: Iterable[Any]) -> Dict[str, Any]:
    nums = sorted(value for value in (_float_or_none(v) for v in values) if value is not None)
    if not nums:
        return {"mean": "", "median": "", "p10": "", "p90": ""}
    p10_idx = max(0, min(len(nums) - 1, round((len(nums) - 1) * 0.10)))
    p90_idx = max(0, min(len(nums) - 1, round((len(nums) - 1) * 0.90)))
    return {
        "mean": round(float(statistics.mean(nums)), 2),
        "median": round(float(statistics.median(nums)), 2),
        "p10": round(float(nums[p10_idx]), 2),
        "p90": round(float(nums[p90_idx]), 2),
    }


def _first_nonempty(items: Iterable[Any]) -> str:
    for item in items:
        text = str(item or "").strip()
        if text:
            return text
    return ""


PITCH_LEAKAGE_TERMS = (
    "音高",
    "語調",
    "韻律",
    "アクセント",
    "イントネーション",
    "ピッチ",
    "pitch",
    "prosody",
    "intonation",
    "accent",
)


NEGATIVE_EXPECTED = {
    "content_mismatch_should_not_score",
    "recording_bad_should_not_score",
    "alignment_bad_should_not_score",
}


NEGATIVE_AUDIO_TYPES = {
    "wrong_target",
    "wrong_sentence",
    "content_mismatch",
    "nonsense",
    "random_speech",
    "english",
    "bad_recording",
}


def _pitch_leakage_terms(message: str) -> List[str]:
    lower = message.lower()
    return [term for term in PITCH_LEAKAGE_TERMS if term.lower() in lower]


def _join_counts(values: Iterable[Any]) -> str:
    counts = Counter(str(v or "").strip() for v in values if str(v or "").strip())
    return ", ".join(f"{key}:{value}" for key, value in counts.most_common()) or "-"


def _row_for_case(case: Mapping[str, str], *, cache_path: str | Path) -> Dict[str, Any]:
    sample_id = case.get("sample_id") or Path(str(case.get("audio_path", ""))).stem
    try:
        result = evaluate_mode(
            "reference",
            case["audio_path"],
            cache_path=case.get("cache_path") or cache_path,
            target_text=case.get("target_text") or None,
        )
        user_facing = render_user_facing_result(result, mode="reference")
        debug = user_facing.get("debug") if isinstance(user_facing.get("debug"), Mapping) else {}
        details = result.get("details") if isinstance(result.get("details"), Mapping) else {}
        gate = user_facing.get("debug", {}).get("reliability_gate", {}) if isinstance(user_facing.get("debug"), Mapping) else {}
        score_policy = user_facing.get("debug", {}).get("user_score_policy", {}) if isinstance(user_facing.get("debug"), Mapping) else {}
        scoring_gate = score_policy.get("scoring_gate", {}) if isinstance(score_policy, Mapping) else {}
        decisions = debug.get("special_mora_decisions") if isinstance(debug.get("special_mora_decisions"), list) else []
        user_facing_special = sum(1 for item in decisions if isinstance(item, Mapping) and item.get("user_feedback_allowed"))
        special_suppression = [
            str(item.get("suppression_reason") or "")
            for item in decisions
            if isinstance(item, Mapping) and item.get("suppression_reason")
        ]
        special_evidence = [
            str(item.get("confidence") or item.get("evidence_confidence") or "")
            for item in decisions
            if isinstance(item, Mapping)
        ]
        suppressed_reasons = [str(item) for item in user_facing.get("suppressed_reasons") or []]
        warning_codes = [str(item) for item in user_facing.get("score_policy_warnings") or []]
        pitch_allowed = bool(_pick(debug, "reliability_gate", "allow_pitch_feedback", default=False))
        pitch_reasons = [
            reason for reason in suppressed_reasons
            if reason in {"low_f0_coverage", "short_utterance", "fallback_alignment", "alignment_confidence_low", "content_mismatch", "recording_quality_bad", "pitch_not_fixed_reference"}
        ]
        user_score_policy = debug.get("user_score_policy") if isinstance(debug.get("user_score_policy"), Mapping) else {}
        scoring_gate = user_score_policy.get("scoring_gate", {}) if isinstance(user_score_policy, Mapping) else {}
        message = " / ".join(str(item) for item in user_facing.get("user_messages") or [])
        leakage_terms = _pitch_leakage_terms(message) if not pitch_allowed else []
        before_cap = user_score_policy.get("display_score_before_cap")
        after_cap = user_score_policy.get("display_score_after_cap")
        before_cap_num = _float_or_none(before_cap)
        after_cap_num = _float_or_none(after_cap)
        cap_reduction = ""
        if before_cap_num is not None and after_cap_num is not None:
            cap_reduction = round(max(0.0, before_cap_num - after_cap_num), 2)
        return {
            "sample_id": sample_id,
            "target_text": case.get("target_text") or result.get("target_text") or "",
            "audio_type": case.get("audio_type", ""),
            "reference_type": case.get("reference_type", ""),
            "expected_behavior": case.get("expected_behavior", ""),
            "recording_gate": "ok" if scoring_gate.get("recording_ok", True) else "blocked",
            "content_gate": _pick(details, "content_match", "status", default="unknown"),
            "alignment_gate": "ok" if scoring_gate.get("alignment_ok", True) else "blocked",
            "pronunciation_evidence_gate": "ok" if scoring_gate.get("pronunciation_evidence_ok", True) else "blocked",
            "score_available": scoring_gate.get("score_available", user_facing.get("display_score") is not None),
            "display_score": user_facing.get("display_score"),
            "display_score_before_cap": before_cap,
            "display_score_after_cap": after_cap,
            "display_cap_applied": user_score_policy.get("display_cap_applied", False),
            "display_cap_reason": user_score_policy.get("display_cap_reason", ""),
            "display_cap_reduction": cap_reduction,
            "pronunciation_score": user_facing.get("pronunciation_clarity_score"),
            "raw_total_score": result.get("total_score"),
            "raw_prosody_score": result.get("prosody_score"),
            "dtw_distance": _pick(debug, "alignment", "dtw_distance"),
            "alignment_confidence": debug.get("alignment_confidence"),
            "f0_coverage": debug.get("f0_voiced_coverage"),
            "pitch_feedback_allowed": pitch_allowed,
            "pitch_suppression_reason": "" if pitch_allowed else _first_nonempty(pitch_reasons or suppressed_reasons),
            "pitch_text_leakage_warning": bool(leakage_terms),
            "pitch_text_leakage_terms": ";".join(leakage_terms),
            "special_mora_user_facing_count": user_facing_special,
            "special_mora_evidence_level": _first_nonempty(special_evidence),
            "special_mora_suppression_reason": _first_nonempty(special_suppression),
            "special_mora_suppressed": bool(special_suppression) and user_facing_special == 0,
            "rhythm_timing_penalty_reason": _pick(debug, "user_score_policy", "score_caps", "short_sentence_display_cap"),
            "warning_codes": ";".join(warning_codes),
            "suppressed_reasons": ";".join(suppressed_reasons),
            "user_message_type": scoring_gate.get("user_message_type", ""),
            "user_facing_message": message,
            "status": user_facing.get("status"),
            "error": "",
        }
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "target_text": case.get("target_text", ""),
            "audio_type": case.get("audio_type", ""),
            "reference_type": case.get("reference_type", ""),
            "expected_behavior": case.get("expected_behavior", ""),
            "recording_gate": "",
            "content_gate": "",
            "alignment_gate": "",
            "pronunciation_evidence_gate": "",
            "score_available": "",
            "display_score": "",
            "display_score_before_cap": "",
            "display_score_after_cap": "",
            "display_cap_applied": "",
            "display_cap_reason": "",
            "display_cap_reduction": "",
            "pronunciation_score": "",
            "raw_total_score": "",
            "raw_prosody_score": "",
            "dtw_distance": "",
            "alignment_confidence": "",
            "f0_coverage": "",
            "pitch_feedback_allowed": "",
            "pitch_suppression_reason": "",
            "pitch_text_leakage_warning": "",
            "pitch_text_leakage_terms": "",
            "special_mora_user_facing_count": "",
            "special_mora_evidence_level": "",
            "special_mora_suppression_reason": "",
            "special_mora_suppressed": "",
            "rhythm_timing_penalty_reason": "",
            "warning_codes": "",
            "suppressed_reasons": "",
            "user_message_type": "",
            "user_facing_message": "",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_audit(cases: Iterable[Mapping[str, str]], *, cache_path: str | Path) -> List[Dict[str, Any]]:
    return [_row_for_case(case, cache_path=cache_path) for case in cases]


def _group_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("audio_type") or "unknown"),
        str(row.get("reference_type") or "unknown"),
        str(row.get("expected_behavior") or "unknown"),
    )


def _warning_counts(rows: List[Mapping[str, Any]], field: str) -> str:
    counts: Counter[str] = Counter()
    for row in rows:
        for item in str(row.get(field) or "").split(";"):
            item = item.strip()
            if item:
                counts[item] += 1
    return ", ".join(f"{key}:{value}" for key, value in counts.most_common()) or "-"


def _ids(rows: Iterable[Mapping[str, Any]]) -> str:
    data = [str(row.get("sample_id") or "") for row in rows if str(row.get("sample_id") or "")]
    return ", ".join(data) if data else "-"


def _count_by(rows: Iterable[Mapping[str, Any]], field: str) -> str:
    return _join_counts(row.get(field) for row in rows)


def write_markdown_summary(path: str | Path, rows: List[Dict[str, Any]], *, manifest_path: str | Path | None = None) -> None:
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)

    lines: List[str] = ["# Fixed-reference Scoring Audit Summary", ""]
    lines.extend([
        "## Experiment Overview",
        "",
        f"- manifest_path: `{manifest_path or ''}`",
        f"- total_samples: {len(rows)}",
        f"- expected_behavior_counts: {_count_by(rows, 'expected_behavior')}",
        f"- audio_type_counts: {_count_by(rows, 'audio_type')}",
        f"- reference_type_counts: {_count_by(rows, 'reference_type')}",
        "",
        "## Main Diagnostic Questions",
        "",
    ])

    negative_rows = [
        row for row in rows
        if str(row.get("expected_behavior") or "") in NEGATIVE_EXPECTED
        or str(row.get("audio_type") or "") in NEGATIVE_AUDIO_TYPES
    ]
    weak_rows = [
        row for row in rows
        if str(row.get("expected_behavior") or "") == "weak_reference_should_not_score"
        or "weak_reference" in str(row.get("reference_type") or "")
    ]
    fallback_rows = [
        row for row in rows
        if str(row.get("expected_behavior") or "") == "alignment_bad_should_not_score"
        or str(row.get("audio_type") or "") == "fallback_alignment_case"
        or str(row.get("alignment_gate") or "") == "blocked"
    ]
    native_rows = [row for row in rows if str(row.get("expected_behavior") or "") == "native_should_score_high"]
    bad_learner_rows = [row for row in rows if str(row.get("expected_behavior") or "") == "bad_learner_should_not_score_high"]
    negative_score = [row for row in negative_rows if _is_true(row.get("score_available")) or _float_or_none(row.get("display_score")) is not None]
    negative_pitch = [row for row in negative_rows if _is_true(row.get("pitch_feedback_allowed"))]
    weak_score = [row for row in weak_rows if _float_or_none(row.get("display_score")) is not None]
    weak_pitch = [row for row in weak_rows if _is_true(row.get("pitch_feedback_allowed"))]
    fallback_pitch = [row for row in fallback_rows if _is_true(row.get("pitch_feedback_allowed"))]
    native_rejected = [row for row in native_rows if not _is_true(row.get("score_available")) or _float_or_none(row.get("display_score")) is None]
    native_low = [row for row in native_rows if (_float_or_none(row.get("display_score")) or 100.0) < 75.0]
    bad_high = [row for row in bad_learner_rows if (_float_or_none(row.get("display_score")) or 0.0) >= 80.0]
    native_special = [row for row in native_rows if float(row.get("special_mora_user_facing_count") or 0) > 0]
    native_cap = [row for row in native_rows if (_float_or_none(row.get("display_cap_reduction")) or 0.0) >= 10.0]

    question_rows = [
        ("Are negative controls blocked?", not negative_score and not negative_pitch, _ids(negative_score + negative_pitch)),
        ("Are weak-reference samples blocked?", not weak_score and not weak_pitch, _ids(weak_score + weak_pitch)),
        ("Are fallback alignment samples blocked from pitch feedback?", not fallback_pitch, _ids(fallback_pitch)),
        ("Are native samples rejected too often?", len(native_rejected) == 0, _ids(native_rejected)),
        ("Are native scores too low?", len(native_low) == 0, _ids(native_low)),
        ("Are bad learner samples still suspiciously high?", len(bad_high) == 0, _ids(bad_high)),
        ("Is special mora warning shown for native?", len(native_special) == 0, _ids(native_special)),
        ("Is display cap frequently applied to native?", len(native_cap) == 0, _ids(native_cap)),
    ]
    for question, passed, sample_ids in question_rows:
        lines.append(f"- {question} {'OK' if passed else 'CHECK'}; affected_sample_id={sample_ids}")

    lines.extend(["", "## Group Summary", ""])
    lines.append("| audio_type | reference_type | expected | n | score_available | display mean/median/p10/p90 | pron mean/median | raw prosody mean/median | pitch allowed | pitch leakage | cap rate | avg cap reduction | fallback | content fail | pron evidence fail | special shown | special suppressed |")
    lines.append("|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, group in sorted(grouped.items()):
        display = _summary_stats(row.get("display_score") for row in group)
        pron = _summary_stats(row.get("pronunciation_score") for row in group)
        prosody = _summary_stats(row.get("raw_prosody_score") for row in group)
        cap_reductions = [_float_or_none(row.get("display_cap_reduction")) or 0.0 for row in group]
        avg_cap_reduction = round(float(statistics.mean(cap_reductions)), 2) if cap_reductions else 0.0
        n = len(group)
        lines.append(
            "| "
            + " | ".join([
                key[0],
                key[1],
                key[2],
                str(n),
                str(_rate(_is_true(row.get("score_available")) for row in group)),
                f"{display['mean']}/{display['median']}/{display['p10']}/{display['p90']}",
                f"{pron['mean']}/{pron['median']}",
                f"{prosody['mean']}/{prosody['median']}",
                str(_rate(_is_true(row.get("pitch_feedback_allowed")) for row in group)),
                str(_rate(_is_true(row.get("pitch_text_leakage_warning")) for row in group)),
                str(_rate(_is_true(row.get("display_cap_applied")) for row in group)),
                str(avg_cap_reduction),
                str(_rate(str(row.get("alignment_gate")) == "blocked" for row in group)),
                str(_rate(str(row.get("content_gate")) == "fail" for row in group)),
                str(_rate(str(row.get("pronunciation_evidence_gate")) == "blocked" for row in group)),
                str(_rate(float(row.get("special_mora_user_facing_count") or 0) > 0 for row in group)),
                str(_rate(_is_true(row.get("special_mora_suppressed")) for row in group)),
            ])
            + " |"
        )

    findings: List[str] = []
    for key, group in sorted(grouped.items()):
        expected = key[2]
        display = _summary_stats(row.get("display_score") for row in group)
        score_available_rate = _rate(_is_true(row.get("score_available")) for row in group)
        if expected == "native_should_score_high":
            if score_available_rate < 0.90:
                findings.append(f"- WARN_native_score_available_rate: group={key}, rate={score_available_rate}")
            if display["median"] != "" and float(display["median"]) < 85:
                findings.append(f"- WARN_native_median_display_low: group={key}, median={display['median']}")
            if display["p10"] != "" and float(display["p10"]) < 75:
                findings.append(f"- WARN_native_p10_display_low: group={key}, p10={display['p10']}")
        if expected == "bad_learner_should_not_score_high":
            high = [row for row in group if (_float_or_none(row.get("display_score")) or 0.0) >= 80.0]
            high_rate = _rate(row in high for row in group)
            if high_rate > 0.10:
                ids = ", ".join(str(row.get("sample_id")) for row in high)
                findings.append(f"- WARN_bad_learner_suspicious_high_rate: group={key}, rate={high_rate}, samples={ids}")
        if expected == "pitch_unverified_should_suppress_pitch":
            bad = [row for row in group if _is_true(row.get("pitch_feedback_allowed"))]
            for row in bad:
                findings.append(f"- FAIL_pitch_unverified_allowed: group={key}, sample_id={row.get('sample_id')}")

    for row in rows:
        expected = str(row.get("expected_behavior") or "")
        audio_type = str(row.get("audio_type") or "")
        display_present = _float_or_none(row.get("display_score")) is not None
        score_available = _is_true(row.get("score_available"))
        pitch_allowed = _is_true(row.get("pitch_feedback_allowed"))
        weak = expected == "weak_reference_should_not_score" or "weak_reference" in str(row.get("reference_type") or "")
        negative = expected in NEGATIVE_EXPECTED or audio_type in NEGATIVE_AUDIO_TYPES
        if negative and (score_available or display_present):
            findings.append(
                f"- FAIL_negative_user_facing_score: group={_group_key(row)}, sample_id={row.get('sample_id')}, "
                f"score_available={row.get('score_available')}, display={row.get('display_score')}"
            )
        if negative and pitch_allowed:
            findings.append(f"- FAIL_negative_pitch_allowed: group={_group_key(row)}, sample_id={row.get('sample_id')}")
        if weak and display_present:
            findings.append(f"- FAIL_weak_reference_display_score: group={_group_key(row)}, sample_id={row.get('sample_id')}, display={row.get('display_score')}")
        if weak and pitch_allowed:
            findings.append(f"- FAIL_weak_reference_pitch_allowed: group={_group_key(row)}, sample_id={row.get('sample_id')}")
        if str(row.get("alignment_gate")) == "blocked" and pitch_allowed:
            findings.append(f"- FAIL_fallback_alignment_pitch_allowed: group={_group_key(row)}, sample_id={row.get('sample_id')}")
        if _is_true(row.get("pitch_text_leakage_warning")):
            findings.append(
                f"- WARN_pitch_text_leakage: sample_id={row.get('sample_id')}, terms={row.get('pitch_text_leakage_terms')}"
            )
        if expected == "native_should_score_high" and float(row.get("special_mora_user_facing_count") or 0) > 0:
            findings.append(f"- WARN_special_mora_native_user_facing: sample_id={row.get('sample_id')}")
        low_evidence_special = str(row.get("special_mora_evidence_level") or "") in {"low", "uncertain"}
        if low_evidence_special and "special_mora" in str(row.get("display_cap_reason") or ""):
            findings.append(f"- FAIL_low_evidence_special_mora_deducted_display: sample_id={row.get('sample_id')}")

    lines.extend(["", "## Failures and Warnings", ""])
    if findings:
        lines.extend(findings)
    else:
        lines.append("- No automatic failures or warnings detected.")

    leakage = [row for row in rows if _is_true(row.get("pitch_text_leakage_warning"))]
    large_caps = [row for row in rows if (_float_or_none(row.get("display_cap_reduction")) or 0.0) >= 10.0]
    lines.extend(["", "## Pitch Text Leakage", ""])
    lines.append(f"- pitch_text_leakage_count: {len(leakage)}")
    if leakage:
        lines.append("- affected_sample_id: " + ", ".join(str(row.get("sample_id")) for row in leakage))

    lines.extend(["", "## Display Cap Reductions", ""])
    lines.append(f"- display_cap_applied_rate: {_rate(_is_true(row.get('display_cap_applied')) for row in rows)}")
    reductions = [_float_or_none(row.get("display_cap_reduction")) or 0.0 for row in rows]
    lines.append(f"- average_cap_reduction: {round(float(statistics.mean(reductions)), 2) if reductions else 0.0}")
    if large_caps:
        lines.append("- reduction_gte_10: " + ", ".join(
            f"{row.get('sample_id')}({row.get('display_cap_reduction')})" for row in large_caps
        ))

    lines.extend(["", "## Suspicious Sample List", ""])
    suspicious_lists = [
        ("negative_controls_with_display_score", negative_score),
        ("negative_controls_with_pitch_feedback_allowed", negative_pitch),
        ("weak_reference_with_display_score", weak_score),
        ("fallback_with_pitch_feedback_allowed", fallback_pitch),
        ("bad_learner_display_score_gte_80", bad_high),
        ("native_display_score_lt_75", native_low),
        ("native_rejected_by_gate", native_rejected),
        ("native_with_special_mora_user_facing_warning", native_special),
        ("native_with_large_display_cap_reduction", native_cap),
    ]
    for label, data in suspicious_lists:
        lines.append(f"- {label}: {_ids(data)}")

    lines.extend(["", "## Decision Hints", ""])
    lines.append("- If negative controls still show scores or pitch feedback, fix gates or message leakage first.")
    lines.append("- If native samples are often rejected, check reference audio, alignment, VAD, and sampling rate before changing score mapping.")
    lines.append("- If native score availability is acceptable but display scores are low, inspect score mapping and display cap behavior next.")
    lines.append("- If bad learner samples remain 80+, calibrate pronunciation_score in the next round after gate behavior is confirmed.")
    lines.append("- This audit does not auto-tune thresholds.")

    lines.extend(["", "## Warning Code Counts", ""])
    lines.append(f"- score_policy_warnings: {_warning_counts(rows, 'warning_codes')}")
    lines.append(f"- suppressed_reasons: {_warning_counts(rows, 'suppressed_reasons')}")
    lines.append(f"- user_message_type: {_warning_counts(rows, 'user_message_type')}")
    lines.append(f"- special_mora_evidence_level: {_join_counts(row.get('special_mora_evidence_level') for row in rows)}")
    lines.append(f"- special_mora_suppression_reason: {_join_counts(row.get('special_mora_suppression_reason') for row in rows)}")
    lines.append(f"- rhythm_timing_penalty_reason: {_join_counts(row.get('rhythm_timing_penalty_reason') for row in rows)}")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit product-safe fixed-reference scoring gates.")
    parser.add_argument("--manifest", required=True, help="CSV with sample_id,audio_path,target_text,audio_type,reference_type.")
    parser.add_argument("--cache", default="cache/ramen_kudasai", help="Default sentence cache prefix.")
    parser.add_argument("--out", default="outputs/fixed_reference_scoring_audit.csv")
    parser.add_argument("--summary-out", default="reports/fixed_reference_scoring_audit_summary.md")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = _read_manifest(args.manifest)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("Manifest is empty. Add at least one row with audio_path.")
    rows = run_audit(cases, cache_path=args.cache)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")
    write_markdown_summary(args.summary_out, rows, manifest_path=args.manifest)
    print(f"Wrote {args.summary_out}")
    false_high = [
        row for row in rows
        if str(row.get("audio_type")) in {"wrong_target", "nonsense", "english", "bad_recording"}
        and str(row.get("display_score") or "").isdigit()
        and int(row["display_score"]) >= 80
    ]
    if false_high:
        print(f"WARNING: {len(false_high)} suspicious high user-facing scores found.")
    else:
        print("No suspicious 80+ user-facing scores for marked bad/wrong samples.")


if __name__ == "__main__":
    main()
