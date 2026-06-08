from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

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
    "pronunciation_score",
    "raw_total_score",
    "raw_prosody_score",
    "dtw_distance",
    "alignment_confidence",
    "f0_coverage",
    "pitch_feedback_allowed",
    "pitch_suppression_reason",
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
            "pronunciation_score": user_facing.get("pronunciation_clarity_score"),
            "raw_total_score": result.get("total_score"),
            "raw_prosody_score": result.get("prosody_score"),
            "dtw_distance": _pick(debug, "alignment", "dtw_distance"),
            "alignment_confidence": debug.get("alignment_confidence"),
            "f0_coverage": debug.get("f0_voiced_coverage"),
            "pitch_feedback_allowed": pitch_allowed,
            "pitch_suppression_reason": "" if pitch_allowed else _first_nonempty(pitch_reasons or suppressed_reasons),
            "special_mora_user_facing_count": user_facing_special,
            "special_mora_evidence_level": _first_nonempty(special_evidence),
            "special_mora_suppression_reason": _first_nonempty(special_suppression),
            "special_mora_suppressed": bool(special_suppression) and user_facing_special == 0,
            "rhythm_timing_penalty_reason": _pick(debug, "user_score_policy", "score_caps", "short_sentence_display_cap"),
            "warning_codes": ";".join(warning_codes),
            "suppressed_reasons": ";".join(suppressed_reasons),
            "user_message_type": scoring_gate.get("user_message_type", ""),
            "user_facing_message": " / ".join(str(item) for item in user_facing.get("user_messages") or []),
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
            "pronunciation_score": "",
            "raw_total_score": "",
            "raw_prosody_score": "",
            "dtw_distance": "",
            "alignment_confidence": "",
            "f0_coverage": "",
            "pitch_feedback_allowed": "",
            "pitch_suppression_reason": "",
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


def write_markdown_summary(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)

    lines: List[str] = ["# Fixed-reference Scoring Audit Summary", ""]
    lines.append("| audio_type | reference_type | expected | n | score_available | display mean/median/p10/p90 | pron mean/median | raw prosody mean/median | pitch allowed | pitch suppressed | fallback | low align | content fail | recording fail | pron evidence fail | special shown | special suppressed |")
    lines.append("|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, group in sorted(grouped.items()):
        display = _summary_stats(row.get("display_score") for row in group)
        pron = _summary_stats(row.get("pronunciation_score") for row in group)
        prosody = _summary_stats(row.get("raw_prosody_score") for row in group)
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
                str(_rate(not _is_true(row.get("pitch_feedback_allowed")) for row in group)),
                str(_rate(str(row.get("alignment_gate")) == "blocked" for row in group)),
                str(_rate("low_alignment" in str(row.get("warning_codes")) for row in group)),
                str(_rate(str(row.get("content_gate")) == "fail" for row in group)),
                str(_rate(str(row.get("recording_gate")) == "blocked" for row in group)),
                str(_rate(str(row.get("pronunciation_evidence_gate")) == "blocked" for row in group)),
                str(_rate(float(row.get("special_mora_user_facing_count") or 0) > 0 for row in group)),
                str(_rate(_is_true(row.get("special_mora_suppressed")) for row in group)),
            ])
            + " |"
        )

    negative = {"wrong_target", "nonsense", "english", "bad_recording"}
    failures = [
        row for row in rows
        if str(row.get("audio_type")) in negative
        and (_is_true(row.get("pitch_feedback_allowed")) or _float_or_none(row.get("display_score")) is not None)
    ]
    suspicious = [
        row for row in rows
        if str(row.get("audio_type")) in {"bad_learner", "clear_bad", "learner_bad"}
        and (_float_or_none(row.get("display_score")) or 0.0) >= 80.0
    ]
    lines.extend(["", "## Failure Checks", ""])
    if failures:
        lines.append("### Negative controls with user-facing score or pitch")
        for row in failures:
            lines.append(f"- {row.get('sample_id')}: display={row.get('display_score')} pitch={row.get('pitch_feedback_allowed')} warnings={row.get('warning_codes')}")
    else:
        lines.append("- Negative controls: pass. No user-facing score/pitch feedback found.")
    if suspicious:
        lines.append("### Suspicious high scores")
        for row in suspicious:
            lines.append(f"- {row.get('sample_id')}: display={row.get('display_score')} pron={row.get('pronunciation_score')} warnings={row.get('warning_codes')}")
    else:
        lines.append("- Suspicious high scores: none found for marked bad learner groups.")

    lines.extend(["", "## Warning Code Counts", ""])
    lines.append(f"- score_policy_warnings: {_warning_counts(rows, 'warning_codes')}")
    lines.append(f"- suppressed_reasons: {_warning_counts(rows, 'suppressed_reasons')}")
    lines.append(f"- user_message_type: {_warning_counts(rows, 'user_message_type')}")

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
    write_markdown_summary(args.summary_out, rows)
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
