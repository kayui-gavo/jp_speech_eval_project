from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from jp_speech_eval.eval_modes import evaluate_mode
from jp_speech_eval.feedback_renderer import render_user_facing_result


FIELDNAMES = [
    "sample_id",
    "target_text",
    "audio_type",
    "reference_type",
    "recording_gate",
    "content_gate",
    "alignment_gate",
    "score_available",
    "display_score",
    "pronunciation_score",
    "raw_total_score",
    "raw_prosody_score",
    "dtw_distance",
    "alignment_confidence",
    "f0_coverage",
    "pitch_feedback_allowed",
    "special_mora_user_facing_count",
    "warning_codes",
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
        return {
            "sample_id": sample_id,
            "target_text": case.get("target_text") or result.get("target_text") or "",
            "audio_type": case.get("audio_type", ""),
            "reference_type": case.get("reference_type", ""),
            "recording_gate": "ok" if scoring_gate.get("recording_ok", True) else "blocked",
            "content_gate": _pick(details, "content_match", "status", default="unknown"),
            "alignment_gate": "ok" if scoring_gate.get("alignment_ok", True) else "blocked",
            "score_available": scoring_gate.get("score_available", user_facing.get("display_score") is not None),
            "display_score": user_facing.get("display_score"),
            "pronunciation_score": user_facing.get("pronunciation_clarity_score"),
            "raw_total_score": result.get("total_score"),
            "raw_prosody_score": result.get("prosody_score"),
            "dtw_distance": _pick(debug, "alignment", "dtw_distance"),
            "alignment_confidence": debug.get("alignment_confidence"),
            "f0_coverage": debug.get("f0_voiced_coverage"),
            "pitch_feedback_allowed": _pick(debug, "reliability_gate", "allow_pitch_feedback"),
            "special_mora_user_facing_count": user_facing_special,
            "warning_codes": ";".join(str(item) for item in user_facing.get("score_policy_warnings") or []),
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
            "recording_gate": "",
            "content_gate": "",
            "alignment_gate": "",
            "score_available": "",
            "display_score": "",
            "pronunciation_score": "",
            "raw_total_score": "",
            "raw_prosody_score": "",
            "dtw_distance": "",
            "alignment_confidence": "",
            "f0_coverage": "",
            "pitch_feedback_allowed": "",
            "special_mora_user_facing_count": "",
            "warning_codes": "",
            "user_facing_message": "",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_audit(cases: Iterable[Mapping[str, str]], *, cache_path: str | Path) -> List[Dict[str, Any]]:
    return [_row_for_case(case, cache_path=cache_path) for case in cases]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit product-safe fixed-reference scoring gates.")
    parser.add_argument("--manifest", required=True, help="CSV with sample_id,audio_path,target_text,audio_type,reference_type.")
    parser.add_argument("--cache", default="cache/ramen_kudasai", help="Default sentence cache prefix.")
    parser.add_argument("--out", default="outputs/fixed_reference_scoring_audit.csv")
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
