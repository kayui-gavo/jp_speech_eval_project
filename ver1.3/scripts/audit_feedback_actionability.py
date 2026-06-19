#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.feedback_renderer import render_user_facing_result


def _base_result(*, weak: bool = True) -> Dict[str, Any]:
    mode = "asr_confirmed_weak_reference" if weak else "reference_based"
    pitch_source = "openjtalk_accent_phrase_chain" if weak else "reference_audio_f0_cache"
    reliability = "heuristic" if weak else "reliable"
    return {
        "target_text": "ラーメンをください",
        "kana": "ラーメンヲクダサイ",
        "moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"],
        "mora_table": [
            {"mora": "ラ", "start_sec": 0.00, "end_sec": 0.20},
            {"mora": "ー", "start_sec": 0.20, "end_sec": 0.40},
            {"mora": "メ", "start_sec": 0.40, "end_sec": 0.60},
            {"mora": "ン", "start_sec": 0.60, "end_sec": 0.80},
            {"mora": "ヲ", "start_sec": 0.80, "end_sec": 1.00},
            {"mora": "ク", "start_sec": 1.00, "end_sec": 1.20},
            {"mora": "ダ", "start_sec": 1.20, "end_sec": 1.40},
            {"mora": "サ", "start_sec": 1.40, "end_sec": 1.60},
            {"mora": "イ", "start_sec": 1.60, "end_sec": 1.80},
        ],
        "duration_sec": 1.8,
        "total_score": 88,
        "pronunciation_score": 88,
        "prosody_score": 72,
        "fluency_score": 90,
        "tone_score": 99,
        "feedback": [],
        "alignment_mode": "cached_dtw",
        "pause_info": {"pause_count": 0, "pause_ratio": 0.05},
        "weak_prosody_naturalness_score": 90 if weak else None,
        "weak_overall_practice_score": 89 if weak else None,
        "score_type": "weak_reference_native_likeness" if weak else "strict_reference",
        "strict_reference_available": not weak,
        "details": {
            "mode": mode,
            "weak_reference": weak,
            "pitch_target_source": pitch_source,
            "pitch_target_reliability": reliability,
            "verified_level": None if weak else "human_checked",
            "reliability": {"level": "high", "overall": 0.92, "alignment": 0.9, "f0_coverage": 0.9},
            "recording_quality": {"score": 0.92},
            "content_match": {"status": "pass"},
            "alignment": {"mode": "cached_dtw"},
            "pronunciation": {"mora_duration_cv": 0.12},
            "fluency": {
                "speech_rate_mora_per_sec": 5.0,
                "delivery_fluency_components": {"long_pause_count": 0, "pause_ratio": 0.05},
                "rhythm_timing_score": 90,
                "delivery_fluency_score": 90,
            },
            "prosody": {"pitch_target_source": pitch_source, "pitch_target_reliability": reliability},
            "weak_reference_native_likeness": {
                "available": True,
                "weak_prosody_naturalness_score": 90,
                "weak_overall_practice_score": 89,
                "f0_coverage": 0.9,
                "flatness_penalty": 0.0,
                "instability_penalty": 0.0,
                "transition_smoothness": 0.9,
                "component_scores": {"range": 0.9, "movement": 0.9},
                "weak_overall_guardrail": {"status": "ok", "reasons": [], "display_allowed": True},
            },
            "mora_evidence": [
                {
                    "judgement_available": True,
                    "boundary_confidence": 0.95,
                    "energy_coverage": 0.9,
                    "mapping_success": True,
                    "start_sec": i * 0.2,
                    "end_sec": (i + 1) * 0.2,
                }
                for i in range(9)
            ],
        },
    }


def _merge_details(result: Dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result["details"].get(key), Mapping):
            result["details"][key] = {**result["details"][key], **value}
        else:
            result["details"][key] = value


def _fixture(case_id: str) -> Dict[str, Any]:
    result = _base_result(weak=case_id not in {"special_mora_long_vowel_short", "special_mora_nasal_short"})
    if case_id == "too_short_japanese":
        result.update({"target_text": "はい", "kana": "ハイ", "moras": ["ハ", "イ"], "mora_table": result["mora_table"][:2], "duration_sec": 0.4})
        _merge_details(result, {"weak_reference_native_likeness": {
            "weak_overall_practice_score": None,
            "weak_overall_guardrail": {"status": "no_score", "reasons": ["short_utterance_insufficient_evidence"], "display_allowed": False},
        }})
        result["weak_overall_practice_score"] = None
    elif case_id == "random_english_latin":
        _merge_details(result, {"content_match": {
            "status": "pass", "asr_provider": "whisper", "transcript": "please give me ramen",
            "transcript_kana": "", "target_kana": "ラーメンヲクダサイ", "kana_similarity": 0.0,
        }})
    elif case_id == "flat_pitch":
        _merge_details(result, {"weak_reference_native_likeness": {
            "flatness_penalty": 0.35, "component_scores": {"range": 0.2, "movement": 0.1},
        }})
    elif case_id == "random_pitch":
        _merge_details(result, {"weak_reference_native_likeness": {
            "instability_penalty": 0.25, "transition_smoothness": 0.35,
        }})
    elif case_id == "low_f0_coverage":
        _merge_details(result, {
            "reliability": {"f0_coverage": 0.2},
            "weak_reference_native_likeness": {"available": False, "f0_coverage": 0.2},
        })
    elif case_id == "long_pause_many":
        result["pause_info"] = {"pause_count": 6, "pause_ratio": 0.35}
        _merge_details(result, {"fluency": {"delivery_fluency_components": {"long_pause_count": 6, "pause_ratio": 0.35}}})
    elif case_id == "fast_rate":
        _merge_details(result, {"fluency": {"speech_rate_mora_per_sec": 7.2}})
    elif case_id == "special_mora_long_vowel_short":
        result["mora_table"][1]["end_sec"] = 0.23
        for index in range(2, len(result["mora_table"])):
            result["mora_table"][index]["start_sec"] -= 0.17
            result["mora_table"][index]["end_sec"] -= 0.17
    elif case_id == "special_mora_nasal_short":
        result["mora_table"][3]["end_sec"] = 0.63
        for index in range(4, len(result["mora_table"])):
            result["mora_table"][index]["start_sec"] -= 0.17
            result["mora_table"][index]["end_sec"] -= 0.17
    elif case_id == "alignment_fallback":
        result["alignment_mode"] = "cached_dtw_fallback_equal"
        _merge_details(result, {"alignment": {"mode": "cached_dtw_fallback_equal"}, "reliability": {"overall": 0.7, "alignment": 0.55}})
    return result


def audit_rows(plan_path: Path) -> list[dict[str, Any]]:
    with plan_path.open(encoding="utf-8") as stream:
        plan = list(csv.DictReader(stream))
    rows: list[dict[str, Any]] = []
    forbidden = ("高低重音错", "重音核错", "标准音调不一致", "accent drop incorrect", "prosody contour")
    for planned in plan:
        case_id = planned["case_id"]
        rendered = render_user_facing_result(_fixture(case_id), mode=None, special_mora_threshold_profile="v2_limited_candidate")
        candidates = rendered.get("feedback_candidates") or []
        focus = candidates[0] if candidates else {}
        actual_type = str(focus.get("evidence_type") or "none")
        location = focus.get("location")
        tip = str(focus.get("practice_tip") or "")
        caveat = str(focus.get("caveat") or "")
        visible_text = " ".join(rendered.get("user_messages") or []) + " " + str(rendered.get("primary_suggestion_text") or "")
        overclaim = any(term in visible_text for term in forbidden)
        expected_location = planned["expected_location"].lower() == "true"
        expected_caveat = planned["expected_caveat"].lower() == "true"
        passed = (
            actual_type == planned["expected_feedback_type"]
            and bool(location) == expected_location
            and bool(tip)
            and (bool(caveat) if expected_caveat else True)
            and not overclaim
            and len(rendered.get("user_messages") or []) <= 2
        )
        rows.append({
            **planned,
            "actual_feedback_type": actual_type,
            "dimension": focus.get("dimension"),
            "severity": focus.get("severity"),
            "confidence": focus.get("confidence"),
            "has_location": bool(location),
            "location": location,
            "user_message": focus.get("user_message"),
            "practice_tip": tip,
            "has_practice_tip": bool(tip),
            "caveat": caveat,
            "has_caveat_if_needed": bool(caveat) if expected_caveat else True,
            "overclaim_risk": overclaim,
            "visible_message_count": len(rendered.get("user_messages") or []),
            "display_score": rendered.get("display_score"),
            "pass_or_fail": "PASS" if passed else "FAIL",
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    passed = sum(row["pass_or_fail"] == "PASS" for row in rows)
    lines = [
        "# Feedback actionability audit",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- cases: {len(rows)}",
        f"- passed: {passed}/{len(rows)}",
        "- scope: feedback selection and wording only; scoring, aggregate, calibration, and providers are unchanged.",
        "- smoke inputs are policy/component fixtures, not a new real-audio corpus.",
        "",
        "## Existing feedback audit",
        "",
        "- Previous rendering selected legacy strings without a common evidence/location/practice schema.",
        "- Fluency rate/pause and high-confidence long-vowel/nasal timing have usable evidence.",
        "- Raw pitch-reference strings could be too strong for weak-reference arbitrary speech if surfaced directly.",
        "- Generic advice such as 'try again' did not always explain the evidence or the next practice action.",
        "- Content/no-score and fallback gates already existed, but structured feedback now makes their suppression explicit.",
        "- Sokuon/yoon and weak special-mora details remain gated; no unsupported location is invented.",
        "",
        "## Smoke results",
        "",
        "| case | expected | actual | location | tip | caveat | overclaim | result |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['expected_feedback_type']} | {row['actual_feedback_type']} | "
            f"{row['has_location']} | {row['has_practice_tip']} | {row['has_caveat_if_needed']} | "
            f"{row['overclaim_risk']} | {row['pass_or_fail']} |"
        )
    lines.extend([
        "",
        "## Remaining maturity limits",
        "",
        "- Feedback wording is evidence-aware, but learner usefulness still needs listening tests with Chinese-L1 learners.",
        "- Weak pitch can describe flatness or instability only; wrong accent drop remains insufficiently separated.",
        "- Pause aggregates do not provide a reliable phrase-level time location, so no pause location is claimed.",
        "- Special-mora location is shown only when the runtime decision is user-facing allowed and alignment evidence is strong.",
        "- Normal positive feedback remains a practice observation, not proof of phoneme correctness.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit feedback actionability policy fixtures.")
    parser.add_argument("--plan", default="data/feedback_smoke_set_plan.csv")
    parser.add_argument("--out-csv", default="results/calibration/feedback_actionability_audit.csv")
    parser.add_argument("--out-report", default="reports/feedback_actionability_audit.md")
    args = parser.parse_args()
    rows = audit_rows(ROOT / args.plan)
    write_csv(ROOT / args.out_csv, rows)
    write_report(ROOT / args.out_report, rows)
    print(f"wrote {ROOT / args.out_csv}")
    print(f"wrote {ROOT / args.out_report}")


if __name__ == "__main__":
    main()
