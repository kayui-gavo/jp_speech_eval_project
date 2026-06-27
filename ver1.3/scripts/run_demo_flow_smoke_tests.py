#!/usr/bin/env python
from __future__ import annotations

import csv
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from demo_user_facing_policy_examples import _base_result, _with
from jp_speech_eval.feedback_renderer import render_user_facing_result


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_LEARNER_WORDS = (
    "間違っています",
    "発音できていません",
    "ネイティブ度",
    "完全な発音正確度",
    "発音は不正確",
)


def _clear_short_long_vowel() -> Dict[str, Any]:
    return _with(_base_result(), mora_table=[
        {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2},
        {"mora": "ー", "start_sec": 0.2, "end_sec": 0.22},
        {"mora": "メ", "start_sec": 0.22, "end_sec": 0.42},
        {"mora": "ン", "start_sec": 0.42, "end_sec": 0.62},
        {"mora": "ヲ", "start_sec": 0.62, "end_sec": 0.82},
        {"mora": "ク", "start_sec": 0.82, "end_sec": 1.02},
        {"mora": "ダ", "start_sec": 1.02, "end_sec": 1.22},
        {"mora": "サ", "start_sec": 1.22, "end_sec": 1.42},
        {"mora": "イ", "start_sec": 1.42, "end_sec": 1.62},
    ])


def _near_boundary_long_vowel() -> Dict[str, Any]:
    return _with(_base_result(), mora_table=[
        {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2},
        {"mora": "ー", "start_sec": 0.2, "end_sec": 0.245},
        {"mora": "メ", "start_sec": 0.245, "end_sec": 0.445},
        {"mora": "ン", "start_sec": 0.445, "end_sec": 0.645},
        {"mora": "ヲ", "start_sec": 0.645, "end_sec": 0.845},
        {"mora": "ク", "start_sec": 0.845, "end_sec": 1.045},
        {"mora": "ダ", "start_sec": 1.045, "end_sec": 1.245},
        {"mora": "サ", "start_sec": 1.245, "end_sec": 1.445},
        {"mora": "イ", "start_sec": 1.445, "end_sec": 1.645},
    ])


def _learner_view(rendered: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": rendered.get("status"),
        "practice_score": rendered.get("practice_score"),
        "confidence": rendered.get("confidence"),
        "summary_text": rendered.get("summary_text"),
        "primary_suggestion_text": rendered.get("primary_suggestion_text"),
        "suggestion_type": rendered.get("suggestion_type"),
        "mode_notice": rendered.get("mode_notice"),
    }


def _flatten_values(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _flatten_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _flatten_values(child)
    elif value is not None:
        yield str(value)


def _has_forbidden_copy(rendered: Dict[str, Any]) -> bool:
    text = "\n".join(_flatten_values(_learner_view(rendered)))
    return any(word in text for word in FORBIDDEN_LEARNER_WORDS)


def _raw_score_leakage(rendered: Dict[str, Any]) -> bool:
    learner = _learner_view(rendered)
    text = "\n".join(learner.keys()) + "\n" + "\n".join(_flatten_values(learner))
    return any(token in text for token in ("debug_total_score", "prosody_score", "DTW", "threshold_low"))


def _special_mora_user_feedback(rendered: Dict[str, Any]) -> bool:
    return any(
        bool(item.get("user_feedback_allowed"))
        for item in rendered.get("debug", {}).get("special_mora_decisions", [])
    )


def _scenario_rows() -> List[Dict[str, Any]]:
    base = _base_result()
    fallback = _with(
        base,
        rhythm_score=72,
        weak_pronunciation_naturalness_score=78,
        weak_rhythm_naturalness_score=72,
        weak_prosody_naturalness_score=76,
        weak_overall_practice_score=79,
        alignment_mode="cached_dtw_fallback_equal",
        details={
            "alignment": {"mode": "cached_dtw_fallback_equal"},
            "reliability": {"level": "medium", "overall": 0.70, "alignment": 0.50, "f0_coverage": 0.80},
            "weak_reference_native_likeness": {
                "weak_pronunciation_naturalness_score": 78,
                "weak_rhythm_naturalness_score": 72,
                "weak_prosody_naturalness_score": 76,
                "weak_overall_practice_score": 79,
                "weak_overall_guardrail": {"status": "ok", "display_allowed": True},
            },
        },
    )
    clear_short = _clear_short_long_vowel()
    near = _near_boundary_long_vowel()
    rows = [
        {"name": "fixed_normal_pass", "result": base, "kwargs": {}, "expect": {"status": "pass"}},
        {"name": "fixed_fallback_degrades_to_four_practice_scores", "result": fallback, "kwargs": {}, "expect": {"status": "practice_suggestion", "degraded_four_dimensions": True, "no_special_feedback": True}},
        {"name": "fixed_poor_recording_retry", "result": _with(base, details={"recording_quality": {"score": 0.1}}), "kwargs": {}, "expect": {"status": "retry"}},
        {"name": "fixed_near_boundary_special_mora_accepted", "result": near, "kwargs": {"special_mora_threshold_profile": "v2_limited_candidate", "enable_user_facing_calibrated_special_mora": True}, "expect": {"no_special_feedback": True}},
        {"name": "fixed_clear_long_vowel_default_safe", "result": clear_short, "kwargs": {}, "expect": {"status": "pass", "no_special_feedback": True}},
        {"name": "fixed_clear_long_vowel_explicit_gentle", "result": clear_short, "kwargs": {"special_mora_threshold_profile": "v2_limited_candidate", "enable_user_facing_calibrated_special_mora": True}, "expect": {"status": "practice_suggestion", "suggestion_type": "special_mora"}},
        {"name": "weak_asr_unconfirmed", "result": _with(base, details={"mode": "asr_pseudo_reference", "weak_reference": True}), "kwargs": {"mode": "asr_pseudo_reference"}, "expect": {"status": "debug_only", "practice_score_none": True}},
        {"name": "weak_asr_confirmed", "result": _with(base, details={"mode": "asr_confirmed_weak_reference", "weak_reference": True}), "kwargs": {"mode": "asr_confirmed_weak_reference"}, "expect": {"weak_reference": True}},
        {"name": "weak_special_mora_suppressed", "result": _with(clear_short, details={"mode": "asr_confirmed_weak_reference", "weak_reference": True}), "kwargs": {"mode": "asr_confirmed_weak_reference", "special_mora_threshold_profile": "v2_limited_candidate", "enable_user_facing_calibrated_special_mora": True}, "expect": {"no_special_feedback": True}},
        {"name": "kanade_reference_mocked", "result": _with(base, details={"mode": "kanade_asr_voice_reference", "demo_only": True, "playback_only": True, "exclude_from_pronunciation_score": True}), "kwargs": {"mode": "kanade_asr_voice_reference"}, "expect": {"status": "debug_only", "practice_score_none": True}},
        {"name": "kanade_excluded_from_scoring", "result": _with(base, details={"mode": "kanade_asr_voice_reference", "demo_only": True, "playback_only": True, "exclude_from_pronunciation_score": True}), "kwargs": {"mode": "kanade_asr_voice_reference", "special_mora_threshold_profile": "v2_limited_candidate", "enable_user_facing_calibrated_special_mora": True}, "expect": {"kanade_excluded": True}},
        {"name": "kanade_notice_visible", "result": _with(base, details={"mode": "kanade_asr_voice_reference", "demo_only": True, "playback_only": True, "exclude_from_pronunciation_score": True}), "kwargs": {"mode": "kanade_asr_voice_reference"}, "expect": {"kanade_notice": True}},
    ]
    return rows


def _evaluate_expectations(rendered: Dict[str, Any], expect: Dict[str, Any]) -> tuple[bool, List[str]]:
    notes: List[str] = []
    ok = True
    if _raw_score_leakage(rendered):
        ok = False
        notes.append("raw_score_leakage")
    if _has_forbidden_copy(rendered):
        ok = False
        notes.append("forbidden_copy")
    if expect.get("status") and rendered.get("status") != expect["status"]:
        ok = False
        notes.append(f"status={rendered.get('status')}")
    if expect.get("suggestion_type") and rendered.get("suggestion_type") != expect["suggestion_type"]:
        ok = False
        notes.append(f"suggestion_type={rendered.get('suggestion_type')}")
    if expect.get("practice_score_none") and rendered.get("practice_score", {}).get("value") is not None:
        ok = False
        notes.append("practice_score_should_be_none")
    if expect.get("no_special_feedback") and _special_mora_user_feedback(rendered):
        ok = False
        notes.append("special_mora_feedback_leaked")
    if expect.get("weak_reference") and not rendered.get("debug", {}).get("weak_reference"):
        ok = False
        notes.append("weak_reference_missing")
    if expect.get("degraded_four_dimensions"):
        dimensions = rendered.get("dimension_scores") or {}
        confidence = rendered.get("dimension_confidence") or {}
        if not rendered.get("debug", {}).get("degraded_reference_practice"):
            ok = False
            notes.append("degraded_reference_practice_missing")
        if any(dimensions.get(name) is None for name in ("pronunciation", "rhythm", "fluency", "pitch")):
            ok = False
            notes.append("degraded_dimension_missing")
        if any(confidence.get(name) != "low" for name in ("pronunciation", "rhythm", "fluency", "pitch")):
            ok = False
            notes.append("degraded_confidence_not_low")
        if rendered.get("debug", {}).get("visible_prosody_score") == rendered.get("debug", {}).get("prosody_score"):
            ok = False
            notes.append("strict_pitch_score_leaked")
    policy = rendered.get("debug", {}).get("scoring_policy", {})
    if expect.get("kanade_excluded") and not policy.get("exclude_from_pronunciation_score"):
        ok = False
        notes.append("kanade_not_excluded")
    if expect.get("kanade_notice") and "声" not in str(rendered.get("mode_notice", "")):
        ok = False
        notes.append("kanade_notice_missing")
    return ok, notes


def run() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for scenario in _scenario_rows():
        result = deepcopy(scenario["result"])
        rendered = render_user_facing_result(result, **scenario["kwargs"])
        passed, notes = _evaluate_expectations(rendered, scenario["expect"])
        out.append({
            "scenario": scenario["name"],
            "mode": rendered.get("mode"),
            "status": rendered.get("status"),
            "practice_score_value": rendered.get("practice_score", {}).get("value"),
            "practice_score_label": rendered.get("practice_score", {}).get("label"),
            "suggestion_type": rendered.get("suggestion_type"),
            "raw_score_leakage": _raw_score_leakage(rendered),
            "kanade_scoring_leakage": (
                "kanade" in str(rendered.get("mode", ""))
                and not rendered.get("debug", {}).get("scoring_policy", {}).get("exclude_from_pronunciation_score")
            ),
            "strong_special_mora_correction": _has_forbidden_copy(rendered),
            "weak_reference_conservative": not (
                rendered.get("debug", {}).get("weak_reference")
                and _special_mora_user_feedback(rendered)
            ),
            "passed": passed,
            "notes": ";".join(notes),
            "summary_text": rendered.get("summary_text"),
            "primary_suggestion_text": rendered.get("primary_suggestion_text"),
            "mode_notice": rendered.get("mode_notice"),
        })
    return out


def write_outputs(rows: List[Dict[str, Any]]) -> None:
    results_dir = ROOT / "results"
    reports_dir = ROOT / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "demo_flow_smoke_tests.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    pass_count = sum(1 for row in rows if row["passed"])
    lines = [
        "# Demo flow smoke test report",
        "",
        f"- scenarios: {len(rows)}",
        f"- passed: {pass_count}",
        f"- failed: {len(rows) - pass_count}",
        "",
        "## Guardrails",
        "",
        f"- no raw score leakage to learner fields: {all(not row['raw_score_leakage'] for row in rows)}",
        f"- no Kanade scoring leakage: {all(not row['kanade_scoring_leakage'] for row in rows)}",
        f"- no strong special mora correction by default: {all(not row['strong_special_mora_correction'] for row in rows)}",
        f"- weak-reference remains conservative: {all(row['weak_reference_conservative'] for row in rows)}",
        "- fixed-reference is the most reliable path: documented in `reports/fixed_reference_demo_flow.md`",
        "",
        "## Scenarios",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['scenario']}: {'PASS' if row['passed'] else 'FAIL'} ({row['status']}) {row['notes']}".rstrip())
    (reports_dir / "demo_flow_smoke_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = run()
    write_outputs(rows)
    print({"scenarios": len(rows), "passed": sum(1 for row in rows if row["passed"]), "report": str(ROOT / "reports" / "demo_flow_smoke_test_report.md")})
    if not all(row["passed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
