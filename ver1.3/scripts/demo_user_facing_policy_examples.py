#!/usr/bin/env python
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from jp_speech_eval.feedback_renderer import render_user_facing_result


ROOT = Path(__file__).resolve().parents[1]


def _base_result() -> Dict[str, Any]:
    return {
        "target_text": "ラーメンをください",
        "kana": "ラーメンヲクダサイ",
        "moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"],
        "mora_table": [
            {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.2},
            {"mora": "ー", "start_sec": 0.2, "end_sec": 0.38},
            {"mora": "メ", "start_sec": 0.38, "end_sec": 0.58},
            {"mora": "ン", "start_sec": 0.58, "end_sec": 0.78},
            {"mora": "ヲ", "start_sec": 0.78, "end_sec": 0.98},
            {"mora": "ク", "start_sec": 0.98, "end_sec": 1.18},
            {"mora": "ダ", "start_sec": 1.18, "end_sec": 1.38},
            {"mora": "サ", "start_sec": 1.38, "end_sec": 1.58},
            {"mora": "イ", "start_sec": 1.58, "end_sec": 1.78},
        ],
        "total_score": 88,
        "pronunciation_score": 90,
        "prosody_score": 65,
        "fluency_score": 92,
        "tone_score": 70,
        "feedback": ["今回の練習は大きな問題なく確認できました。"],
        "alignment_mode": "cached_dtw",
        "details": {
            "mode": "reference_based",
            "verified_level": "human_checked",
            "pitch_target_source": "human_checked",
            "reliability": {"level": "high", "overall": 0.95, "alignment": 0.9, "f0_coverage": 0.9},
            "recording_quality": {"score": 0.95},
            "content_match": {"status": "pass"},
            "alignment": {"mode": "cached_dtw"},
            "pronunciation": {"mora_duration_cv": 0.1, "special_mora_penalty": 0, "special_mora_diagnostics": []},
            "prosody": {"contour_corr": 0.8, "transition_agreement": 0.8, "final_intonation_score": 85},
            "fluency": {"rhythm_timing_score": 92, "delivery_fluency_score": 94},
            "mora_evidence": [
                {"judgement_available": True, "boundary_confidence": 0.9, "energy_coverage": 0.9}
                for _ in range(9)
            ],
        },
    }


def _with(base: Dict[str, Any], **updates: Any) -> Dict[str, Any]:
    out = deepcopy(base)
    details = updates.pop("details", None)
    for key, value in updates.items():
        out[key] = value
    if details:
        out["details"].update(details)
    return out


def build_examples() -> List[Dict[str, Any]]:
    base = _base_result()
    near = _with(base, mora_table=[
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
    clear_short = _with(base, mora_table=[
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
    return [
        {"name": "fixed-reference normal pass", "result": base, "kwargs": {}},
        {"name": "near-boundary long vowel accepted", "result": near, "kwargs": {"special_mora_threshold_profile": "v2_limited_candidate", "enable_user_facing_calibrated_special_mora": True}},
        {"name": "clear long vowel too_short flag off", "result": clear_short, "kwargs": {"special_mora_threshold_profile": "v2_limited_candidate"}},
        {"name": "clear long vowel too_short flag on gentle only", "result": clear_short, "kwargs": {"special_mora_threshold_profile": "v2_limited_candidate", "enable_user_facing_calibrated_special_mora": True}},
        {"name": "weak-reference ASR text not confirmed", "result": _with(base, details={"weak_reference": True, "mode": "asr_pseudo_reference"}), "kwargs": {"mode": "asr_pseudo_reference"}},
        {"name": "ASR+Kanade playback only", "result": _with(base, details={"mode": "kanade_asr_voice_reference", "demo_only": True, "exclude_from_pronunciation_score": True}), "kwargs": {"mode": "kanade_asr_voice_reference"}},
        {"name": "poor recording quality retry", "result": _with(base, details={"recording_quality": {"score": 0.1}}), "kwargs": {}},
        {"name": "low F0 coverage suppresses prosody", "result": _with(base, details={"reliability": {"level": "high", "overall": 0.9, "alignment": 0.9, "f0_coverage": 0.1}}), "kwargs": {}},
        {"name": "sokuon issue blocked", "result": _with(base, target_text="きって", kana="キッテ", moras=["キ", "ッ", "テ"], mora_table=[{"mora": "キ", "start_sec": 0.0, "end_sec": 0.2}, {"mora": "ッ", "start_sec": 0.2, "end_sec": 0.21}, {"mora": "テ", "start_sec": 0.21, "end_sec": 0.41}]), "kwargs": {"special_mora_threshold_profile": "v2_limited_candidate", "enable_user_facing_calibrated_special_mora": True}},
        {"name": "yoon duration debug-only", "result": _with(base, target_text="きゃ", kana="キャ", moras=["キャ"], mora_table=[{"mora": "キャ", "start_sec": 0.0, "end_sec": 0.12}]), "kwargs": {"special_mora_threshold_profile": "v2_limited_candidate", "enable_user_facing_calibrated_special_mora": True}},
    ]


def main() -> None:
    lines = ["# User-facing policy examples", ""]
    for example in build_examples():
        rendered = render_user_facing_result(example["result"], **example["kwargs"])
        lines.extend([
            f"## {example['name']}",
            "",
            f"- raw condition: `{example['kwargs']}`",
            f"- status: {rendered.get('status')}",
            f"- practice_score: {rendered.get('practice_score')}",
            f"- summary_text: {rendered.get('summary_text')}",
            f"- primary_suggestion_text: {rendered.get('primary_suggestion_text')}",
            f"- suppressed_reasons: {rendered.get('suppressed_reasons')}",
            "",
        ])
    out = ROOT / "reports" / "user_facing_policy_examples.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print({"examples": len(build_examples()), "report": str(out)})


if __name__ == "__main__":
    main()
