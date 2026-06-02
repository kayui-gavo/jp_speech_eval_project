#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_demo_flow_smoke_tests import _clear_short_long_vowel, _near_boundary_long_vowel
from demo_user_facing_policy_examples import _base_result, _with
from jp_speech_eval.feedback_renderer import render_user_facing_result


ROOT = Path(__file__).resolve().parents[1]


def _debug_summary(rendered: Dict[str, Any]) -> Dict[str, Any]:
    debug = rendered.get("debug", {})
    return {
        "weak_reference": debug.get("weak_reference"),
        "demo_only": debug.get("demo_only"),
        "alignment_confidence": debug.get("alignment_confidence"),
        "f0_voiced_coverage": debug.get("f0_voiced_coverage"),
        "special_mora_decision_count": len(debug.get("special_mora_decisions") or []),
        "allow_pitch_feedback": debug.get("scoring_policy", {}).get("allow_pitch_feedback"),
        "exclude_from_pronunciation_score": debug.get("scoring_policy", {}).get("exclude_from_pronunciation_score"),
    }


def _safe_example(name: str, condition: Dict[str, Any], result: Dict[str, Any], kwargs: Dict[str, Any], reason: str) -> Dict[str, Any]:
    rendered = render_user_facing_result(result, **kwargs)
    user_facing = {key: value for key, value in rendered.items() if key != "debug"}
    return {
        "name": name,
        "input_condition": condition,
        "response": {
            "user_facing": user_facing,
            "debug_summary": _debug_summary(rendered),
        },
        "why_safe_for_c_end_ui": reason,
    }


def build_examples() -> List[Dict[str, Any]]:
    base = _base_result()
    clear_short = _clear_short_long_vowel()
    near = _near_boundary_long_vowel()
    auto_target = _with(
        base,
        target_text="コーヒーをください",
        kana="コーヒーヲクダサイ",
        moras=["コ", "ー", "ヒ", "ー", "ヲ", "ク", "ダ", "サ", "イ"],
        details={"verified_level": "auto_pyopenjtalk", "pitch_target_source": "auto_pyopenjtalk"},
    )
    return [
        _safe_example(
            "fixed_reference_pass",
            {"mode": "reference", "target_id": "ramen_kudasai"},
            base,
            {},
            "Learner fields contain only status, practice score, summary, and mode notice.",
        ),
        _safe_example(
            "fixed_reference_practice_suggestion",
            {"mode": "reference", "special_mora_flag": "on"},
            clear_short,
            {"special_mora_threshold_profile": "v2_limited_candidate", "enable_user_facing_calibrated_special_mora": True},
            "Special mora appears only as a gentle practice point, not a correctness penalty.",
        ),
        _safe_example(
            "fixed_reference_retry_poor_recording",
            {"mode": "reference", "recording_quality": "poor"},
            _with(base, details={"recording_quality": {"score": 0.1}}),
            {},
            "Retry is framed as recording quality, not pronunciation failure.",
        ),
        _safe_example(
            "weak_reference_unconfirmed_asr",
            {"mode": "asr_pseudo_reference", "confirmed": False},
            _with(base, details={"mode": "asr_pseudo_reference", "weak_reference": True}),
            {"mode": "asr_pseudo_reference"},
            "Unconfirmed ASR stays debug-only and has no practice score value.",
        ),
        _safe_example(
            "weak_reference_confirmed",
            {"mode": "asr_confirmed_weak_reference", "confirmed": True},
            _with(base, details={"mode": "asr_confirmed_weak_reference", "weak_reference": True}),
            {"mode": "asr_confirmed_weak_reference"},
            "Confirmed weak-reference keeps a weak-reference notice visible.",
        ),
        _safe_example(
            "asr_kanade_playback_notice",
            {"mode": "kanade_asr_voice_reference", "kanade_audio": "mocked"},
            _with(base, details={"mode": "kanade_asr_voice_reference", "demo_only": True, "playback_only": True, "exclude_from_pronunciation_score": True}),
            {"mode": "kanade_asr_voice_reference"},
            "Kanade notice says voice similarity is not scored; correctness scoring is excluded.",
        ),
        _safe_example(
            "special_mora_suppressed",
            {"mode": "reference", "special_mora_flag": "off"},
            near,
            {"special_mora_threshold_profile": "v2_limited_candidate"},
            "Near-boundary special-mora evidence remains hidden from learner fields.",
        ),
        _safe_example(
            "pitch_suppressed_unverified_target",
            {"mode": "reference", "target_id": "coffee_kudasai", "verified_level": "auto_pyopenjtalk"},
            auto_target,
            {},
            "Pitch feedback is suppressed because the target is not OJAD/manual verified.",
        ),
    ]


def write_outputs(examples: List[Dict[str, Any]]) -> None:
    out_dir = ROOT / "results" / "demo_readiness"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "demo_api_examples.json"
    json_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Demo API response examples", ""]
    for item in examples:
        lines.extend([
            f"## {item['name']}",
            "",
            f"- input condition: `{item['input_condition']}`",
            f"- user_facing.status: `{item['response']['user_facing'].get('status')}`",
            f"- user_facing.practice_score: `{item['response']['user_facing'].get('practice_score')}`",
            f"- user_facing.summary_text: {item['response']['user_facing'].get('summary_text')}",
            f"- user_facing.primary_suggestion_text: {item['response']['user_facing'].get('primary_suggestion_text')}",
            f"- debug summary: `{item['response']['debug_summary']}`",
            f"- why safe: {item['why_safe_for_c_end_ui']}",
            "",
        ])
    (ROOT / "reports" / "demo_api_response_examples.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    examples = build_examples()
    write_outputs(examples)
    print({"examples": len(examples), "json": str(ROOT / "results" / "demo_readiness" / "demo_api_examples.json")})


if __name__ == "__main__":
    main()
