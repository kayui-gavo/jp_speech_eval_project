from __future__ import annotations

from pathlib import Path

from jp_speech_eval.feedback_renderer import render_user_facing_result

ROOT = Path(__file__).resolve().parents[1]


def _fixed_result():
    moras = ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"]
    return {
        "target_text": "ラーメンをください",
        "moras": moras,
        "mora_table": [{"mora": m, "start_sec": i * .18, "end_sec": (i + 1) * .18} for i, m in enumerate(moras)],
        "total_score": 84, "pronunciation_score": 80, "prosody_score": 82, "fluency_score": 88, "tone_score": 80,
        "feedback": [], "alignment_mode": "cached_dtw",
        "details": {
            "mode": "reference", "verified_level": "human_checked", "pitch_target_source": "human_checked",
            "recording_quality": {"score": .92}, "content_match": {"status": "pass"},
            "alignment": {"mode": "cached_dtw", "available": True, "confidence": .90},
            "reliability": {"level": "high", "overall": .95, "endpointing": .95, "alignment": .90, "mora_evidence": .90, "f0_coverage": .90, "recording_quality": .92},
            "fluency": {"rate_score": 88, "pause_score": 90, "speech_rate_mora_per_sec": 5.2},
            "pronunciation": {"mora_duration_cv": .12, "special_mora_diagnostics": []},
            "prosody": {"contour_corr": .70, "contour_valid_mora_count": 9, "transition_agreement": .75, "pitch_target_source": "human_checked", "note": "ok"},
            "tone": {"pitch_range_log": .35, "pitch_score": 82},
            "mora_evidence": [{"judgement_available": True, "boundary_confidence": .85, "energy_coverage": .8} for _ in moras],
        },
    }


def test_user_facing_payload_exposes_measurement_context_not_one_confidence_probability():
    rendered = render_user_facing_result(_fixed_result(), mode="reference")
    context = rendered["measurement_context"]
    assert context["schema"] == "consumer_measurement_context_v1"
    assert context["recording_state"] == "good"
    assert context["dimension_evidence"]["dimension_count"] == 4
    assert context["single_confidence_percentage_allowed"] is False
    assert context["raw_reliability_numeric_user_facing"] is False
    assert rendered["evidence_schema_version"] == "consumer_evidence_v5"
    assert rendered["debug"]["reliability_gate"]["reliability"] == "high"


def test_space_ui_uses_measurement_facts_not_raw_reliability_percent():
    html = (ROOT / "debug_ui" / "index.html").read_text(encoding="utf-8")
    assert 'data-consumer-copy="evidenceBasis"' in html
    assert 'cc("recordingStatus")' in html
    assert 'cc("scoreEvidence")' in html
    assert 'cc("localDetail")' in html
    assert 'cc("resultLimitation")' in html
    assert 'measurement_context' in html
    assert 'Number(reliability.overall || 0) * 100' not in html
    assert '${t("reliabilitySimple")}：${reliabilityDisplay(reliability)}' not in html


def test_space_ui_has_evidence_context_copy_for_all_locales():
    html = (ROOT / "debug_ui" / "index.html").read_text(encoding="utf-8")
    for text in (
        'evidenceBasis:"本次判断依据"', 'evidenceBasis:"本次判斷依據"',
        'evidenceBasis:"今回の判断根拠"', 'evidenceBasis:"Evidence used this time"',
    ):
        assert text in html
