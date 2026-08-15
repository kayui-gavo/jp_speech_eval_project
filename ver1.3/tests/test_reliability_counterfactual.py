from __future__ import annotations

import numpy as np
import soundfile as sf

from jp_speech_eval.reliability_counterfactual import (
    _component_consistency,
    reliability_cap_triggers,
    rescore_without_reliability_caps,
    summarize_counterfactual_reports,
)


def _result_fixture() -> dict:
    return {
        "target_text": "あいうえ",
        "kana": "アイウエ",
        "moras": ["ア", "イ", "ウ", "エ"],
        "target_pitch": ["L", "H", "H", "L"],
        "duration_sec": 1.0,
        "alignment_mode": "cached_dtw_fallback_equal",
        "pronunciation_score": 60,
        "prosody_score": 55,
        "fluency_score": 96,
        "tone_score": 85,
        "total_score": 72,
        "pause_info": {
            "pause_count": 0,
            "pause_total": 0.0,
            "pause_ratio": 0.0,
            "pause_segments": [],
            "analysis_duration_sec": 1.0,
        },
        "mora_table": [
            {"start_sec": 0.00, "end_sec": 0.25, "f0_hz": 180.0},
            {"start_sec": 0.25, "end_sec": 0.50, "f0_hz": 220.0},
            {"start_sec": 0.50, "end_sec": 0.75, "f0_hz": 215.0},
            {"start_sec": 0.75, "end_sec": 1.00, "f0_hz": 175.0},
        ],
        "details": {
            "reliability": {"overall": 0.60, "f0_coverage": 0.40},
            "mora_evidence_summary": {"judgement_available_count": 1},
            "prosody": {"pitch_target_source": "test"},
            "reference_f0_by_mora": [180.0, 220.0, 215.0, 175.0],
            "accent_phrases": [],
            "aggregate": {
                "weights": {
                    "pronunciation": 0.35,
                    "prosody": 0.40,
                    "fluency": 0.25,
                    "tone": 0.0,
                }
            },
        },
    }


def test_cap_trigger_audit_mirrors_legacy_predicates() -> None:
    triggers = reliability_cap_triggers(_result_fixture())
    assert triggers["applicable"] is True
    assert triggers["alignment_equal_fallback"] is True
    assert triggers["mora_evidence_below_threshold"] is True
    assert triggers["f0_coverage_below_0_50"] is True
    assert triggers["overall_reliability_below_0_75"] is True


def test_broad_mode_missing_mora_evidence_is_not_miscounted_as_cap_trigger() -> None:
    broad = {
        "alignment_mode": "none",
        "moras": ["ア", "イ"],
        "mora_table": [],
        "details": {"reliability": {"overall": 0.5, "f0_coverage": 0.2}},
    }
    triggers = reliability_cap_triggers(broad)
    assert triggers["applicable"] is False
    assert triggers["mora_evidence_below_threshold"] is False
    assert triggers["f0_coverage_below_0_50"] is False
    assert triggers["overall_reliability_below_0_75"] is False


def test_untriggered_component_change_is_historical_scorer_drift() -> None:
    item = _component_consistency(
        key="fluency",
        observed=80,
        replayed_pre_cap=85,
        expected_post_cap=85,
        triggers={},
        same_run=False,
    )
    assert item["consistent"] is False
    assert item["identifiability"] == "historical_scorer_or_config_drift"


def test_cap_equation_match_can_still_be_censored_at_ceiling() -> None:
    item = _component_consistency(
        key="prosody",
        observed=55,
        replayed_pre_cap=91,
        expected_post_cap=55,
        triggers={"f0_coverage_below_0_50": True},
        same_run=False,
    )
    assert item["consistent"] is True
    assert item["effective_cap"] == 55
    assert item["identifiability"] == "historical_censored_at_cap"


def test_cap_nonbinding_historical_score_is_identifiable() -> None:
    item = _component_consistency(
        key="prosody",
        observed=42,
        replayed_pre_cap=42,
        expected_post_cap=42,
        triggers={"f0_coverage_below_0_50": True},
        same_run=False,
    )
    assert item["consistent"] is True
    assert item["identifiability"] == "historical_cap_nonbinding_exact"


def test_formula_replay_is_product_neutral_and_marks_compatibility(tmp_path) -> None:
    sr = 16000
    wav = tmp_path / "utterance.wav"
    t = np.arange(sr, dtype=float) / sr
    y = 0.15 * np.sin(2.0 * np.pi * 200.0 * t)
    sf.write(wav, y, sr, subtype="FLOAT")

    report = rescore_without_reliability_caps(_result_fixture(), wav_path=wav, sample_rate=sr)
    assert report["available"] is True
    assert report["product_behavior_changed"] is False
    assert report["observed_legacy_evaluator_scores"]["pronunciation"] == 60
    assert report["candidate_pre_cap_scores_from_current_scorer"]["pronunciation"] >= 60
    assert "historical_replay_compatible" in report["replay_consistency"]
    assert report["counterfactual_trust_level"] in {
        "historical_scorer_or_config_drift",
        "historical_cap_compatible_but_pre_cap_censored",
        "historical_exact_for_uncensored_components",
    }


def test_wav_free_replay_carries_stored_tone_without_claiming_tone_replay() -> None:
    result = _result_fixture()
    result["details"]["aggregate"]["weights"]["tone"] = 0.10
    report = rescore_without_reliability_caps(result, wav_path=None)
    assert report["source_wav_available"] is False
    assert report["tone_replayed"] is False
    assert report["candidate_pre_cap_scores_from_current_scorer"]["tone"] == 85
    assert report["replay_consistency"]["tone"]["checked"] is False
    assert report["replay_consistency"]["tone"]["identifiability"] == "stored_unchanged_no_cap"
    assert report["candidate_pre_cap_scores_from_current_scorer"]["total"] is not None


def test_same_run_consistency_marks_component_replay_exact() -> None:
    item = _component_consistency(
        key="pronunciation",
        observed=60,
        replayed_pre_cap=95,
        expected_post_cap=60,
        triggers={"alignment_equal_fallback": True, "mora_evidence_below_threshold": True},
        same_run=True,
    )
    assert item["consistent"] is True
    assert item["identifiability"] == "same_run_exact"


def test_summary_excludes_drift_and_censoring_from_trusted_delta_stats() -> None:
    reports = [
        {
            "available": True,
            "candidate_pre_cap_minus_observed": {"pronunciation": 0, "prosody": 0, "fluency": 0, "tone": 0, "total": 0},
            "counterfactual_trustworthy": True,
            "counterfactual_trust_level": "historical_exact_for_uncensored_components",
            "replay_consistency": {"historical_replay_compatible": True},
            "cap_triggers": {"applicable": True, "alignment_equal_fallback": False, "mora_evidence_below_threshold": False, "f0_coverage_below_0_50": False, "overall_reliability_below_0_75": False},
        },
        {
            "available": True,
            "candidate_pre_cap_minus_observed": {"pronunciation": 40, "prosody": 0, "fluency": 0, "tone": 0, "total": 14},
            "counterfactual_trustworthy": False,
            "counterfactual_trust_level": "historical_cap_compatible_but_pre_cap_censored",
            "replay_consistency": {"historical_replay_compatible": True},
            "cap_triggers": {"applicable": True, "alignment_equal_fallback": True, "mora_evidence_below_threshold": True, "f0_coverage_below_0_50": False, "overall_reliability_below_0_75": False},
        },
        {
            "available": True,
            "candidate_pre_cap_minus_observed": {"pronunciation": 0, "prosody": -30, "fluency": 0, "tone": 0, "total": -12},
            "counterfactual_trustworthy": False,
            "counterfactual_trust_level": "historical_scorer_or_config_drift",
            "replay_consistency": {"historical_replay_compatible": False},
            "cap_triggers": {"applicable": True, "alignment_equal_fallback": False, "mora_evidence_below_threshold": False, "f0_coverage_below_0_50": False, "overall_reliability_below_0_75": False},
        },
        {"available": False, "cap_triggers": {"applicable": False}},
    ]
    summary = summarize_counterfactual_reports(reports)
    assert summary["report_count"] == 4
    assert summary["applicable_count"] == 3
    assert summary["historical_replay_compatible_count"] == 2
    assert summary["historical_scorer_or_config_drift_count"] == 1
    assert summary["historical_pre_cap_censored_count"] == 1
    assert summary["counterfactual_trustworthy_count"] == 1
    assert summary["raw_candidate_delta_stats"]["total"]["n"] == 3
    assert summary["trusted_counterfactual_delta_stats"]["total"]["n"] == 1
    assert summary["trusted_counterfactual_delta_stats"]["total"]["mean"] == 0.0
    assert summary["decision"] == "none"
