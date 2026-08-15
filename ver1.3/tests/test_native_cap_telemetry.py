from __future__ import annotations

from pathlib import Path

from jp_speech_eval.evaluator import evaluate_utterance
from jp_speech_eval.native_cap_telemetry import (
    has_native_cap_telemetry,
    report_from_native_cap_telemetry,
)


def test_fixed_evaluator_persists_exact_pre_post_cap_snapshot() -> None:
    root = Path(__file__).resolve().parents[1]
    prefix = root / "assets" / "reference_cache" / "ramen_kudasai_aivis"
    wav = prefix.with_suffix(".ref.wav")
    assert wav.exists()

    result = evaluate_utterance(
        wav_path=wav,
        alignment_mode="cached_dtw",
        cache_path=prefix,
        use_content_match=False,
        profile=False,
    ).to_dict()

    assert has_native_cap_telemetry(result) is True
    audit = result["details"]["reliability_cap_audit"]
    assert audit["schema_version"] == "reliability_cap_audit_v1"
    assert audit["post_component_cap_scores"]["pronunciation"] == result["pronunciation_score"]
    assert audit["post_component_cap_scores"]["prosody"] == result["prosody_score"]
    assert audit["post_component_cap_scores"]["fluency"] == result["fluency_score"]
    assert audit["post_component_cap_scores"]["tone"] == result["tone_score"]
    assert audit["post_overall_cap_total"] == result["total_score"]
    assert audit["product_score_changed"] is False

    report = report_from_native_cap_telemetry(result)
    assert report["counterfactual_trustworthy"] is True
    assert report["counterfactual_trust_level"] == "native_pre_cap_telemetry_exact"
    assert report["candidate_pre_cap_scores_from_current_scorer"]["pronunciation"] == audit["pre_cap_scores"]["pronunciation"]
    assert report["candidate_pre_cap_scores_from_current_scorer"]["total"] == audit["pre_component_cap_total"]
    assert report["observed_legacy_evaluator_scores"]["total"] == result["total_score"]
