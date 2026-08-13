import numpy as np
import pytest
from unittest.mock import patch

from jp_speech_eval.prosody_shadows import (
    compute_accent_nucleus_shadow,
    compute_phrase_intonation_shadow,
)
from jp_speech_eval.shadow_assessment import run_assessment_shadows
from jp_speech_eval.special_mora_shadow_v2 import compute_special_mora_v2_shadow
from jp_speech_eval.ssl_features import (
    aggregate_reference_distances,
    cosine_dtw_distance,
    fuse_layer_distances,
)
from jp_speech_eval.unified_result import unify_evaluation_result


def _result():
    return {
        "target_text": "ラーメン",
        "target_pitch": ["L", "H", "H", "L"],
        "moras": ["ラ", "ー", "メ", "ン"],
        "mora_table": [
            {"mora": "ラ", "start_sec": 0.0, "end_sec": 0.1, "f0_hz": 120.0},
            {"mora": "ー", "start_sec": 0.1, "end_sec": 0.3, "f0_hz": 150.0},
            {"mora": "メ", "start_sec": 0.3, "end_sec": 0.4, "f0_hz": 145.0},
            {"mora": "ン", "start_sec": 0.4, "end_sec": 0.5, "f0_hz": 110.0},
        ],
        "details": {
            "reference_id": "fixture",
            "pitch_target_source": "auto_pyopenjtalk",
            "reliability": {"alignment": 0.8},
        },
    }


def test_cosine_dtw_is_zero_for_identical_embeddings():
    features = np.eye(4, dtype=np.float32)
    distance = cosine_dtw_distance(features, features)
    assert distance["normalized_cumulative_distance"] == 0.0
    assert distance["reference_frame_count"] == 4


def test_ssl_multi_reference_aggregation_is_explicit():
    values = [0.1, 0.2, 0.9]
    assert aggregate_reference_distances(values, "median") == 0.2
    assert aggregate_reference_distances(values, "nearest") == 0.1
    assert aggregate_reference_distances(values, "top_k_mean", top_k=2) == pytest.approx(0.15)


def test_ssl_layer_fusion_is_native_normalized_before_weighting():
    fused = fuse_layer_distances(
        .24, .42,
        native_12=[.18, .20, .22], native_24=[.30, .32, .34], alpha=.5,
    )
    assert fused["normalized_distance_layer12"] > 0
    assert fused["normalized_distance_layer24"] > 0
    assert fused["fused_normalized_distance"] == pytest.approx(
        (fused["normalized_distance_layer12"] + fused["normalized_distance_layer24"]) / 2
    )


def test_local_special_mora_shadow_is_structured_and_not_user_facing():
    payload = compute_special_mora_v2_shadow(_result(), np.ones(8000, dtype=np.float32) * 0.1, 16000)
    assert payload["available"]
    assert {item["type"] for item in payload["evidence"]} == {"long_vowel", "moraic_nasal"}
    assert all(item["user_facing"] is False for item in payload["evidence"])
    assert all("features" in item for item in payload["evidence"])


def test_local_special_mora_shadow_does_not_claim_precise_decision_on_equal_fallback():
    result = _result()
    result["alignment_mode"] = "cached_dtw_fallback_equal"
    payload = compute_special_mora_v2_shadow(result, np.ones(8000, dtype=np.float32) * 0.1, 16000)
    assert payload["available"]
    assert payload["decision_available"] is False
    assert all(item["evidence_confidence"] == "low" for item in payload["evidence"])
    assert all(item["shadow_decision"] == "unavailable_alignment_or_roi_unreliable" for item in payload["evidence"])


def test_prosody_shadows_are_normalized_and_not_user_facing():
    phrase = compute_phrase_intonation_shadow(_result())
    nucleus = compute_accent_nucleus_shadow(_result())
    assert phrase["available"]
    assert phrase["user_facing"] is False
    assert nucleus["weak_target"] is True
    assert nucleus["user_facing"] is False
    assert phrase["phrase_intonation_score"] is None


def test_phrase_shadow_does_not_bridge_missing_mora_f0():
    result = _result()
    result["mora_table"][1]["f0_hz"] = None
    phrase = compute_phrase_intonation_shadow(result)
    # transitions 0->1 and 1->2 are both invalid; only 2->3 remains.
    assert phrase["adjacent_transition_count"] == 1


def test_accent_shadow_is_phrase_scoped_and_heiban_has_no_required_nucleus():
    result = _result()
    result["details"]["pitch_target_source"] = "human_checked"
    result["details"]["accent_phrases"] = [
        {"start_mora_index": 1, "end_mora_index": 2, "accent_position": 0},
        {"start_mora_index": 3, "end_mora_index": 4, "accent_position": 1},
    ]
    nucleus = compute_accent_nucleus_shadow(result)
    assert len(nucleus["phrases"]) == 2
    assert nucleus["phrases"][0]["target_type"] == "heiban"
    assert nucleus["phrases"][0]["target_correctness_candidate"] is None


def test_accent_nucleus_uses_pitch_target_provenance_not_reference_audio_identity():
    result = _result()
    result["details"].pop("pitch_target_source")
    result["details"]["reference_source"] = "jvs_native_external_reference_wav"
    result["prosody_metrics"] = {"hl_target_source": "openjtalk_accent_phrase_chain"}
    nucleus = compute_accent_nucleus_shadow(result)
    assert nucleus["target_source"] == "openjtalk_accent_phrase_chain"
    assert nucleus["weak_target"] is True


def test_disabled_shadows_do_not_load_audio_or_change_scores():
    result = _result()
    result["total_score"] = 77
    run_assessment_shadows(result, user_audio_path="missing.wav")
    assert result["total_score"] == 77
    assert result["details"]["shadow"] == {}


def test_shadow_failure_isolated_from_product_score():
    result = _result()
    result["total_score"] = 77
    with patch("jp_speech_eval.shadow_assessment._audio", side_effect=RuntimeError("fixture failure")):
        run_assessment_shadows(result, user_audio_path="missing.wav", enable_ssl_shadow=True)
    assert result["total_score"] == 77
    assert result["details"]["shadow"]["ssl_pronunciation"]["available"] is False


def test_ssl_shadow_can_reuse_preserved_fixed_reference_after_broad_fallback(tmp_path):
    prefix = tmp_path / "reference"
    prefix.with_suffix(".ref.wav").write_bytes(b"fixture")
    result = _result()
    result["details"]["fixed_reference_debug"] = {"cache_prefix": str(prefix)}

    class Extractor:
        def extract_layer(self, waveform, layer, sr):
            return np.eye(2, dtype=np.float32)

    with patch("jp_speech_eval.shadow_assessment._audio", return_value=(np.ones(32), 16000)):
        run_assessment_shadows(
            result,
            user_audio_path="user.wav",
            enable_ssl_shadow=True,
            ssl_extractor=Extractor(),
        )
    assert result["details"]["shadow"]["ssl_pronunciation"]["available"] is True


def test_unified_result_preserves_zero_values():
    raw = {
        "moras": [],
        "alignment_mode": "",
        "pause_info": {"pause_ratio": 0.0, "pause_count": 0},
        "endpointing": {"speech_duration": 0.0},
        "details": {
            "acoustic_features": {"speech_duration_sec": 9.0, "pause_ratio": 0.5, "pause_count": 4},
            "structure_features": {"mora_count": 0, "mora_rate": 0.0},
            "content_match": {"asr_provider": ""},
            "alignment": {"mode": "fallback"},
            "tone": {"pitch_range_log": 0.0, "energy": {"mean": 0.0, "cv": 0.0}},
        },
    }
    features = unify_evaluation_result(raw).features
    assert features["speech_duration_sec"] == 0.0
    assert features["pause_ratio"] == 0.0
    assert features["pause_count"] == 0
    assert features["mora_count"] == 0
    assert features["alignment_mode"] == ""
    assert features["asr_provider"] == ""
