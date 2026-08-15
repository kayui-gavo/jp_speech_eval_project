from __future__ import annotations

from jp_speech_eval.spontaneous_fluency import (
    _adjacent_repetition_candidates,
    breakdown_fluency_features,
    build_spontaneous_fluency_evidence,
    speed_fluency_features,
    transcript_repair_features,
)


def test_speed_separates_speech_rate_from_articulation_rate() -> None:
    features = speed_fluency_features(
        mora_count=20,
        speech_duration_sec=5.0,
        silent_pause_total_sec=1.0,
    )
    assert features["speech_rate_mora_per_sec"] == 4.0
    assert features["articulation_rate_mora_per_sec"] == 5.0
    assert features["phonation_time_ratio"] == 0.8


def test_breakdown_reports_pause_distribution_without_inventing_clause_location() -> None:
    pause_info = {
        "pause_segments": [(1.0, 1.4), (3.0, 3.6)],
        "pause_count": 2,
        "pause_total": 1.0,
        "pause_ratio": 0.2,
    }
    features = breakdown_fluency_features(
        pause_info,
        speech_duration_sec=5.0,
        mora_count=20,
    )
    assert features["silent_pause_count"] == 2
    assert features["silent_pause_total_sec"] == 1.0
    assert features["silent_pause_mean_sec"] == 0.5
    assert features["silent_pauses_per_100_mora"] == 10.0
    assert features["pause_location_available"] is False
    assert features["pause_location_reason"] == "no_word_timestamps"


def test_word_timing_produces_only_weak_punctuation_boundary_candidates() -> None:
    pause_info = {"pause_segments": [(0.9, 1.3), (2.0, 2.4)]}
    words = [
        {"start_sec": 0.10, "end_sec": 0.88, "text": "今日は、", "probability": 0.95},
        {"start_sec": 1.32, "end_sec": 1.95, "text": "映画を", "probability": 0.91},
        {"start_sec": 2.42, "end_sec": 2.90, "text": "見ます", "probability": 0.90},
    ]
    features = breakdown_fluency_features(
        pause_info,
        speech_duration_sec=3.0,
        mora_count=12,
        word_timestamps=words,
    )
    assert features["pause_location_available"] is True
    assert features["pause_location_confidence"] == "low"
    assert features["pause_location_counts"]["after_asr_punctuation_candidate"] == 1
    assert features["pause_location_counts"]["within_asr_phrase_candidate"] == 1
    assert "not_syntactic_clause_labels" in features["pause_location_reason"]


def test_transcript_repairs_keep_certain_fillers_separate_from_ambiguous_markers() -> None:
    evidence = transcript_repair_features(
        "えーと、あの、今日は、うーん、まあ映画を見ました。",
        transcript_source="external_test",
    )
    assert evidence["high_precision_filled_pause_count"] == 2
    assert evidence["ambiguous_discourse_marker_counts"]["あの"] == 1
    assert evidence["ambiguous_discourse_marker_counts"]["まあ"] == 1
    assert evidence["score_mapped"] is False
    assert evidence["repair_evidence_confidence"] == "low"


def test_exact_adjacent_repetition_is_candidate_not_error_count() -> None:
    candidates = _adjacent_repetition_candidates(["私", "は", "私", "は", "学生", "です"])
    assert len(candidates) == 1
    assert candidates[0]["surface"] == "私は"
    assert candidates[0]["evidence"] == "exact_adjacent_token_ngram_repetition"


def test_full_payload_is_shadow_only() -> None:
    payload = build_spontaneous_fluency_evidence(
        mora_count=12,
        speech_duration_sec=3.0,
        pause_info={"pause_segments": [(1.0, 1.4)]},
        transcript="えっと今日は映画です",
        transcript_source="asr",
    )
    assert payload["construct"] == "utterance_fluency_speed_breakdown_repair"
    assert payload["score_mapped"] is False
    assert payload["product_calibrated"] is False
    assert payload["user_facing"] is False
