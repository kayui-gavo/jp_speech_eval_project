from __future__ import annotations

import csv
from pathlib import Path

from scripts.analyze_free_speech_promotion_v5 import analyze


SCORE_SURFACES = {
    "clarity.shadow_candidate_score": "clarity_comprehensibility",
    "rhythm.shadow_candidate_score": "rhythm_naturalness",
    "fluency.current_product_proxy": "fluency",
    "intonation.shadow_candidate_score": "intonation_utterance_naturalness",
}
ALL_CANDIDATES = [
    ("clarity.asr_recoverability_index", "clarity_comprehensibility"),
    ("clarity.word_probability_median", "clarity_comprehensibility"),
    ("clarity.shadow_candidate_score", "clarity_comprehensibility"),
    ("rhythm.negative_local_tempo_mad", "rhythm_naturalness"),
    ("rhythm.negative_local_tempo_spread", "rhythm_naturalness"),
    ("rhythm.shadow_candidate_score", "rhythm_naturalness"),
    ("fluency.current_product_proxy", "fluency"),
    ("fluency.speech_rate_mora_per_sec", "fluency"),
    ("intonation.robust_range_semitones", "intonation_utterance_naturalness"),
    ("intonation.shadow_candidate_score", "intonation_utterance_naturalness"),
]
SHADOWS = {
    "clarity.shadow_candidate_score",
    "rhythm.shadow_candidate_score",
    "intonation.shadow_candidate_score",
}


def _write(path: Path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_fixture(tmp_path: Path, *, blank_channel_pairs: bool = False, bad_contract: bool = False):
    human_rows = []
    evidence_rows = []
    criteria = list(SCORE_SURFACES.values())
    for i in range(50):
        pair = i // 2
        speaker_num = pair % 10
        speaker = f"spk{speaker_num}"
        group = "learner" if speaker_num < 6 else "native"
        task = "spontaneous" if pair % 2 == 0 else "controlled_dialogue"
        condition = "clean" if i % 2 == 0 else "low_level"
        pair_id = "" if blank_channel_pairs else f"pair{pair}"
        source_recording_id = f"source{pair}"
        score = 50.0 + i * 0.8
        rating = min(7, 1 + i // 8)
        sample_id = f"s{i:02d}"
        for criterion in criteria:
            for rater in range(3):
                human_rows.append(
                    {
                        "sample_id": sample_id,
                        "criterion": criterion,
                        "human_rating": rating,
                        "rater_id": f"r{rater}",
                        "presentation_id": f"{sample_id}_{criterion}_{rater}",
                        "presentation_variant": "isolated",
                        "task": task,
                        "speaker_id": speaker,
                        "speaker_group": group,
                        "l1": "zh" if group == "learner" else "ja",
                        "prompt_id": f"prompt{pair}",
                        "subset": "held",
                        "expected_language": "ja",
                        "condition": condition,
                        "channel_pair_id": pair_id,
                        "source_recording_id": source_recording_id,
                        "context_type": "prompt",
                        "context_id": f"ctx{pair}",
                    }
                )
        for candidate, construct in ALL_CANDIDATES:
            if candidate in SCORE_SURFACES:
                numeric_value = score
            elif candidate.startswith("clarity."):
                numeric_value = score / 100.0
            elif candidate.startswith("rhythm."):
                numeric_value = (score - 70.0) / 100.0
            elif candidate == "fluency.speech_rate_mora_per_sec":
                numeric_value = 4.5 + (i % 5) * 0.25
            else:
                numeric_value = 5.0 + (i % 7) * 0.2

            available = True
            fallback_numeric_value = numeric_value if candidate in SHADOWS else ""
            evidence_value = numeric_value
            failure_reason = ""
            if candidate == "intonation.robust_range_semitones" and i == 30:
                evidence_value = ""
                available = False
                failure_reason = "f0_unavailable"
            if candidate == "intonation.shadow_candidate_score" and i == 30:
                # Real v4 surface intentionally keeps neutral 70 while marking
                # source evidence unavailable. It must not enter correlation.
                evidence_value = ""
                fallback_numeric_value = 70.0
                available = False
                failure_reason = "neutral_placeholder_without_source_evidence"

            contract = "wrong_contract" if bad_contract and i == 0 else "consumer_four_score_v2"
            evidence_rows.append(
                {
                    "sample_id": sample_id,
                    "speaker_id": speaker,
                    "speaker_group": group,
                    "l1": "zh" if group == "learner" else "ja",
                    "task": task,
                    "prompt_id": f"prompt{pair}",
                    "subset": "held",
                    "expected_language": "ja",
                    "condition": condition,
                    "channel_pair_id": pair_id,
                    "source_recording_id": source_recording_id,
                    "context_type": "prompt",
                    "context_id": f"ctx{pair}",
                    "candidate": candidate,
                    "candidate_construct": construct,
                    "evidence_value": evidence_value,
                    "fallback_numeric_value": fallback_numeric_value,
                    "available": "true" if available else "false",
                    "failure_reason": failure_reason,
                    "evidence_direction": "higher_better_hypothesis",
                    "model_id": "test",
                    "model_version": "v1",
                    "batch_status": "ok",
                    "product_score_available": "true",
                    "language_gate_eligible": "true",
                    "language_gate_reason": "ja",
                    "recording_quality_score": "0.9",
                    "scoring_used_gold_transcript": "false",
                    "score_contract_version": contract,
                    "evidence_schema_version": "consumer_evidence_v3",
                    "candidate_surface_policy_id": "free_speech_candidate_surface_v1_shadow",
                    "export_schema": "test",
                }
            )

    for j, language in enumerate(["en", "zh", "non_speech", "other"]):
        sample_id = f"neg{j}"
        for candidate, construct in ALL_CANDIDATES:
            evidence_rows.append(
                {
                    "sample_id": sample_id,
                    "speaker_id": f"negspk{j}",
                    "speaker_group": "negative_control",
                    "l1": "",
                    "task": "spontaneous",
                    "prompt_id": "negative",
                    "subset": "held",
                    "expected_language": language,
                    "condition": "clean",
                    "channel_pair_id": "",
                    "source_recording_id": f"negative_source{j}",
                    "context_type": "none",
                    "context_id": "",
                    "candidate": candidate,
                    "candidate_construct": construct,
                    "evidence_value": "",
                    "fallback_numeric_value": "",
                    "available": "false",
                    "failure_reason": "language_reject",
                    "evidence_direction": "higher_better_hypothesis",
                    "model_id": "test",
                    "model_version": "v1",
                    "batch_status": "ok",
                    "product_score_available": "false",
                    "language_gate_eligible": "false",
                    "language_gate_reason": "non_ja",
                    "recording_quality_score": "0.9",
                    "scoring_used_gold_transcript": "false",
                    "score_contract_version": "consumer_four_score_v2",
                    "evidence_schema_version": "consumer_evidence_v3",
                    "candidate_surface_policy_id": "free_speech_candidate_surface_v1_shadow",
                    "export_schema": "test",
                }
            )

    human_path = tmp_path / "human.csv"
    evidence_path = tmp_path / "evidence.csv"
    human_fields = list(human_rows[0].keys())
    evidence_fields = list(evidence_rows[0].keys())
    _write(human_path, human_fields, human_rows)
    _write(evidence_path, evidence_fields, evidence_rows)
    return human_path, evidence_path


def test_score_surfaces_can_pass_only_full_cend_gate(tmp_path):
    human, evidence = _build_fixture(tmp_path)
    report = analyze(human, evidence, "data/research_eval/free_speech_v5_promotion_protocol.json")
    for candidate in SCORE_SURFACES:
        candidate_report = report["candidates"][candidate]
        assert candidate_report["promotion_decision"] == "pass"
        assert set(candidate_report["gate_states"].values()) == {"pass"}
        assert candidate_report["version_consistency"]["score_contract_match"] is True
    assert report["candidates"]["clarity.asr_recoverability_index"]["promotion_decision"] == "diagnostic_only"
    assert report["candidates"]["intonation.robust_range_semitones"]["promotion_decision"] == "diagnostic_only"
    assert "intonation_contextual_appropriateness" not in report["candidates"]["intonation.shadow_candidate_score"]["construct_match"]
    assert report["candidates"]["intonation.shadow_candidate_score"]["criteria"]["intonation_utterance_naturalness"]["held_available_count"] == 49
    assert report["f0_missingness_safety"]["f0_missing_case_count"] == 1
    assert report["f0_missingness_safety"]["pass"] is True
    assert report["f0_missingness_safety"]["cases"][0]["candidate_score"] == 70.0
    assert report["global_dataset"]["negative_control_no_score_rate"] == 1.0
    assert report["overall_direct_promotion_state"] == "pass"


def test_missing_channel_controls_yields_insufficient_not_false_pass(tmp_path):
    human, evidence = _build_fixture(tmp_path, blank_channel_pairs=True)
    report = analyze(human, evidence, "data/research_eval/free_speech_v5_promotion_protocol.json")
    clarity = report["candidates"]["clarity.shadow_candidate_score"]
    assert clarity["gate_states"]["channel_pair_coverage"] == "insufficient"
    assert clarity["gate_states"]["channel_drift_points"] == "insufficient"
    assert clarity["promotion_decision"] == "insufficient"


def test_mixed_score_contract_versions_fail_closed(tmp_path):
    human, evidence = _build_fixture(tmp_path, bad_contract=True)
    report = analyze(human, evidence, "data/research_eval/free_speech_v5_promotion_protocol.json")
    clarity = report["candidates"]["clarity.shadow_candidate_score"]
    assert clarity["version_consistency"]["score_contract_match"] is False
    assert clarity["gate_states"]["score_contract_version_match"] == "fail"
    assert clarity["promotion_decision"] == "fail"
