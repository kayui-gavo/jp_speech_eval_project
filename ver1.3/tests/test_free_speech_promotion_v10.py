from __future__ import annotations

import csv
import json
from pathlib import Path

import scripts.analyze_free_speech_promotion_v10 as v10
from scripts.export_free_speech_v10_evidence import _speech_duration_sec


DIRECT = {
    "clarity.shadow_candidate_score": "clarity_comprehensibility",
    "rhythm.shadow_candidate_score": "rhythm_naturalness",
    "fluency.current_product_proxy": "fluency",
    "intonation.shadow_candidate_score": "intonation_utterance_naturalness",
}


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path, *, collapsed_learner: bool) -> tuple[Path, Path]:
    human: list[dict] = []
    evidence: list[dict] = []
    for i in range(40):
        learner = i < 30
        group = "learner" if learner else "native"
        speaker = f"spk{i % 10}"
        duration = 2.0 if i % 2 == 0 else 10.0
        human_value = 2.0 + i * 0.11 if learner else 6.5
        if learner:
            machine_value = 68.0 + i * (0.03 if collapsed_learner else 0.9)
        else:
            machine_value = 94.0 + (i - 30) * 0.2
        sample_id = f"s{i:02d}"
        for candidate, criterion in DIRECT.items():
            for rater in range(3):
                human.append(
                    {
                        "sample_id": sample_id,
                        "criterion": criterion,
                        "human_rating": human_value,
                        "rater_id": f"r{rater}",
                        "presentation_id": f"{sample_id}_{criterion}_{rater}",
                        "presentation_variant": "isolated",
                        "task": "spontaneous" if i % 4 < 2 else "controlled_dialogue",
                        "speaker_id": speaker,
                        "speaker_group": group,
                        "l1": "zh" if learner else "ja",
                        "prompt_id": f"p{i % 8}",
                        "subset": "held",
                        "expected_language": "ja",
                        "condition": "clean",
                        "channel_pair_id": "",
                        "source_recording_id": sample_id,
                        "context_type": "prompt",
                        "context_id": f"ctx{i % 8}",
                    }
                )
            evidence.append(
                {
                    "sample_id": sample_id,
                    "speaker_id": speaker,
                    "speaker_group": group,
                    "l1": "zh" if learner else "ja",
                    "task": "spontaneous" if i % 4 < 2 else "controlled_dialogue",
                    "prompt_id": f"p{i % 8}",
                    "subset": "held",
                    "expected_language": "ja",
                    "condition": "clean",
                    "channel_pair_id": "",
                    "source_recording_id": sample_id,
                    "context_type": "prompt",
                    "context_id": f"ctx{i % 8}",
                    "speech_duration_sec": duration,
                    "candidate": candidate,
                    "candidate_construct": criterion,
                    "evidence_value": machine_value,
                    "fallback_numeric_value": machine_value,
                    "available": "true",
                    "failure_reason": "",
                    "evidence_direction": "higher_better_hypothesis",
                    "model_id": "fixture",
                    "model_version": "v1",
                    "batch_status": "ok",
                    "product_score_available": "true",
                    "language_gate_eligible": "true",
                    "language_gate_reason": "ja",
                    "recording_quality_score": 0.9,
                    "scoring_used_gold_transcript": "false",
                    "score_contract_version": "consumer_four_score_v2",
                    "evidence_schema_version": "consumer_evidence_v3",
                    "candidate_surface_policy_id": "free_speech_candidate_surface_v1_shadow",
                    "export_schema": "free_speech_v10_evidence_export_v1",
                }
            )
    human_path = tmp_path / "human.csv"
    evidence_path = tmp_path / "evidence.csv"
    _write(human_path, human)
    _write(evidence_path, evidence)
    return human_path, evidence_path


def _mock_v5_pass(monkeypatch) -> None:
    monkeypatch.setattr(
        v10,
        "analyze_v5",
        lambda *args, **kwargs: {
            "schema": "free_speech_promotion_analysis_v5",
            "protocol_schema": "free_speech_v5_promotion_protocol_v1",
            "candidates": {candidate: {"promotion_decision": "pass"} for candidate in DIRECT},
            "overall_direct_promotion_state": "pass",
        },
    )


def test_v10_blocks_candidate_that_only_separates_groups_but_collapses_learners(tmp_path, monkeypatch) -> None:
    _mock_v5_pass(monkeypatch)
    human, evidence = _fixture(tmp_path, collapsed_learner=True)
    report = v10.analyze(
        human,
        evidence,
        "data/research_eval/free_speech_v5_promotion_protocol.json",
        "data/research_eval/free_speech_v10_consumer_promotion_protocol.json",
    )
    clarity = report["consumer_discrimination"]["clarity.shadow_candidate_score"]
    assert clarity["gate_states"]["base_v5_scientific_promotion"] == "pass"
    assert clarity["gate_states"]["held_learner_spearman"] == "pass"
    assert clarity["gate_states"]["held_learner_candidate_iqr"] == "fail"
    assert clarity["native_learner_group_direction"]["state"] == "pass"
    assert clarity["promotion_readiness"] == "fail"
    assert report["overall_v10_promotion_readiness"] == "fail"


def test_v10_can_pass_when_learner_range_and_short_long_behavior_are_healthy(tmp_path, monkeypatch) -> None:
    _mock_v5_pass(monkeypatch)
    human, evidence = _fixture(tmp_path, collapsed_learner=False)
    report = v10.analyze(
        human,
        evidence,
        "data/research_eval/free_speech_v5_promotion_protocol.json",
        "data/research_eval/free_speech_v10_consumer_promotion_protocol.json",
    )
    for candidate in DIRECT:
        item = report["consumer_discrimination"][candidate]
        assert set(item["gate_states"].values()) == {"pass"}
        assert item["promotion_readiness"] == "pass"
    assert report["overall_v10_promotion_readiness"] == "pass"
    assert report["product_score_changed"] is False
    assert report["thresholds_fitted_on_held_data"] is False


def test_speech_duration_comes_from_product_result_not_gold_transcript() -> None:
    row = {
        "raw_result": {
            "endpointing": {"speech_duration": 2.75},
            "details": {"asr": {"text": "これは機械の書き起こし"}},
        },
        "metadata": {"gold_transcript": "should not be used"},
    }
    assert _speech_duration_sec(row) == 2.75


def test_v10_protocol_is_additive_and_does_not_define_a_new_score_mapping() -> None:
    protocol = json.loads(Path("data/research_eval/free_speech_v10_consumer_promotion_protocol.json").read_text())
    assert protocol["inherits"] == "free_speech_v5_promotion_protocol_v1"
    assert protocol["rationale"]["no_new_score_mapping"]
    assert "held_learner_candidate_iqr_min_points" in protocol["consumer_discrimination_gates"]
