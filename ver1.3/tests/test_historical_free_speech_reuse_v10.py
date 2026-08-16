from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.audit_historical_free_speech_reuse_v10 import audit, classify_inventory_row


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_broad_mode_does_not_turn_jvs_read_speech_into_free_speech() -> None:
    item = classify_inventory_row(
        {
            "sample_id": "broad_jvs",
            "source": "JVS_parallel100",
            "speaker": "jvs002",
            "mode": "transcript_assisted_light",
            "expected_category": "broad_mode",
            "wav_path": "/corpora/JVS/jvs002/read.wav",
        }
    )
    assert item["reuse_class"] == "native_fixed_reading_regression"
    assert item["free_speech_criterion_eligible"] is False
    assert "scripted read speech" in item["reason"]


def test_janon_learner_audio_remains_fixed_reading_regression() -> None:
    item = classify_inventory_row(
        {
            "sample_id": "learner_chf1_i74",
            "source": "JANON",
            "speaker": "chf1",
            "mode": "reference",
            "expected_category": "learner",
            "wav_path": "/corpora/JANON/chf1_i74.wav",
        }
    )
    assert item["reuse_class"] == "fixed_reading_regression"
    assert item["free_speech_criterion_eligible"] is False


def test_audit_keeps_legacy_fixed_word_plan_out_of_four_dimension_free_speech(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.csv"
    plan = tmp_path / "plan.csv"
    controls = tmp_path / "controls.csv"
    _write(
        inventory,
        [
            {
                "sample_id": "j1", "source": "JVS_parallel100", "speaker": "jvs1",
                "mode": "transcript_assisted_light", "expected_category": "broad_mode", "wav_path": "/jvs.wav",
            },
            {
                "sample_id": "l1", "source": "JANON", "speaker": "chf1",
                "mode": "reference", "expected_category": "learner", "wav_path": "/janon.wav",
            },
        ],
    )
    _write(
        plan,
        [
            {
                "recruitment_id": "REC01_target_01",
                "planned_speaker_id_anonymized": "REC01",
                "target_id": "target_01",
                "target_text": "うっとうしい",
                "normalized_kana": "ウットーシイ",
                "required_recording": "one_clean_real_learner_recording",
                "required_metadata": "self_reported_native_language",
                "status": "needed",
            }
        ],
    )
    _write(
        controls,
        [
            {
                "sample_id": "english_tts",
                "wav_path": "/private/tmp/run/english_tts.wav",
                "control_type": "synthetic_engineering_control",
                "generation": "tts",
            }
        ],
    )

    report = audit(inventory, learner_plan_csv=plan, negative_controls_csv=controls)
    assert report["free_speech_criterion_ready_from_history"] is False
    assert report["free_speech_criterion_eligible_count"] == 0
    assert report["broad_mode_fixed_reading_count"] == 1
    assert report["legacy_learner_collection_plan"]["free_speech_four_dimension_ready"] is False
    assert report["historical_negative_controls"]["private_tmp_path_count"] == 1


def test_minimum_collection_plan_has_buffer_above_v10_learner_gate() -> None:
    plan = json.loads(Path("data/human_eval/free_speech_v10_minimum_collection_plan.json").read_text())
    protocol = json.loads(Path("data/research_eval/free_speech_v10_consumer_promotion_protocol.json").read_text())
    held_learner = plan["held_japanese_core"]["learner"]["clean_clip_total"]
    required = protocol["consumer_discrimination_gates"]["held_learner_construct_matched_pair_count"]
    assert held_learner > required
    assert plan["estimated_held_clean_japanese_recordings"] == 56
    assert "formal psychometric sample-size claim" in plan["explicit_non_goal"]
