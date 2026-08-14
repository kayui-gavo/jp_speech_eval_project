from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from jp_speech_eval.pronunciation_calibration import default_pronunciation_calibration
from scripts.build_human_pronunciation_study import (
    CHANNEL_COMPONENT,
    CHANNEL_VALIDATION_SCOPE,
    FINAL_CHANNEL_CLIP_COUNT,
    FINAL_MIN_CHANNEL_SETS,
    FINAL_NATIVE_CLIP_COUNT,
    FINAL_PRIMARY_CLIP_COUNT,
    FINAL_STUDY_STAGE,
    PILOT_STUDY_STAGE,
    PRIMARY_COMPONENT,
    PRIMARY_VALIDATION_SCOPE,
    assignments,
    complete_channel_set_ids,
    final_assignment_from_real_manifest,
    primary_mapping_rows,
)


ROOT = Path(__file__).resolve().parents[1]
HUMAN_EVAL = ROOT / "data/human_eval"
PILOT = HUMAN_EVAL / "pilot"
FINAL_TEMPLATE = HUMAN_EVAL / "final_template"


def _rows(name: str, *, directory: Path = PILOT) -> list[dict[str, str]]:
    with (directory / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pilot_manifest_is_explicitly_seed_and_not_final_assignment():
    master = _rows("pronunciation_listener_manifest_v1.csv")
    assignments = _rows("listener_assignment_v1.csv")
    metadata = json.loads((PILOT / "pilot_study_metadata.json").read_text(encoding="utf-8"))
    assert master and {row["study_stage"] for row in master} == {PILOT_STUDY_STAGE}
    assert assignments and {row["study_stage"] for row in assignments} == {PILOT_STUDY_STAGE}
    assert metadata["study_stage"] == PILOT_STUDY_STAGE
    assert metadata["repeatability_status"] == "insufficient_for_stable_per_rater_repeatability"
    assert not (FINAL_TEMPLATE / "listener_assignment_final.csv").exists()


def test_pilot_manifest_separates_primary_and_channel_studies():
    master = _rows("pronunciation_listener_manifest_v1.csv")
    assert all(row["study_component"] for row in master)
    primary = [row for row in master if row["study_component"] == PRIMARY_COMPONENT]
    channel = [row for row in master if row["study_component"] == CHANNEL_COMPONENT]
    assert len(primary) == 42
    assert len(channel) == 56
    assert {row["validation_scope"] for row in primary} == {PRIMARY_VALIDATION_SCOPE}
    assert {row["validation_scope"] for row in channel} == {CHANNEL_VALIDATION_SCOPE}
    assert all(row["dataset"] == "JANON" and row["target_text"] in {"うっとうしい", "がっしり", "さっさと", "ばっちり", "オイル", "バグ", "酸味"} for row in primary)
    assert all(row["dataset"] == "JVS" and row["pair_id"] for row in channel)


def test_blind_files_do_not_leak_hidden_source_or_condition_fields():
    rows = _rows("pronunciation_listener_blind_v1.csv")
    forbidden = {"speaker_group_hidden", "dataset", "condition", "wavlm_layer12", "wavlm_layer24", "wavlm_median_index", "alignment_available", "recording_quality", "audio_path", "study_component", "validation_scope", "future_mapping_research_eligible"}
    assert rows
    assert not (forbidden & set(rows[0]))
    assert all(row["sample_id"].startswith("clip_") for row in rows)
    assert all(row["audio_asset_id"].startswith("asset_") for row in rows)
    serialized = "\n".join(",".join(row.values()) for row in rows)
    assert not any(token in serialized for token in ("learner_", "anchor_", "channel_pair", "_clean", "_codec", "_rir", "noise_15db"))
    assignments = _rows("listener_assignment_v1.csv")
    assignment_serialized = "\n".join(",".join(row.values()) for row in assignments)
    assert not any(token in assignment_serialized for token in ("learner_", "anchor_", "channel_pair", "_clean", "_codec", "_rir", "noise_15db"))


def test_each_pilot_clip_has_at_least_five_assignment_slots():
    master = _rows("pronunciation_listener_manifest_v1.csv")
    assigned = Counter(row["sample_id"] for row in _rows("listener_assignment_v1.csv"))
    assert all(assigned[row["blind_sample_id"]] >= 5 for row in master)


def test_channel_set_requires_all_four_conditions():
    master = _rows("pronunciation_listener_manifest_v1.csv")
    complete = complete_channel_set_ids(master)
    assert len(complete) == 14
    assert len(complete) < FINAL_MIN_CHANNEL_SETS
    incomplete_pair = next(iter(complete))
    missing_codec = [row for row in master if not (row["pair_id"] == incomplete_pair and row["condition"] == "codec")]
    assert incomplete_pair not in complete_channel_set_ids(missing_codec)


def test_mapping_eligibility_excludes_all_channel_rows():
    master = _rows("pronunciation_listener_manifest_v1.csv")
    mapping_rows = primary_mapping_rows(master)
    assert len(mapping_rows) == 42
    assert all(row["study_component"] == PRIMARY_COMPONENT for row in mapping_rows)
    assert all(row["future_mapping_research_eligible"] == "true" for row in mapping_rows)
    assert not any(row["sample_id"].startswith("channel_pair") for row in mapping_rows)


def test_channel_set_members_and_duplicates_are_not_adjacent_in_pilot_schedule():
    pair_for_sample = {row["blind_sample_id"]: row["pair_id"] for row in _rows("pronunciation_listener_manifest_v1.csv")}
    per_listener: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _rows("listener_assignment_v1.csv"):
        per_listener[row["listener_slot_id"]].append(row)
    for schedule in per_listener.values():
        ordered = sorted(schedule, key=lambda row: int(row["display_order"]))
        for before, after in zip(ordered, ordered[1:]):
            pair = pair_for_sample[before["sample_id"]]
            assert not pair or pair != pair_for_sample[after["sample_id"]]
            assert before["sample_id"] != after["sample_id"]


def test_pilot_duplicates_are_explicitly_insufficient_but_final_plan_requires_five_per_rater():
    pilot_duplicates = [row for row in _rows("listener_assignment_v1.csv") if row["is_duplicate"] == "true"]
    assert len(pilot_duplicates) == 10
    assert all(row["duplicate_of_sample_id"] == row["sample_id"] for row in pilot_duplicates)
    plan = json.loads((FINAL_TEMPLATE / "final_assignment_plan.json").read_text(encoding="utf-8"))
    assert plan["minimum_hidden_duplicates_per_rater"] >= 5
    assert plan["study_stage"] == FINAL_STUDY_STAGE
    master = _rows("pronunciation_listener_manifest_v1.csv")
    formal_algorithm_only = assignments(master, hidden_duplicates_per_rater=5, study_stage=FINAL_STUDY_STAGE)
    per_listener = Counter(row["listener_slot_id"] for row in formal_algorithm_only if row["is_duplicate"] == "true")
    assert set(per_listener.values()) == {5}
    assert {row["study_stage"] for row in formal_algorithm_only} == {FINAL_STUDY_STAGE}
    scheduled: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in formal_algorithm_only:
        scheduled[row["listener_slot_id"]].append(row)
    for rows in scheduled.values():
        ordered = sorted(rows, key=lambda row: int(row["display_order"]))
        assert all(before["sample_id"] != after["sample_id"] for before, after in zip(ordered, ordered[1:]))


def test_final_assignment_refuses_missing_real_learner_audio():
    seed = _rows("pronunciation_listener_manifest_v1.csv")
    with pytest.raises(ValueError, match="final assignment blocked"):
        final_assignment_from_real_manifest(seed)
    with pytest.raises(ValueError, match="primary isolated-word cohort is incomplete"):
        final_assignment_from_real_manifest([{**row, "study_stage": FINAL_STUDY_STAGE} for row in seed])
    template_rows = _rows("pronunciation_listener_manifest_final_template.csv", directory=FINAL_TEMPLATE)
    assert template_rows == []


def test_final_projection_requires_separate_primary_and_channel_totals():
    metadata = json.loads((FINAL_TEMPLATE / "final_study_template_metadata.json").read_text(encoding="utf-8"))
    assert metadata["projected_primary_unique_clip_count"] == FINAL_PRIMARY_CLIP_COUNT == 98
    assert metadata["projected_channel_unique_clip_count"] == FINAL_CHANNEL_CLIP_COUNT == 60
    assert metadata["projected_total_unique_clip_count"] == 158
    assert metadata["projected_base_ratings"] == 790
    assert metadata["projected_hidden_repeat_presentations"] == 50
    assert metadata["projected_total_presentations"] == 840
    assert metadata["required_complete_channel_sets_for_promotion"] >= 15
    assert FINAL_NATIVE_CLIP_COUNT == 28


def test_rating_contract_and_json_schema_keep_analyzability_separate():
    contract = json.loads((PILOT / "human_rating_schema.json").read_text(encoding="utf-8"))
    machine_schema = json.loads((PILOT / "human_rating_json_schema_v1.json").read_text(encoding="utf-8"))
    assert contract["schema_kind"] == "study_contract"
    assert contract["study_stage"] == PILOT_STUDY_STAGE
    assert contract["primary_construct"] == "pronunciation_accuracy"
    assert "null" in contract["fields"]["pronunciation_accuracy_1to7"]["type"]
    assert any("analyzable_yes_no is no" in rule for rule in contract["conditional_rules"])
    assert machine_schema["$schema"].endswith("2020-12/schema")
    assert len(machine_schema["allOf"]) == 2


def test_calibration_scope_is_not_generic_and_mapping_stays_disabled():
    calibration = default_pronunciation_calibration()
    assert calibration.valid_target_scope == "janon_7target_isolated_word_validation_v1"
    assert "same_target_multi_reference_only" != calibration.valid_target_scope
    assert calibration.production_enabled is False
    assert calibration.mapping is None
    assert calibration.mapping_version is None
    assert not hasattr(calibration, "map_score")
