from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from jp_speech_eval.pronunciation_calibration import default_pronunciation_calibration


ROOT = Path(__file__).resolve().parents[1]
HUMAN_EVAL = ROOT / "data/human_eval"


def _rows(name: str) -> list[dict[str, str]]:
    with (HUMAN_EVAL / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_blind_manifest_does_not_leak_hidden_source_or_condition_fields():
    rows = _rows("pronunciation_listener_blind_v1.csv")
    forbidden = {"speaker_group_hidden", "dataset", "condition", "wavlm_layer12", "wavlm_layer24", "wavlm_median_index", "alignment_available", "recording_quality", "audio_path"}
    assert rows
    assert not (forbidden & set(rows[0]))
    assert all(row["sample_id"].startswith("clip_") for row in rows)
    assert all(row["audio_asset_id"].startswith("asset_") for row in rows)
    serialized = "\n".join(",".join(row.values()) for row in rows)
    assert not any(token in serialized for token in ("learner_", "anchor_", "channel_pair", "_clean", "_codec", "_rir", "noise_15db"))
    assignments = _rows("listener_assignment_v1.csv")
    assignment_serialized = "\n".join(",".join(row.values()) for row in assignments)
    assert not any(token in assignment_serialized for token in ("learner_", "anchor_", "channel_pair", "_clean", "_codec", "_rir", "noise_15db"))


def test_each_master_clip_has_at_least_five_assignment_slots():
    master = _rows("pronunciation_listener_manifest_v1.csv")
    assigned = Counter(row["sample_id"] for row in _rows("listener_assignment_v1.csv"))
    assert all(assigned[row["blind_sample_id"]] >= 5 for row in master)


def test_channel_pair_members_are_not_adjacent_in_a_listener_schedule():
    pair_for_sample = {row["blind_sample_id"]: row["pair_id"] for row in _rows("pronunciation_listener_manifest_v1.csv")}
    per_listener: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _rows("listener_assignment_v1.csv"):
        per_listener[row["listener_slot_id"]].append(row)
    for schedule in per_listener.values():
        ordered = sorted(schedule, key=lambda row: int(row["display_order"]))
        for before, after in zip(ordered, ordered[1:]):
            pair = pair_for_sample[before["sample_id"]]
            assert not pair or pair != pair_for_sample[after["sample_id"]]


def test_assignment_has_hidden_duplicate_presentations_for_intra_rater_qc():
    rows = _rows("listener_assignment_v1.csv")
    duplicates = [row for row in rows if row["is_duplicate"] == "true"]
    assert len(duplicates) == 10
    assert all(row["duplicate_of_sample_id"] == row["sample_id"] for row in duplicates)


def test_rating_schema_allows_null_accuracy_only_when_unanalyzable():
    schema = json.loads((HUMAN_EVAL / "human_rating_schema.json").read_text(encoding="utf-8"))
    assert "null" in schema["fields"]["pronunciation_accuracy_1to7"]["type"]
    assert any("analyzable_yes_no is no" in rule for rule in schema["conditional_rules"])
    assert schema["primary_construct"] == "pronunciation_accuracy"


def test_production_calibration_is_disabled_and_has_no_score_mapping():
    calibration = default_pronunciation_calibration()
    assert calibration.production_enabled is False
    assert calibration.mapping is None
    assert calibration.mapping_version is None
    assert not hasattr(calibration, "map_score")
