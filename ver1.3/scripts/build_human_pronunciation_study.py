"""Construct blinded listener-study artifacts without scoring or mapping audio.

The script consumes frozen v3.2 audit metadata when it exists.  It never loads
WavLM, evaluates ProductScore, changes a distance, or creates synthetic speech.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/Users/ryukayuiii/Documents/jp_speech_eval_project")
OUT = ROOT / "data/human_eval"
PILOT_OUT = OUT / "pilot"
FINAL_TEMPLATE_OUT = OUT / "final_template"
TARGETS = ("うっとうしい", "がっしり", "さっさと", "ばっちり", "オイル", "バグ", "酸味")
CHANNEL_CONDITIONS = ("clean", "rir", "noise_15db", "codec")
LISTENER_SLOTS = tuple(f"LS{i:02d}" for i in range(1, 11))
RATINGS_PER_CLIP = 5
RANDOM_SEED = 33017
PILOT_STUDY_STAGE = "pilot_seed"
FINAL_STUDY_STAGE = "final_calibration"
FINAL_MIN_LEARNER_CLIPS = 70
FINAL_MIN_LEARNER_SPEAKERS = 10
FINAL_MIN_CHANNEL_SETS = 15
FINAL_NATIVE_CLIP_COUNT = 28
PRIMARY_COMPONENT = "primary_pronunciation_calibration"
CHANNEL_COMPONENT = "channel_bias_control"
PRIMARY_VALIDATION_SCOPE = "janon_7target_isolated_word_validation_v1"
CHANNEL_VALIDATION_SCOPE = "jvs_channel_bias_long_sentence_v1"
FINAL_PRIMARY_CLIP_COUNT = 98
FINAL_CHANNEL_CLIP_COUNT = FINAL_MIN_CHANNEL_SETS * len(CHANNEL_CONDITIONS)
FINAL_PROJECTED_CLIP_COUNT = FINAL_PRIMARY_CLIP_COUNT + FINAL_CHANNEL_CLIP_COUNT


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        names = fieldnames or (list(rows[0]) if rows else [])
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        if names:
            writer.writeheader()
            writer.writerows(rows)


def asset_id(sample_id: str) -> str:
    return "asset_" + hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]


def blind_sample_id(sample_id: str) -> str:
    return "clip_" + hashlib.sha256(("listener-v1:" + sample_id).encode("utf-8")).hexdigest()[:16]


def target_id(target: str) -> str:
    return f"target_{TARGETS.index(target) + 1:02d}"


def _frozen_ssl() -> dict[tuple[str, str, str], dict[str, str]]:
    """Read existing v3.2 evidence only; missing values remain explicitly null."""
    path = ROOT / "outputs/product_score_v32/wavlm_7target_raw.csv"
    if not path.exists():
        return {}
    return {
        (row["target_text"], row["relation"], row["user_speaker"]): row
        for row in read_csv(path)
        if row.get("aggregation") == "median"
    }


def _ssl_fields(row: dict[str, str] | None) -> dict[str, Any]:
    if row is None:
        return {
            "wavlm_layer12": "",
            "wavlm_layer24": "",
            "wavlm_median_index": "",
            "wavlm_status": "pending_frozen_baseline_extraction",
        }
    layer12, layer24 = float(row["raw_distance_layer12"]), float(row["raw_distance_layer24"])
    return {
        "wavlm_layer12": round(layer12, 6),
        "wavlm_layer24": round(layer24, 6),
        "wavlm_median_index": round((layer12 + layer24) / 2.0, 6),
        "wavlm_status": "frozen_v32_median_reference_evidence",
    }


def _native_and_learner_rows() -> list[dict[str, Any]]:
    references = read_csv(ROOT / "data/audit/native_reference_bank.csv")
    frozen_ssl = _frozen_ssl()
    rows: list[dict[str, Any]] = []
    for index, reference in enumerate(references, start=1):
        target = reference["target_text"]
        speaker = Path(reference["wav_path"]).parts[-3]
        sample_id = f"anchor_{target_id(target)}_{speaker}"
        rows.append({
            "sample_id": sample_id,
            "pair_id": "",
            "target_id": target_id(target),
            "target_text": target,
            "normalized_kana": reference["normalized_kana"],
            "speaker_id_anonymized": f"N{index:03d}",
            "speaker_group_hidden": "native_anchor",
            "dataset": "JANON",
            "audio_path": reference["wav_path"],
            "condition": "clean",
            "is_clean": "true",
            "reference_bank_id": "wavlm_7target_human_native_v1",
            "alignment_available": "not_required_for_listener_rating",
            "recording_quality": "pending_pre_rating_qc",
            "study_component": PRIMARY_COMPONENT,
            "validation_scope": PRIMARY_VALIDATION_SCOPE,
            "future_mapping_research_eligible": "true",
            **_ssl_fields(frozen_ssl.get((target, "native_loo", speaker))),
        })

    janon = read_csv(DATA_ROOT / "JANON/data.csv")
    learner_rows = [
        row for row in janon
        if row.get("Speaker") in {"chf1", "enf1"} and row.get("Stmiulus") in TARGETS and row.get("Stimulus Type") == "isolated"
    ]
    for index, learner in enumerate(sorted(learner_rows, key=lambda row: (row["Speaker"], TARGETS.index(row["Stmiulus"]))), start=1):
        target, speaker = learner["Stmiulus"], learner["Speaker"]
        rows.append({
            "sample_id": f"learner_{target_id(target)}_{speaker}",
            "pair_id": "",
            "target_id": target_id(target),
            "target_text": target,
            "normalized_kana": next(row["normalized_kana"] for row in references if row["target_text"] == target),
            "speaker_id_anonymized": f"L{1 if speaker == 'chf1' else 2:03d}",
            "speaker_group_hidden": "learner_candidate",
            "dataset": "JANON",
            "audio_path": str(Path("JANON") / learner["Path"]),
            "condition": "clean",
            "is_clean": "true",
            "reference_bank_id": "wavlm_7target_human_native_v1",
            "alignment_available": "not_required_for_listener_rating",
            "recording_quality": "pending_pre_rating_qc",
            "study_component": PRIMARY_COMPONENT,
            "validation_scope": PRIMARY_VALIDATION_SCOPE,
            "future_mapping_research_eligible": "true",
            **_ssl_fields(frozen_ssl.get((target, "learner", speaker))),
        })
    return rows


def _channel_rows() -> list[dict[str, Any]]:
    source = DATA_ROOT / "ver1.3/reports/tier0_batch3_audio"
    controls = {
        "rir": {path.name.split("_ref-")[0]: path for path in (source / "C_channel").glob("*_rir_mild.wav")},
        "noise_15db": {path.name.split("_ref-")[0]: path for path in (source / "C_channel").glob("*_mild_noise_15db.wav")},
        "codec": {path.name.split("_ref-")[0]: path for path in (source / "B1_pronunciation").glob("*_codec_control.wav")},
    }
    common = sorted(set(controls["rir"]) & set(controls["noise_15db"]) & set(controls["codec"]))
    rows: list[dict[str, Any]] = []
    texts = {
        "VOICEACTRESS100_001": "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。",
        "VOICEACTRESS100_002": "ニューイングランド風は、牛乳をベースとした、白いクリームスープであり、ボストンクラムチャウダーとも呼ばれる。",
    }
    readings = {
        "VOICEACTRESS100_001": "マタ、トージノヨーニ、ゴダイミョーオートヨバレル、シュヨーナミョーオーノチューオーニハイサレルコトモオーイ。",
        "VOICEACTRESS100_002": "ニューイングランドフーワ、ギューニューヲベーストシタ、シロイクリームスープデアリ、ボストンクラムチャウダートモヨバレル。",
    }
    for pair_index, key in enumerate(common, start=1):
        speaker, utterance = key.split("_", 1)
        clean = DATA_ROOT / "JVS" / speaker / "parallel100/wav24kHz16bit" / f"{utterance}.wav"
        if not clean.exists():
            raise FileNotFoundError(clean)
        pair_id = f"channel_pair_{pair_index:02d}"
        for condition in CHANNEL_CONDITIONS:
            audio = clean if condition == "clean" else controls[condition][key]
            if not audio.exists():
                raise FileNotFoundError(audio)
            rows.append({
                "sample_id": f"{pair_id}_{condition}",
                "pair_id": pair_id,
                "target_id": utterance,
                "target_text": texts[utterance],
                "normalized_kana": readings[utterance],
                "speaker_id_anonymized": f"C{pair_index:03d}",
                "speaker_group_hidden": "channel_control",
                "dataset": "JVS",
                "audio_path": str(audio.relative_to(DATA_ROOT)),
                "condition": condition,
                "is_clean": str(condition == "clean").lower(),
                "reference_bank_id": "not_applicable_channel_bias_subset",
                "alignment_available": "not_evaluated_in_manifest_construction",
                "recording_quality": "pending_pre_rating_qc",
                "study_component": CHANNEL_COMPONENT,
                "validation_scope": CHANNEL_VALIDATION_SCOPE,
                "future_mapping_research_eligible": "false",
                **_ssl_fields(None),
            })
    return rows


def master_manifest() -> list[dict[str, Any]]:
    rows = _native_and_learner_rows() + _channel_rows()
    if len(rows) != len({row["sample_id"] for row in rows}):
        raise ValueError("sample_id must be unique")
    for row in rows:
        if not (DATA_ROOT / row["audio_path"]).exists():
            raise FileNotFoundError(row["audio_path"])
        row["blind_sample_id"] = blind_sample_id(row["sample_id"])
        row["audio_asset_id"] = asset_id(row["sample_id"])
        row["study_stage"] = PILOT_STUDY_STAGE
    return rows


def blind_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "study_stage": PILOT_STUDY_STAGE,
        "sample_id": row["blind_sample_id"],
        "audio_asset_id": row["audio_asset_id"],
        "target_id": row["target_id"],
        "target_text": row["target_text"],
        "normalized_kana": row["normalized_kana"],
        "rater_instruction_version": "pronunciation_accuracy_ja_v1",
    } for row in rows]


def _schedule(items: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    """Order a listener's clips without adjacent channel-pair members."""
    remaining = list(items)
    rng.shuffle(remaining)
    ordered: list[dict[str, Any]] = []
    while remaining:
        previous = ordered[-1] if ordered else None
        candidates = [
            item for item in remaining
            if not previous or (item["pair_id"] != previous["pair_id"] and item["sample_id"] != previous["sample_id"])
        ]
        if not candidates:
            candidates = [item for item in remaining if item["sample_id"] != previous["sample_id"]] or remaining
        target_counts = defaultdict(int)
        for item in ordered[-4:]:
            target_counts[item["target_id"]] += 1
        least_recent = min(target_counts.get(item["target_id"], 0) for item in candidates)
        candidates = [item for item in candidates if target_counts.get(item["target_id"], 0) == least_recent]
        chosen = rng.choice(candidates)
        ordered.append(chosen)
        remaining.remove(chosen)
    return ordered


def assignments(rows: list[dict[str, Any]], *, hidden_duplicates_per_rater: int = 1, study_stage: str = PILOT_STUDY_STAGE) -> list[dict[str, Any]]:
    """Balanced ten-slot plan with non-adjacent hidden repeat presentations."""
    if hidden_duplicates_per_rater < 1:
        raise ValueError("at least one duplicate is required for a study assignment")
    by_slot: dict[str, list[dict[str, Any]]] = {slot: [] for slot in LISTENER_SLOTS}
    for index, row in enumerate(rows):
        for offset in range(RATINGS_PER_CLIP):
            by_slot[LISTENER_SLOTS[(index + offset) % len(LISTENER_SLOTS)]].append(row)
    rng = random.Random(RANDOM_SEED)
    output: list[dict[str, Any]] = []
    for slot_index, slot in enumerate(LISTENER_SLOTS):
        assigned = list(by_slot[slot])
        selected: list[dict[str, Any]] = []
        groups = ("native_anchor", "learner_candidate", "channel_control")
        for duplicate_index in range(hidden_duplicates_per_rater):
            preferred_group = groups[duplicate_index % len(groups)]
            candidates = [row for row in assigned if row["speaker_group_hidden"] == preferred_group and row not in selected]
            candidates = candidates or [row for row in assigned if row not in selected]
            duplicate = candidates[(slot_index * 7 + duplicate_index * 11) % len(candidates)]
            selected.append(duplicate)
            assigned.append({**duplicate, "duplicate_of_sample_id": duplicate["sample_id"]})
        for order, row in enumerate(_schedule(assigned, rng), start=1):
            duplicate_of = row.get("duplicate_of_sample_id", "")
            presentation_id = f"{slot}_P{order:03d}"
            output.append({
                "study_stage": study_stage,
                "assignment_id": f"A_{presentation_id}",
                "listener_slot_id": slot,
                "presentation_id": presentation_id,
                "sample_id": row["blind_sample_id"],
                "audio_asset_id": row["audio_asset_id"],
                "display_order": order,
                "target_id": row["target_id"],
                "target_text": row["target_text"],
                "normalized_kana": row["normalized_kana"],
                "is_duplicate": str(bool(duplicate_of)).lower(),
                "duplicate_of_sample_id": blind_sample_id(duplicate_of) if duplicate_of else "",
            })
    return output


def complete_channel_set_ids(rows: list[dict[str, Any]]) -> set[str]:
    """Return only channel sets that contain the entire four-condition quartet."""
    conditions_by_pair: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("study_component") == CHANNEL_COMPONENT and row.get("pair_id"):
            conditions_by_pair[row["pair_id"]].add(str(row.get("condition", "")))
    expected = set(CHANNEL_CONDITIONS)
    return {pair_id for pair_id, conditions in conditions_by_pair.items() if conditions == expected}


def primary_mapping_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the only rows eligible for future calibration/mapping research.

    Channel-bias controls are deliberately excluded even though they may be
    rated: their long-sentence task is a robustness control, not construct
    validation for the seven isolated JANON targets.
    """
    return [
        row for row in rows
        if row.get("study_component") == PRIMARY_COMPONENT
        and row.get("validation_scope") == PRIMARY_VALIDATION_SCOPE
        and row.get("future_mapping_research_eligible") == "true"
    ]


def final_assignment_from_real_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build final assignment only after all required real learner audio exists.

    This guard deliberately refuses the pilot/seed rows.  It prevents a
    template or recruitment target from being mistaken for a final cohort.
    """
    if any(row.get("study_stage") != FINAL_STUDY_STAGE for row in rows):
        raise ValueError("final assignment blocked: manifest is not marked final_calibration")
    primary = primary_mapping_rows(rows)
    if len(primary) != FINAL_PRIMARY_CLIP_COUNT:
        raise ValueError("final assignment blocked: primary isolated-word cohort is incomplete")
    learners = [row for row in primary if row.get("speaker_group_hidden") == "learner_candidate" and row.get("is_clean") == "true"]
    native_anchors = [row for row in primary if row.get("speaker_group_hidden") == "native_anchor" and row.get("is_clean") == "true"]
    speakers = {row.get("speaker_id_anonymized") for row in learners}
    missing_audio = [row.get("sample_id", "") for row in learners if not row.get("audio_path") or not (DATA_ROOT / str(row["audio_path"])).exists()]
    if len(native_anchors) != FINAL_NATIVE_CLIP_COUNT or len(learners) != FINAL_MIN_LEARNER_CLIPS or len(speakers) < FINAL_MIN_LEARNER_SPEAKERS or missing_audio:
        raise ValueError("final assignment blocked: required real learner audio cohort is incomplete")
    if len(complete_channel_set_ids(rows)) < FINAL_MIN_CHANNEL_SETS:
        raise ValueError("final assignment blocked: fewer than 15 complete channel sets")
    return assignments(rows, hidden_duplicates_per_rater=5, study_stage=FINAL_STUDY_STAGE)


def learner_recording_needed() -> list[dict[str, Any]]:
    """Recruit eight new learners × seven targets: 56, yielding 70 with current 14."""
    kana = {row["target_text"]: row["normalized_kana"] for row in read_csv(ROOT / "data/audit/native_reference_bank.csv")}
    return [{
        "recruitment_id": f"REC{speaker_index:02d}_{target_id(target)}",
        "planned_speaker_id_anonymized": f"REC{speaker_index:02d}",
        "target_id": target_id(target),
        "target_text": target,
        "normalized_kana": kana[target],
        "required_recording": "one_clean_real_learner_recording",
        "required_metadata": "self_reported_native_language,japanese_proficiency,recording_device",
        "study_component": PRIMARY_COMPONENT,
        "validation_scope": PRIMARY_VALIDATION_SCOPE,
        "status": "needed",
    } for speaker_index in range(1, 9) for target in TARGETS]


def schema() -> dict[str, Any]:
    return {
        "schema_kind": "study_contract",
        "schema_version": "human_pronunciation_rating_v1",
        "study_stage": PILOT_STUDY_STAGE,
        "primary_construct": "pronunciation_accuracy",
        "primary_instruction_ja": "画面に示された語を基準として、発音そのものがどの程度正確に実現されているかを評価してください。話す速さ、声の高さ、感情表現、録音音質は、可能な限り評価に含めないでください。",
        "accuracy_scale": {str(score): label for score, label in enumerate(["非常に不正確", "不正確", "やや不正確", "中程度", "やや正確", "正確", "非常に正確"], start=1)},
        "fields": {
            "rater_id": {"type": "string", "required": True, "anonymized": True},
            "sample_id": {"type": "string", "required": True},
            "presentation_id": {"type": "string", "required": True},
            "pronunciation_accuracy_1to7": {"type": ["integer", "null"], "minimum": 1, "maximum": 7, "required": True},
            "analyzable_yes_no": {"type": "string", "enum": ["yes", "no"], "required": True},
            "optional_comment": {"type": ["string", "null"], "required": False},
            "timestamp": {"type": "string", "format": "date-time", "required": True},
        },
        "conditional_rules": [
            "If analyzable_yes_no is no, pronunciation_accuracy_1to7 may be null.",
            "If analyzable_yes_no is yes, pronunciation_accuracy_1to7 must be an integer from 1 through 7.",
        ],
        "rater_background_fields": ["rater_id", "native_language", "japanese_proficiency", "phonetics_or_speech_training_experience"],
        "secondary_constructs": {"comprehensibility": "not collected in v1; do not average with pronunciation_accuracy"},
    }


def formal_json_schema() -> dict[str, Any]:
    """Machine-validatable companion to the human-readable study contract."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Human pronunciation accuracy rating v1",
        "type": "object",
        "required": ["rater_id", "sample_id", "presentation_id", "analyzable_yes_no", "pronunciation_accuracy_1to7", "timestamp"],
        "properties": {
            "rater_id": {"type": "string", "minLength": 1},
            "sample_id": {"type": "string", "pattern": "^clip_"},
            "presentation_id": {"type": "string", "minLength": 1},
            "analyzable_yes_no": {"enum": ["yes", "no"]},
            "pronunciation_accuracy_1to7": {"type": ["integer", "null"], "minimum": 1, "maximum": 7},
            "optional_comment": {"type": ["string", "null"]},
            "timestamp": {"type": "string", "format": "date-time"},
        },
        "allOf": [{
            "if": {"properties": {"analyzable_yes_no": {"const": "yes"}}, "required": ["analyzable_yes_no"]},
            "then": {"properties": {"pronunciation_accuracy_1to7": {"type": "integer", "minimum": 1, "maximum": 7}}, "required": ["pronunciation_accuracy_1to7"]},
        }, {
            "if": {"properties": {"analyzable_yes_no": {"const": "no"}}, "required": ["analyzable_yes_no"]},
            "then": {"properties": {"pronunciation_accuracy_1to7": {"type": ["integer", "null"], "minimum": 1, "maximum": 7}}},
        }],
    }


def final_template(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    fields = list(rows[0]) + ["learner_audio_present"]
    metadata = {
        "study_stage": "final_template_unpopulated",
        "assignment_generated": False,
        "rating_collection_started": False,
        "study_components": {
            PRIMARY_COMPONENT: {
                "validation_scope": PRIMARY_VALIDATION_SCOPE,
                "projected_clean_isolated_word_clips": FINAL_PRIMARY_CLIP_COUNT,
                "current_seed_clips": len(primary_mapping_rows(rows)),
            },
            CHANNEL_COMPONENT: {
                "validation_scope": CHANNEL_VALIDATION_SCOPE,
                "projected_minimum_long_sentence_clips": FINAL_CHANNEL_CLIP_COUNT,
                "current_seed_clips": len([row for row in rows if row.get("study_component") == CHANNEL_COMPONENT]),
            },
        },
        "projected_primary_unique_clip_count": FINAL_PRIMARY_CLIP_COUNT,
        "projected_channel_unique_clip_count": FINAL_CHANNEL_CLIP_COUNT,
        "projected_total_unique_clip_count": FINAL_PROJECTED_CLIP_COUNT,
        "projected_base_ratings": FINAL_PROJECTED_CLIP_COUNT * RATINGS_PER_CLIP,
        "projected_hidden_repeat_presentations": len(LISTENER_SLOTS) * 5,
        "projected_total_presentations": FINAL_PROJECTED_CLIP_COUNT * RATINGS_PER_CLIP + len(LISTENER_SLOTS) * 5,
        "required_new_real_learner_clips": 56,
        "required_total_learner_clean_clips": FINAL_MIN_LEARNER_CLIPS,
        "required_total_learner_speakers": FINAL_MIN_LEARNER_SPEAKERS,
        "required_complete_channel_sets_for_promotion": FINAL_MIN_CHANNEL_SETS,
        "current_complete_channel_sets": len(complete_channel_set_ids(rows)),
        "generation_rule": "Populate only after each new learner audio path exists, source metadata is verified, and the fifteenth real channel quartet exists.",
    }
    assignment_plan = {
        "study_stage": FINAL_STUDY_STAGE,
        "assignment_generated": False,
        "ratings_per_clip": RATINGS_PER_CLIP,
        "minimum_hidden_duplicates_per_rater": 5,
        "preferred_hidden_duplicate_fraction_of_workload": "0.05_to_0.10",
        "duplicate_constraints": ["different_presentation_id", "same_underlying_sample", "not_adjacent", "not_disclosed", "cross_target_when_possible", "include_native_learner_and_channel_samples"],
        "generation_guard": "final_assignment_from_real_manifest requires a 98-clip primary isolated-word cohort (70 real learner clean clips across 10 speakers) and 15 complete long-sentence channel quartets",
    }
    return fields, metadata, assignment_plan


def main() -> None:
    rows = master_manifest()
    write_csv(PILOT_OUT / "pronunciation_listener_manifest_v1.csv", rows)
    write_csv(PILOT_OUT / "pronunciation_listener_blind_v1.csv", blind_manifest(rows))
    write_csv(PILOT_OUT / "listener_assignment_v1.csv", assignments(rows))
    (PILOT_OUT / "human_rating_schema.json").parent.mkdir(parents=True, exist_ok=True)
    (PILOT_OUT / "human_rating_schema.json").write_text(json.dumps(schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (PILOT_OUT / "human_rating_json_schema_v1.json").write_text(json.dumps(formal_json_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields, metadata, assignment_plan = final_template(rows)
    write_csv(FINAL_TEMPLATE_OUT / "pronunciation_listener_manifest_final_template.csv", [], fieldnames=fields)
    write_csv(FINAL_TEMPLATE_OUT / "learner_recording_needed.csv", learner_recording_needed())
    (FINAL_TEMPLATE_OUT / "final_study_template_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (FINAL_TEMPLATE_OUT / "final_assignment_plan.json").write_text(json.dumps(assignment_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    primary_rows = primary_mapping_rows(rows)
    channel_rows = [row for row in rows if row.get("study_component") == CHANNEL_COMPONENT]
    (PILOT_OUT / "pilot_study_metadata.json").write_text(json.dumps({
        "study_stage": PILOT_STUDY_STAGE,
        "rating_collection_started": False,
        "unique_clip_count": len(rows),
        "study_components": {
            PRIMARY_COMPONENT: {
                "validation_scope": PRIMARY_VALIDATION_SCOPE,
                "clip_count": len(primary_rows),
                "description": "JANON seven-target isolated-word primary calibration seed",
            },
            CHANNEL_COMPONENT: {
                "validation_scope": CHANNEL_VALIDATION_SCOPE,
                "clip_count": len(channel_rows),
                "description": "JVS long-sentence channel-bias control seed",
            },
        },
        "analysis_rule": "Analyze primary pronunciation calibration and channel-bias control separately; do not report a mixed global WavLM-human correlation.",
        "base_ratings_per_clip": RATINGS_PER_CLIP,
        "hidden_duplicates_per_listener_slot": 1,
        "repeatability_status": "insufficient_for_stable_per_rater_repeatability",
        "complete_channel_sets": len(complete_channel_set_ids(rows)),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
