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
TARGETS = ("うっとうしい", "がっしり", "さっさと", "ばっちり", "オイル", "バグ", "酸味")
CHANNEL_CONDITIONS = ("clean", "rir", "noise_15db", "codec")
LISTENER_SLOTS = tuple(f"LS{i:02d}" for i in range(1, 11))
RATINGS_PER_CLIP = 5
RANDOM_SEED = 33017


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [], lineterminator="\n")
        if rows:
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
    return rows


def blind_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
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


def assignments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Balanced ten-slot plan: five ratings per clip plus one duplicate per slot."""
    by_slot: dict[str, list[dict[str, Any]]] = {slot: [] for slot in LISTENER_SLOTS}
    for index, row in enumerate(rows):
        for offset in range(RATINGS_PER_CLIP):
            by_slot[LISTENER_SLOTS[(index + offset) % len(LISTENER_SLOTS)]].append(row)
    rng = random.Random(RANDOM_SEED)
    output: list[dict[str, Any]] = []
    for slot_index, slot in enumerate(LISTENER_SLOTS):
        assigned = list(by_slot[slot])
        duplicate = assigned[(slot_index * 7) % len(assigned)]
        assigned.append({**duplicate, "duplicate_of_sample_id": duplicate["sample_id"]})
        for order, row in enumerate(_schedule(assigned, rng), start=1):
            duplicate_of = row.get("duplicate_of_sample_id", "")
            presentation_id = f"{slot}_P{order:03d}"
            output.append({
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
        "status": "needed",
    } for speaker_index in range(1, 9) for target in TARGETS]


def schema() -> dict[str, Any]:
    return {
        "schema_version": "human_pronunciation_rating_v1",
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


def main() -> None:
    rows = master_manifest()
    write_csv(OUT / "pronunciation_listener_manifest_v1.csv", rows)
    write_csv(OUT / "pronunciation_listener_blind_v1.csv", blind_manifest(rows))
    write_csv(OUT / "listener_assignment_v1.csv", assignments(rows))
    write_csv(OUT / "learner_recording_needed.csv", learner_recording_needed())
    (OUT / "human_rating_schema.json").write_text(json.dumps(schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
