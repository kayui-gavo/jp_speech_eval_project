"""Run the C-end v2 acceptance matrix against labelled, existing audio.

The script deliberately keeps data discovery/preparation separate from scoring so
the baseline and candidate can consume the exact same WAV and cache files.
It is an audit utility; it does not alter production scoring configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


JVS_SENTENCES = {
    1: "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。",
    2: "ニューイングランド風は、牛乳をベースとした、白いクリームスープであり、ボストンクラムチャウダーとも呼ばれる。",
    3: "コンピュータゲームのメーカーや、業界団体などに関連する人物のカテゴリ。",
    4: "サービスマネージャー導入駅のため、大井町駅から、遠隔管理している。",
    5: "シルバーサーファー襲撃事件までに、リチャーズは、チーム名と共に、国際的にスーパーヒーロー、および、有名人として、認知されている。",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_jvs_cache(cache_root: Path, sentence_id: int, reference_id: str = "jvs001") -> Path:
    sentence = f"VOICEACTRESS100_{sentence_id:03d}"
    candidates = sorted((cache_root / sentence).glob(f"ref_{reference_id}_*.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one cache for {sentence}/{reference_id}, found {candidates}")
    return candidates[0].with_suffix("")


def _prepare_janon_cache(text: str, reference_wav: Path, out: Path, reference_id: str) -> Path:
    from jp_speech_eval.sentence_cache import build_sentence_cache

    if not out.with_suffix(".json").exists() or not out.with_suffix(".npz").exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        build_sentence_cache(
            text,
            out,
            sr=16000,
            save_reference_wav=True,
            reference_wav_path=reference_wav,
            reference_source="janon_japanese_native_external_reference_wav",
            reference_id=reference_id,
        )
    return out


def build_inventory(data_root: Path, output_root: Path) -> list[dict[str, Any]]:
    """Build a fixed manifest from existing, labelled JVS/JANON/demo assets."""

    jvs_root = data_root / "JVS"
    janon_root = data_root / "JANON"
    version_root = data_root / "ver1.3"
    cache_root = version_root / "reports" / "tier0_batch3_reference_cache"
    rows: list[dict[str, Any]] = []

    def add(
        sample_id: str,
        wav: Path,
        source: str,
        speaker: str,
        category: str,
        target: str,
        cache: Path | None,
        *,
        mode: str = "reference",
        transcript: str = "",
        user_confirmed_text: str = "",
        reference_relation: str = "different_speaker",
        notes: str = "",
    ) -> None:
        cache_ok = cache is None or (
            cache.with_suffix(".json").is_file() and cache.with_suffix(".npz").is_file()
        )
        rows.append(
            {
                "sample_id": sample_id,
                "wav_path": str(wav.resolve()),
                "source": source,
                "speaker": speaker,
                "expected_category": category,
                "target_text": target,
                "cache_path": str(cache.resolve()) if cache else "",
                "mode": mode,
                "transcript": transcript,
                "user_confirmed_text": user_confirmed_text,
                "reference_relation": reference_relation,
                "wav_sha256": _sha256(wav) if wav.is_file() else "",
                "runnable": str(wav.is_file() and cache_ok).lower(),
                "notes": notes,
            }
        )

    # Ten native fixed-reading comparisons, with both self and cross-speaker references.
    native_pairs = [
        (1, "jvs001", "self"),
        (1, "jvs002", "different_speaker"),
        (1, "jvs003", "different_speaker"),
        (1, "jvs004", "different_speaker"),
        (2, "jvs001", "self"),
        (2, "jvs002", "different_speaker"),
        (3, "jvs003", "different_speaker"),
        (4, "jvs004", "different_speaker"),
        (5, "jvs002", "different_speaker"),
        (5, "jvs005", "different_speaker"),
    ]
    for sentence_id, speaker, relation in native_pairs:
        wav = jvs_root / speaker / "parallel100" / "wav24kHz16bit" / f"VOICEACTRESS100_{sentence_id:03d}.wav"
        add(
            f"native_{speaker}_s{sentence_id:03d}", wav, "JVS_parallel100", speaker,
            "native", JVS_SENTENCES[sentence_id], _find_jvs_cache(cache_root, sentence_id),
            reference_relation=relation,
        )

    # JANON has the same isolated stimuli across native and learner speakers. Build
    # each cache once from the labelled Japanese-native jpf1 recording.
    janon_items = [
        ("i74", "ばっちり"),
        ("i98", "さっさと"),
        ("i99", "がっしり"),
        ("i63", "うっとうしい"),
        ("i77", "オイル"),
        ("i5", "バグ"),
        ("i60", "酸味"),
    ]
    for stimulus_id, text in janon_items:
        ref_wav = janon_root / "audio" / "jpf1" / "isolated" / f"jpf1_{stimulus_id}.wav"
        cache = _prepare_janon_cache(
            text,
            ref_wav,
            output_root / "caches" / "janon" / stimulus_id / "sentence",
            f"JANON:jpf1:{stimulus_id}",
        )
        add(
            f"janon_native_{stimulus_id}", ref_wav, "JANON", "jpf1", "native",
            text, cache, reference_relation="self",
        )
        for speaker, l1 in (("chf1", "Mandarin"), ("enf1", "English")):
            wav = janon_root / "audio" / speaker / "isolated" / f"{speaker}_{stimulus_id}.wav"
            add(
                f"learner_{speaker}_{stimulus_id}", wav, "JANON", speaker, "learner",
                text, cache, reference_relation="different_speaker",
                notes=f"Japanese fixed-reading by an L1-{l1} learner; not a non-Japanese control",
            )

    # Real Japanese mismatch: unchanged JVS audio is intentionally assigned another target/cache.
    mismatch_wav = jvs_root / "jvs002" / "parallel100" / "wav24kHz16bit" / "VOICEACTRESS100_002.wav"
    mismatch_cache = _find_jvs_cache(cache_root, 1)
    add(
        "mismatch_jvs_s002_as_s001", mismatch_wav, "JVS_parallel100", "jvs002",
        "general_fallback", JVS_SENTENCES[1], mismatch_cache,
        notes="Real Japanese sentence B evaluated against target A; exercises ASR fallback",
    )
    add(
        "mismatch_jvs_s002_as_ramen", mismatch_wav, "JVS_parallel100", "jvs002",
        "general_fallback", "ラーメンをください",
        version_root / "cache" / "production_references" / "ramen_kudasai" / "v1" / "sentence",
        notes="Large real target mismatch intended to force ASR and the broad Japanese fallback",
    )

    # Same audio in broad mode permits a direct fixed-vs-broad comparison.
    broad_wav = jvs_root / "jvs002" / "parallel100" / "wav24kHz16bit" / "VOICEACTRESS100_001.wav"
    add(
        "broad_jvs002_s001", broad_wav, "JVS_parallel100", "jvs002", "broad_mode",
        JVS_SENTENCES[1], None, mode="transcript_assisted_light", transcript=JVS_SENTENCES[1],
        notes="Same WAV as native_jvs002_s001, evaluated without fixed-reference claims",
    )
    add(
        "broad_jvs002_s002", mismatch_wav, "JVS_parallel100", "jvs002", "broad_mode",
        JVS_SENTENCES[2], None, mode="transcript_assisted_light", transcript=JVS_SENTENCES[2],
        notes="Same WAV as the forced ramen mismatch, directly evaluated in broad mode",
    )

    # Controlled edits are derived from real JVS recordings and already existed in the repo.
    controlled = version_root / "reports" / "tier0_batch3_audio"
    cache_s1 = _find_jvs_cache(cache_root, 1)
    add(
        "partial_local_delete_jvs005_s001",
        controlled / "B2_rhythm" / "jvs005_VOICEACTRESS100_001_ref-jvs001_local_mora_delete_like.wav",
        "JVS_controlled_edit", "jvs005", "partial_target", JVS_SENTENCES[1], cache_s1,
        notes="Synthetic local delete-like edit of a real native recording; not a learner error claim",
    )
    add(
        "moderate_noise_jvs005_s001",
        controlled / "C_channel" / "jvs005_VOICEACTRESS100_001_ref-jvs001_mild_noise_15db.wav",
        "JVS_controlled_edit", "jvs005", "moderate_recording", JVS_SENTENCES[1], cache_s1,
        notes="Existing 15 dB mild-noise edit of real native speech",
    )
    add(
        "pause_insert_jvs005_s001",
        controlled / "B3_fluency" / "jvs005_VOICEACTRESS100_001_ref-jvs001_long_pause_insert.wav",
        "JVS_controlled_edit", "jvs005", "partial_target", JVS_SENTENCES[1], cache_s1,
        notes="Existing long-pause edit of real native speech",
    )

    # Existing short demo recording against the manually approved PyOpenJTalk reference cache.
    add(
        "demo_ramen_fixed", version_root / "data" / "ramen.wav", "demo_recording", "unknown",
        "weak_reference", "ラーメンをください",
        version_root / "cache" / "production_references" / "ramen_kudasai" / "v1" / "sentence",
        notes="Real existing demo recording; production cache is a manually reviewed synthetic reference",
    )
    add(
        "demo_ramen_confirmed_weak", version_root / "data" / "ramen.wav", "demo_recording", "unknown",
        "weak_reference", "ラーメンをください",
        version_root / "cache" / "production_references" / "ramen_kudasai" / "v1" / "sentence",
        mode="asr_confirmed_weak_reference", user_confirmed_text="ラーメンをください",
        notes="Real existing demo recording through the user-confirmed weak-reference contract",
    )

    return rows


def write_inventory(rows: Iterable[dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate_inventory(inventory: Path, output: Path, version_label: str) -> None:
    from jp_speech_eval.api import EvaluationRequest, SpeechEvalConfig, SpeechEvaluationClient

    client = SpeechEvaluationClient(SpeechEvalConfig())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in read_inventory(inventory):
            started = time.perf_counter()
            if sample["runnable"] != "true":
                response: dict[str, Any] = {
                    "ok": False, "mode": sample["mode"], "user_facing": {}, "raw_result": {},
                    "error": "inventory_sample_not_runnable",
                }
            else:
                response = client.evaluate(
                    EvaluationRequest(
                        audio_path=sample["wav_path"],
                        mode=sample["mode"],
                        target_text=sample["target_text"] or None,
                        transcript=sample["transcript"] or None,
                        user_confirmed_text=sample["user_confirmed_text"] or None,
                        cache_path=sample["cache_path"] or None,
                    )
                ).to_dict()
            record = {
                "version": version_label,
                "sample": sample,
                "latency_sec": round(time.perf_counter() - started, 6),
                "response": response,
            }
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            print(sample["sample_id"], response.get("ok"), response.get("mode"), record["latency_sec"])


def _records(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            result[record["sample"]["sample_id"]] = record
    return result


def _compact(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response") or {}
    user = response.get("user_facing") or {}
    raw = response.get("raw_result") or {}
    details = raw.get("details") or {}
    content = details.get("content_match") or {}
    policy = details.get("user_score_policy") or {}
    practice = user.get("practice_score") or {}
    dimensions = {
        item.get("key"): item.get("value")
        for item in user.get("score_dimensions") or []
        if isinstance(item, dict)
    }
    display = user.get("display_score")
    return {
        "ok": response.get("ok"),
        "effective_mode": response.get("mode"),
        "score_available": display is not None,
        "display_score": display,
        "practice_score": practice.get("value"),
        "status": user.get("status"),
        "confidence": user.get("confidence"),
        "detail_allowed": user.get("detail_feedback_allowed"),
        "alignment_mode": raw.get("alignment_mode"),
        "content_status": content.get("status"),
        "content_verified": content.get("content_verified"),
        "japanese_content_plausible": content.get("japanese_content_plausible"),
        "asr_transcript": content.get("transcript"),
        "pronunciation": dimensions.get("pronunciation_clarity"),
        "rhythm": dimensions.get("mora_rhythm"),
        "fluency": dimensions.get("delivery_fluency"),
        "pitch": dimensions.get("pitch_accent"),
        "policy_mode": policy.get("scoring_mode"),
        "user_messages": " | ".join(str(v) for v in user.get("user_messages") or []),
        "error": response.get("error"),
        "latency_sec": record.get("latency_sec"),
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def compare(old_path: Path, new_path: Path, output_dir: Path) -> None:
    old, new = _records(old_path), _records(new_path)
    ids = sorted(set(old) | set(new))
    rows: list[dict[str, Any]] = []
    for sample_id in ids:
        old_c, new_c = _compact(old.get(sample_id, {})), _compact(new.get(sample_id, {}))
        sample = (new.get(sample_id) or old.get(sample_id))["sample"]
        rows.append(
            {
                "sample_id": sample_id,
                "category": sample["expected_category"],
                "source": sample["source"],
                "speaker": sample["speaker"],
                **{f"old_{k}": v for k, v in old_c.items()},
                **{f"new_{k}": v for k, v in new_c.items()},
                "mode_change": f"{old_c.get('effective_mode')} -> {new_c.get('effective_mode')}",
                "notes": sample["notes"],
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = output_dir / "comparison.csv"
    with comparison.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    eligible = [r for r in rows if r["new_score_available"] and r["category"] not in {"negative"}]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in eligible:
        grouped[row["category"]].append(float(row["new_display_score"]))
    grouped["all_scorable"] = [float(row["new_display_score"]) for row in eligible]

    distribution = []
    for group, values in sorted(grouped.items()):
        distribution.append(
            {
                "group": group,
                "n": len(values),
                "mean": round(statistics.mean(values), 3),
                "std": round(statistics.pstdev(values), 3),
                "min": min(values),
                "p10": round(_percentile(values, .10) or 0, 3),
                "p25": round(_percentile(values, .25) or 0, 3),
                "median": round(statistics.median(values), 3),
                "p75": round(_percentile(values, .75) or 0, 3),
                "p90": round(_percentile(values, .90) or 0, 3),
                "max": max(values),
            }
        )
    with (output_dir / "score_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distribution[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(distribution)

    categories = Counter(r["category"] for r in rows)
    japanese = [r for r in rows if r["category"] not in {"negative"}]
    old_new = sum(not r["old_score_available"] and r["new_score_available"] for r in japanese)
    summary = [
        "# C-end v2 acceptance run summary",
        "",
        f"- Samples: {len(rows)}",
        f"- Category counts: `{dict(sorted(categories.items()))}`",
        f"- Reasonable-Japanese old no-score -> new score: {old_new}",
        f"- New score availability: {sum(r['new_score_available'] for r in rows)}/{len(rows)}",
        f"- Old score availability: {sum(r['old_score_available'] for r in rows)}/{len(rows)}",
        "- Negative controls: 0 (no labelled English/Mandarin-content, pure-noise, or silence asset found)",
        "",
        "## Distribution",
        "",
        "|group|n|mean|std|min|p10|p25|median|p75|p90|max|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in distribution:
        summary.append("|" + "|".join(str(row[k]) for k in row) + "|")
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--data-root", type=Path, required=True)
    inventory.add_argument("--output-dir", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--inventory", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--version-label", required=True)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--old", type=Path, required=True)
    compare_parser.add_argument("--new", type=Path, required=True)
    compare_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        rows = build_inventory(args.data_root.resolve(), args.output_dir.resolve())
        write_inventory(rows, args.output_dir / "sample_inventory.csv")
        print(Counter(row["expected_category"] for row in rows))
        print(f"runnable={sum(row['runnable'] == 'true' for row in rows)}/{len(rows)}")
    elif args.command == "evaluate":
        evaluate_inventory(args.inventory.resolve(), args.output.resolve(), args.version_label)
    else:
        compare(args.old.resolve(), args.new.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
