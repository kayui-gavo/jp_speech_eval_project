"""Real-audio public-v2 parity harness for the v3.2 alignment refactor."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/ryukayuiii/Documents/jp_speech_eval_project")


def _inventory() -> list[dict[str, str]]:
    path = ROOT / "outputs/c_end_v2_acceptance/sample_inventory.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cache = str(DATA / "ver1.3/reports/tier0_batch3_reference_cache/VOICEACTRESS100_001/ref_jvs001_3471c4df0e")
    channel_root = DATA / "ver1.3/reports/tier0_batch3_audio"
    channel = {
        "codec": channel_root / "B1_pronunciation/jvs005_VOICEACTRESS100_001_ref-jvs001_codec_control.wav",
        "bandlimit": channel_root / "B1_pronunciation/jvs005_VOICEACTRESS100_001_ref-jvs001_bandlimit_control.wav",
        "gain": channel_root / "C_channel/jvs005_VOICEACTRESS100_001_ref-jvs001_gain_plus6db.wav",
        "rir": channel_root / "C_channel/jvs005_VOICEACTRESS100_001_ref-jvs001_rir_mild.wav",
        "noise_15db": channel_root / "C_channel/jvs005_VOICEACTRESS100_001_ref-jvs001_mild_noise_15db.wav",
    }
    target = next(row["target_text"] for row in rows if row["sample_id"] == "native_jvs001_s001")
    rows.extend({
        "sample_id": f"channel_{name}", "wav_path": str(wav), "source": "JVS_controlled_channel",
        "speaker": "jvs005", "expected_category": "channel_perturbation", "target_text": target,
        "cache_path": cache, "mode": "reference", "transcript": "", "user_confirmed_text": "",
        "reference_relation": "different_speaker", "runnable": "true", "notes": "v3.2 parity channel panel",
    } for name, wav in channel.items())
    return rows


def _compact(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response") or {}
    user = response.get("user_facing") or {}
    raw = response.get("raw_result") or {}
    details = raw.get("details") or {}
    dimensions = {item.get("key"): item.get("value") for item in user.get("score_dimensions") or [] if isinstance(item, dict)}
    practice = user.get("practice_score") or {}
    return {
        "display_score": user.get("display_score"), "status": user.get("status"),
        "detail_allowed": user.get("detail_feedback_allowed"), "alignment_mode": raw.get("alignment_mode"),
        "pronunciation": dimensions.get("pronunciation_clarity"), "rhythm": dimensions.get("mora_rhythm"),
        "fluency": dimensions.get("delivery_fluency"), "practice_score": practice.get("value"),
        "ok": response.get("ok"), "error": response.get("error"),
    }


def evaluate(start: int, stop: int, output: Path, label: str) -> None:
    from jp_speech_eval.api import EvaluationRequest, SpeechEvalConfig, SpeechEvaluationClient
    rows = _inventory()[start:stop]
    client = SpeechEvaluationClient(SpeechEvalConfig())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for sample in rows:
            t0 = time.perf_counter()
            response = client.evaluate(EvaluationRequest(
                audio_path=sample["wav_path"], mode=sample["mode"], target_text=sample["target_text"] or None,
                transcript=sample["transcript"] or None, user_confirmed_text=sample["user_confirmed_text"] or None,
                cache_path=sample["cache_path"] or None,
            )).to_dict()
            handle.write(json.dumps({"version": label, "sample": sample, "response": response, "latency_sec": round(time.perf_counter() - t0, 6)}, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            print(sample["sample_id"], flush=True)


def _records(path: Path) -> dict[str, dict[str, Any]]:
    return {row["sample"]["sample_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def compare(old_path: Path, new_path: Path, output: Path) -> None:
    old, new = _records(old_path), _records(new_path)
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(set(old) | set(new)):
        before, after = _compact(old.get(sample_id, {})), _compact(new.get(sample_id, {}))
        delta = None if before["display_score"] is None or after["display_score"] is None else float(after["display_score"]) - float(before["display_score"])
        user_fields_changed = any(before[key] != after[key] for key in ("display_score", "status", "detail_allowed", "pronunciation", "rhythm", "fluency"))
        alignment_changed = before["alignment_mode"] != after["alignment_mode"]
        if alignment_changed and not user_fields_changed:
            category, reason = "A_intentional_provenance_correction", "alignment provenance changed; user-facing fields stayed equal"
        elif before["display_score"] is None and after["display_score"] is not None and str((new.get(sample_id) or {}).get("response", {}).get("mode")) == "reference_mismatch_general_japanese":
            category, reason = "B_explained_non_alignment_change", "pre-existing broad-Japanese fallback now retained this Japanese ASR transcript; not caused by AlignmentResult"
        elif user_fields_changed:
            category, reason = "C_user_facing_regression", "user-facing score/status/detail/dimension changed; investigate before merge"
        else:
            category, reason = "B_no_user_facing_change", "same user-facing result"
        rows.append({
            "sample_id": sample_id,
            "old_display_score": before["display_score"], "new_display_score": after["display_score"], "delta": delta,
            "old_status": before["status"], "new_status": after["status"],
            "old_alignment_mode": before["alignment_mode"], "new_alignment_mode": after["alignment_mode"],
            "old_detail_allowed": before["detail_allowed"], "new_detail_allowed": after["detail_allowed"],
            "old_pronunciation": before["pronunciation"], "new_pronunciation": after["pronunciation"],
            "old_rhythm": before["rhythm"], "new_rhythm": after["rhythm"],
            "old_fluency": before["fluency"], "new_fluency": after["fluency"], "classification": category, "reason": reason,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate"); ev.add_argument("--start", type=int, required=True); ev.add_argument("--stop", type=int, required=True); ev.add_argument("--output", type=Path, required=True); ev.add_argument("--label", required=True)
    cmp = sub.add_parser("compare"); cmp.add_argument("--old", type=Path, required=True); cmp.add_argument("--new", type=Path, required=True); cmp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate": evaluate(args.start, args.stop, args.output, args.label)
    else: compare(args.old, args.new, args.output)


if __name__ == "__main__": main()
