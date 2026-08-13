"""Real-audio, candidate-only ProductScore v3 benchmark.

It intentionally reads the prior C-end content/WavLM panels rather than
silently regenerating labels, and enables v3 only in an output-local scoring
JSON.  No result from this script is consumed by the product renderer.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values); pos = (len(values) - 1) * q; lo, hi = int(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values), "mean": round(statistics.mean(values), 4) if values else None,
        "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0 if values else None,
        "min": round(min(values), 4) if values else None, "p10": None if _pct(values, .1) is None else round(_pct(values, .1), 4),
        "p50": None if _pct(values, .5) is None else round(_pct(values, .5), 4), "p90": None if _pct(values, .9) is None else round(_pct(values, .9), 4),
        "max": round(max(values), 4) if values else None,
    }


def content_cascade(source: Path, out: Path) -> list[dict[str, Any]]:
    """Replay real ASR outputs into five policies without new ASR calls."""
    by_sample: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in _csv(source):
        by_sample[row["sample_id"]][row["system"]] = row
    policies = {
        "always_small": lambda base, small, tiny: (small, False),
        "always_base": lambda base, small, tiny: (base, False),
        "base_first_direct_broad": lambda base, small, tiny: (base, False),
        # This is a data-derived conflict band: in the frozen development
        # panel all hard wrong targets are <= .22 and base false mismatches are
        # .475-.710.  It is audit-only, not a released policy threshold.
        "base_first_selective_small_rescue": lambda base, small, tiny: (
            small if base["content_verified"] != "True" and float(base["kana_similarity"] or 0) >= .60 else base,
            base["content_verified"] != "True" and float(base["kana_similarity"] or 0) >= .60,
        ),
        "tiny_first_small_rescue": lambda base, small, tiny: (
            small if tiny["content_verified"] != "True" else tiny,
            tiny["content_verified"] != "True",
        ),
    }
    rows: list[dict[str, Any]] = []
    for name, decide in policies.items():
        chosen: list[tuple[dict[str, str], dict[str, str], bool]] = []
        for sample_id, systems in by_sample.items():
            base, small, tiny = systems["faster_whisper_base"], systems["faster_whisper_small"], systems["faster_whisper_tiny"]
            result, rescued = decide(base, small, tiny)
            first = small if name == "always_small" else base if name.startswith("base") else tiny
            latency = float(first.get("latency_sec") or 0) + (float(small.get("latency_sec") or 0) if rescued else 0)
            chosen.append((result, first, rescued))
        correct = [item for item in chosen if item[0]["is_correct_target"] == "True"]
        wrong = [item for item in chosen if item[0]["is_correct_target"] != "True"]
        latencies = [float(first.get("latency_sec") or 0) + (float(by_sample[result["sample_id"]]["faster_whisper_small"].get("latency_sec") or 0) if rescued else 0) for result, first, rescued in chosen]
        rows.append({
            "policy": name, "n": len(chosen), "correct_n": len(correct), "wrong_n": len(wrong),
            "wrong_false_verified_rate": round(sum(x[0]["content_verified"] == "True" for x in wrong) / max(1, len(wrong)), 4),
            "fixed_detail_retention": round(sum(x[0]["content_verified"] == "True" for x in correct) / max(1, len(correct)), 4),
            "broad_fallback_rate": round(sum(x[0]["content_verified"] != "True" for x in chosen) / max(1, len(chosen)), 4),
            "small_invocation_rate": round(sum(rescue for _result, _first, rescue in chosen) / max(1, len(chosen)), 4),
            "warm_p50_sec": round(_pct(latencies, .5) or 0, 4), "warm_p90_sec": round(_pct(latencies, .9) or 0, 4),
            "cold_sec_observed": "per-model cold starts are in source summary; cascade cold is not additive-measured",
        })
    _write_csv(out / "content_cascade.csv", rows)
    return rows


def wavlm_fusion(source: Path, out: Path) -> list[dict[str, Any]]:
    """Fit native robust scales on odd sentences, evaluate even held-out."""
    rows = [r for r in _csv(source) if r["strategy"] == "median" and r["layer"] in {"12", "24"}]
    keyed: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        keyed[(row["condition"], row["sentence_id"], row["user_speaker"])][row["layer"]] = float(row["distance"])
    native_dev = [v for (cond, sent, _speaker), v in keyed.items() if cond == "native_leave_one_speaker_out" and int(sent) % 2 and "12" in v and "24" in v]
    med = {layer: statistics.median([v[layer] for v in native_dev]) for layer in ("12", "24")}
    mad = {layer: max(statistics.median([abs(v[layer] - med[layer]) for v in native_dev]), 1e-4) for layer in ("12", "24")}
    output: list[dict[str, Any]] = []
    for alpha in (0.0, .25, .5, .75, 1.0):
        for split, predicate in (("development", lambda s: int(s) % 2 == 1), ("held_out", lambda s: int(s) % 2 == 0)):
            values: dict[str, list[float]] = defaultdict(list)
            for (condition, sentence, _speaker), item in keyed.items():
                if not predicate(sentence) or "12" not in item or "24" not in item:
                    continue
                norm12, norm24 = (item["12"] - med["12"]) / mad["12"], (item["24"] - med["24"]) / mad["24"]
                values[condition].append(alpha * norm12 + (1 - alpha) * norm24)
            native, wrong = values["native_leave_one_speaker_out"], values["wrong_target"]
            output.append({"split": split, "alpha_layer12": alpha, **{f"native_{k}": v for k, v in _stats(native).items()}, **{f"wrong_{k}": v for k, v in _stats(wrong).items()}, "median_separation": round((statistics.median(wrong) - statistics.median(native)), 4) if native and wrong else None})
    _write_csv(out / "wavlm_fusion.csv", output)
    return output


def wavlm_janon_same_target(source: Path, inventory: Path, out: Path) -> list[dict[str, Any]]:
    """Run the cached WavLM model on real same-target JANON learner audio.

    JANON contributes one Japanese-native reference speaker here, so these are
    explicitly single-reference learner measurements.  The separate JVS panel
    supplies the 3-reference leave-one-out stability result.
    """
    import librosa
    from jp_speech_eval.ssl_features import SSLFeatureExtractor, cosine_dtw_distance

    raw = [r for r in _csv(source) if r["strategy"] == "median" and r["layer"] in {"12", "24"} and r["condition"] == "native_leave_one_speaker_out"]
    dev = [r for r in raw if int(r["sentence_id"]) % 2]
    med = {layer: statistics.median(float(r["distance"]) for r in dev if r["layer"] == layer) for layer in ("12", "24")}
    mad = {layer: max(statistics.median(abs(float(r["distance"]) - med[layer]) for r in dev if r["layer"] == layer), 1e-4) for layer in ("12", "24")}
    samples = [r for r in _csv(inventory) if r["source"] == "JANON" and r["mode"] == "reference"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in samples:
        grouped[row["target_text"]].append(row)
    extractor, features, output = SSLFeatureExtractor(), {}, []

    def feat(path: str):
        if path not in features:
            y, sr = librosa.load(path, sr=16000, mono=True)
            features[path] = extractor.extract_all_layers(y, sr)
        return features[path]

    for text, group in grouped.items():
        refs = [r for r in group if r["speaker"] == "jpf1"]
        if len(refs) != 1:
            continue
        ref = refs[0]
        for row in group:
            extracted_ref, extracted_user = feat(ref["wav_path"]), feat(row["wav_path"])
            d12 = cosine_dtw_distance(extracted_ref[12], extracted_user[12])["normalized_cumulative_distance"]
            d24 = cosine_dtw_distance(extracted_ref[24], extracted_user[24])["normalized_cumulative_distance"]
            z12, z24 = (d12 - med["12"]) / mad["12"], (d24 - med["24"]) / mad["24"]
            # α=0 is chosen only from the odd-sentence JVS development split;
            # its retained even-sentence separation is written separately.
            fused = z24
            output.append({"target_text": text, "sample_id": row["sample_id"], "category": row["expected_category"], "speaker": row["speaker"], "reference_count": 1, "distance_layer12": round(d12, 6), "distance_layer24": round(d24, 6), "normalized_layer12": round(z12, 4), "normalized_layer24": round(z24, 4), "fusion_alpha_layer12": 0.0, "fused_normalized_distance": round(fused, 4), "score_mapping": "unavailable_pending_multi_reference_learner_validation"})
    _write_csv(out / "wavlm_janon_same_target.csv", output)
    return output


def _v3_config(out: Path) -> Path:
    config = json.loads((ROOT / "configs/scoring_config.json").read_text(encoding="utf-8"))
    config["content_match"]["enabled"] = False
    config["product_score_v3"]["enabled"] = True
    path = out / "v3_candidate_scoring_config.json"; path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def candidate_rescore(inventory: Path, out: Path) -> list[dict[str, Any]]:
    from jp_speech_eval.evaluator import evaluate_utterance
    config = _v3_config(out)
    records: list[dict[str, Any]] = []
    for item in _csv(inventory):
        if item["runnable"] != "true" or item["mode"] != "reference":
            continue
        result = evaluate_utterance(wav_path=item["wav_path"], cache_path=item["cache_path"], scoring_config_path=config, use_content_match=False).to_dict()
        candidate = result["details"].get("product_score_v3_candidate") or {}
        aggregate = candidate.get("product_score_v3_candidate") or {}
        dims = candidate.get("dimensions") or {}
        records.append({"sample_id": item["sample_id"], "category": item["expected_category"], "speaker": item["speaker"], "alignment_mode": result["alignment_mode"], "v2_total": result["total_score"], "v3_candidate": aggregate.get("value"), "v3_confidence": aggregate.get("confidence"), "pronunciation": (dims.get("pronunciation") or {}).get("value"), "rhythm": (dims.get("rhythm") or {}).get("value"), "fluency": (dims.get("fluency") or {}).get("value"), "intonation": (dims.get("intonation") or {}).get("value"), "timing_source": (candidate.get("timing_features") or {}).get("evidence_source"), "timing_reason": (candidate.get("timing_features") or {}).get("reason")})
    _write_csv(out / "v3_candidate_real_audio.csv", records)
    return records


def candidate_ladder(out: Path) -> list[dict[str, Any]]:
    from jp_speech_eval.evaluator import evaluate_utterance
    config = _v3_config(out)
    # The frozen panel records the source cache explicitly, which also makes
    # this worktree-safe when the original cache remains in the main checkout.
    cache = Path(_csv(ROOT / "outputs/c_end_v2_acceptance/sample_inventory.csv")[0]["cache_path"])
    ladder = ROOT / "outputs/content_shadow_v2_final/calibration_ladder"
    rows: list[dict[str, Any]] = []
    for wav in sorted(ladder.glob("*.wav")):
        result = evaluate_utterance(wav_path=wav, cache_path=cache, scoring_config_path=config, use_content_match=False).to_dict()
        candidate = result["details"].get("product_score_v3_candidate") or {}
        aggregate, dims = candidate.get("product_score_v3_candidate") or {}, candidate.get("dimensions") or {}
        rows.append({"condition": wav.stem, "alignment_mode": result["alignment_mode"], "candidate": aggregate.get("value"), "confidence": aggregate.get("confidence"), "rhythm": (dims.get("rhythm") or {}).get("value"), "fluency": (dims.get("fluency") or {}).get("value"), "timing_source": (candidate.get("timing_features") or {}).get("evidence_source")})
    _write_csv(out / "v3_candidate_ladder.csv", rows)
    return rows


def controlled_channel_and_speech(out: Path) -> list[dict[str, Any]]:
    """Score existing real-JVS perturbations without generating new audio."""
    from jp_speech_eval.evaluator import evaluate_utterance
    config = _v3_config(out)
    source_root = Path("/Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/reports/tier0_batch3_audio")
    clean = Path("/Users/ryukayuiii/Documents/jp_speech_eval_project/JVS/jvs005/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav")
    cache = Path(_csv(ROOT / "outputs/c_end_v2_acceptance/sample_inventory.csv")[0]["cache_path"])
    cases = [
        ("clean", "clean", clean),
        ("gain_plus6", "channel", source_root / "C_channel/jvs005_VOICEACTRESS100_001_ref-jvs001_gain_plus6db.wav"),
        ("noise_15db", "channel", source_root / "C_channel/jvs005_VOICEACTRESS100_001_ref-jvs001_mild_noise_15db.wav"),
        ("rir_mild", "channel", source_root / "C_channel/jvs005_VOICEACTRESS100_001_ref-jvs001_rir_mild.wav"),
        ("codec", "channel", source_root / "B1_pronunciation/jvs005_VOICEACTRESS100_001_ref-jvs001_codec_control.wav"),
        ("bandlimit", "channel", source_root / "B1_pronunciation/jvs005_VOICEACTRESS100_001_ref-jvs001_bandlimit_control.wav"),
        ("consonant_attenuation", "speech", source_root / "B1_pronunciation/jvs005_VOICEACTRESS100_001_ref-jvs001_consonant_attenuation.wav"),
        ("vowel_spectral_tilt", "speech", source_root / "B1_pronunciation/jvs005_VOICEACTRESS100_001_ref-jvs001_spectral_tilt_vowel_like.wav"),
        ("time_stretch_0p9", "speech_tempo", source_root / "B2_rhythm/jvs005_VOICEACTRESS100_001_ref-jvs001_time_stretch_0p9.wav"),
        ("time_stretch_1p1", "speech_tempo", source_root / "B2_rhythm/jvs005_VOICEACTRESS100_001_ref-jvs001_time_stretch_1p1.wav"),
        ("local_delete", "speech_local_timing", source_root / "B2_rhythm/jvs005_VOICEACTRESS100_001_ref-jvs001_local_mora_delete_like.wav"),
        ("long_pause", "speech_pause", source_root / "B3_fluency/jvs005_VOICEACTRESS100_001_ref-jvs001_long_pause_insert.wav"),
        ("hesitation", "speech_pause", source_root / "B3_fluency/jvs005_VOICEACTRESS100_001_ref-jvs001_hesitation_noise_insert.wav"),
    ]
    rows: list[dict[str, Any]] = []
    for name, kind, wav in cases:
        if not wav.is_file():
            continue
        result = evaluate_utterance(wav_path=wav, cache_path=cache, scoring_config_path=config, use_content_match=False).to_dict()
        candidate = result["details"].get("product_score_v3_candidate") or {}; aggregate = candidate.get("product_score_v3_candidate") or {}; dims = candidate.get("dimensions") or {}
        rows.append({"condition": name, "kind": kind, "alignment_mode": result["alignment_mode"], "candidate": aggregate.get("value"), "candidate_confidence": aggregate.get("confidence"), "rhythm": (dims.get("rhythm") or {}).get("value"), "fluency": (dims.get("fluency") or {}).get("value"), "intonation": (dims.get("intonation") or {}).get("value"), "recording_confidence": (result["details"].get("reliability") or {}).get("overall"), "timing_source": (candidate.get("timing_features") or {}).get("evidence_source")})
    _write_csv(out / "channel_and_speech_perturbations.csv", rows)
    return rows


def main() -> None:
    out = ROOT / "outputs/product_score_v3"; out.mkdir(parents=True, exist_ok=True)
    content_cascade(ROOT / "outputs/content_shadow_v2_final/content_gate_v2_results.csv", out)
    wavlm_source = ROOT / "outputs/content_shadow_v2_final/wavlm_multi_reference_results.csv"
    inventory = ROOT / "outputs/c_end_v2_acceptance/sample_inventory.csv"
    wavlm_fusion(wavlm_source, out)
    wavlm_janon_same_target(wavlm_source, inventory, out)
    candidate_rescore(inventory, out)
    candidate_ladder(out)
    controlled_channel_and_speech(out)


if __name__ == "__main__":
    main()
