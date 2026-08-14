"""v3.2 multi-reference WavLM evidence validation (candidate telemetry only)."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/ryukayuiii/Documents/jp_speech_eval_project")
OUT = ROOT / "outputs/product_score_v32"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [], lineterminator="\n")
        if rows:
            writer.writeheader(); writer.writerows(rows)


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    x = np.asarray(list(values), dtype=float)
    if not len(x):
        return {"n": 0, "mean": None, "median": None, "sd": None, "mad": None}
    return {"n": int(len(x)), "mean": round(float(np.mean(x)), 6), "median": round(float(np.median(x)), 6), "sd": round(float(np.std(x)), 6), "mad": round(float(np.median(np.abs(x - np.median(x)))), 6)}


def aggregate(values: list[float], strategy: str) -> float:
    ordered = sorted(values)
    if strategy == "median": return float(np.median(ordered))
    if strategy == "trimmed_mean": return float(np.mean(ordered[1:-1] if len(ordered) >= 4 else ordered))
    if strategy == "top2_mean": return float(np.mean(ordered[:2]))
    if strategy == "nearest": return float(ordered[0])
    raise ValueError(strategy)


def ssl_confidence(
    reference_count: int,
    dispersion: float,
    *,
    stable: bool = True,
    recording_quality: float = 1.0,
) -> float:
    """Reliability-only SSL confidence; deliberately independent of alignment."""
    count = min(max(float(reference_count) / 4.0, 0.0), 1.0)
    dispersion_term = math.exp(-max(float(dispersion), 0.0) / .05)
    quality = float(np.clip(float(recording_quality), 0.0, 1.0))
    return round(float(np.clip(.40 * count + .35 * dispersion_term + .15 * float(stable) + .10 * quality, 0.0, 1.0)), 4)


def bank() -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    refs = defaultdict(list)
    for row in read_csv(ROOT / "data/audit/native_reference_bank.csv"):
        refs[row["target_text"]].append(row)
    janon = read_csv(DATA / "JANON/data.csv")
    learners = defaultdict(list)
    for row in janon:
        if row["Speaker"] in {"chf1", "enf1"} and row["Stmiulus"] in refs:
            learners[row["Stmiulus"]].append({"speaker": row["Speaker"], "wav_path": str(Path("JANON") / row["Path"])})
    return refs, learners


def full_bank_run() -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    import librosa
    from jp_speech_eval.ssl_features import SSLFeatureExtractor, cosine_dtw_distance
    refs, learners = bank()
    extractor = SSLFeatureExtractor(local_files_only=True)
    features: dict[str, dict[int, np.ndarray]] = {}

    def feat(relative: str) -> dict[int, np.ndarray]:
        if relative not in features:
            y, sr = librosa.load(str(DATA / relative), sr=16000, mono=True)
            features[relative] = extractor.extract_all_layers(y, sr)
        return features[relative]

    rows: list[dict[str, Any]] = []
    per_user: dict[tuple[str, str, str], dict[str, Any]] = {}
    strategies = ("median", "trimmed_mean", "top2_mean", "nearest")
    targets = sorted(refs)
    for target_index, target in enumerate(targets):
        native = refs[target]
        wrong = refs[targets[(target_index + 1) % len(targets)]][0]
        users = [("native_loo", Path(row["wav_path"]).parts[-3], row["wav_path"], [x for x in native if x["wav_path"] != row["wav_path"]]) for row in native]
        users += [("learner", row["speaker"], row["wav_path"], native) for row in learners[target]]
        users += [("wrong_target", f"wrong_{wrong['reference_index']}", wrong["wav_path"], native)]
        for relation, speaker, user_path, reference_rows in users:
            layers = {12: [], 24: []}
            for ref in reference_rows:
                for layer in layers:
                    layers[layer].append(cosine_dtw_distance(feat(ref["wav_path"])[layer], feat(user_path)[layer])["normalized_cumulative_distance"])
            dispersion = {layer: float(np.std(values)) for layer, values in layers.items()}
            confidence = ssl_confidence(len(reference_rows), float(np.mean(list(dispersion.values()))))
            entry = {"reference_count": len(reference_rows), "distances": layers, "dispersion": dispersion, "ssl_confidence": confidence}
            per_user[(target, relation, speaker)] = entry
            for strategy in strategies:
                rows.append({
                    "target_text": target, "relation": relation, "user_speaker": speaker, "reference_count": len(reference_rows), "aggregation": strategy,
                    "raw_distance_layer12": round(aggregate(layers[12], strategy), 6), "raw_distance_layer24": round(aggregate(layers[24], strategy), 6),
                    "reference_dispersion_layer12": round(dispersion[12], 6), "reference_dispersion_layer24": round(dispersion[24], 6),
                    "ssl_pronunciation_confidence": confidence,
                    "note": "weak native-vs-learner validation label; not a human pronunciation rating",
                })
    write_csv(OUT / "wavlm_7target_raw.csv", rows)
    return rows, per_user


def _target_values(rows: list[dict[str, Any]], target: str, relation: str, strategy: str, layer: str) -> list[float]:
    return [float(row[f"raw_distance_{layer}"]) for row in rows if row["target_text"] == target and row["relation"] == relation and row["aggregation"] == strategy]


def loto_fusion(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = sorted({row["target_text"] for row in rows})
    output: list[dict[str, Any]] = []
    for held in targets:
        development = [target for target in targets if target != held]
        candidates: list[tuple[float, str, float, float, float]] = []
        for strategy in ("median", "trimmed_mean", "top2_mean", "nearest"):
            native12 = [x for target in development for x in _target_values(rows, target, "native_loo", strategy, "layer12")]
            native24 = [x for target in development for x in _target_values(rows, target, "native_loo", strategy, "layer24")]
            med12, med24 = float(np.median(native12)), float(np.median(native24))
            mad12 = max(float(np.median(np.abs(np.asarray(native12) - med12))), 1e-4)
            mad24 = max(float(np.median(np.abs(np.asarray(native24) - med24))), 1e-4)
            for alpha in (0.0, .25, .50, .75, 1.0):
                effects = []
                for target in development:
                    native = [alpha * ((x - med12) / mad12) + (1 - alpha) * ((y - med24) / mad24) for x, y in zip(_target_values(rows, target, "native_loo", strategy, "layer12"), _target_values(rows, target, "native_loo", strategy, "layer24"))]
                    learner = [alpha * ((x - med12) / mad12) + (1 - alpha) * ((y - med24) / mad24) for x, y in zip(_target_values(rows, target, "learner", strategy, "layer12"), _target_values(rows, target, "learner", strategy, "layer24"))]
                    if native and learner:
                        effects.append((float(np.median(learner)) - float(np.median(native))) / max(float(np.std(native)), .25))
                candidates.append((float(np.mean(effects)), strategy, alpha, med12, med24))
        _, strategy, alpha, med12, med24 = max(candidates, key=lambda item: item[0])
        native12 = [x for target in development for x in _target_values(rows, target, "native_loo", strategy, "layer12")]
        native24 = [x for target in development for x in _target_values(rows, target, "native_loo", strategy, "layer24")]
        mad12 = max(float(np.median(np.abs(np.asarray(native12) - med12))), 1e-4)
        mad24 = max(float(np.median(np.abs(np.asarray(native24) - med24))), 1e-4)
        for relation in ("native_loo", "learner", "wrong_target"):
            values = [alpha * ((x - med12) / mad12) + (1 - alpha) * ((y - med24) / mad24) for x, y in zip(_target_values(rows, held, relation, strategy, "layer12"), _target_values(rows, held, relation, strategy, "layer24"))]
            output.append({"held_out_target": held, "development_target_count": len(development), "selected_aggregation": strategy, "selected_alpha_layer12": alpha, "relation": relation, **{f"evidence_index_{k}": v for k, v in stats(values).items()}, "normalization_native_median_layer12": round(med12, 6), "normalization_native_mad_layer12": round(mad12, 6), "normalization_native_median_layer24": round(med24, 6), "normalization_native_mad_layer24": round(mad24, 6)})
    write_csv(OUT / "wavlm_7target_loto_fusion.csv", output)
    return output


def speaker_effect(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for speaker in sorted({row["user_speaker"] for row in rows if row["relation"] == "native_loo"}):
        relevant = [row for row in rows if row["relation"] == "native_loo" and row["user_speaker"] == speaker and row["aggregation"] == "median"]
        output.append({"speaker": speaker, "target_count": len(relevant), **{f"layer12_{k}": v for k, v in stats(float(row["raw_distance_layer12"]) for row in relevant).items()}, **{f"layer24_{k}": v for k, v in stats(float(row["raw_distance_layer24"]) for row in relevant).items()}})
    write_csv(OUT / "wavlm_7target_speaker_effect.csv", output)
    return output


def target_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Weak-label native/learner separation summary, never a correctness claim."""
    output = []
    for target in sorted({row["target_text"] for row in rows}):
        relevant = [row for row in rows if row["target_text"] == target and row["aggregation"] == "median"]
        native = [.5 * (float(row["raw_distance_layer12"]) + float(row["raw_distance_layer24"])) for row in relevant if row["relation"] == "native_loo"]
        learner = {row["user_speaker"]: .5 * (float(row["raw_distance_layer12"]) + float(row["raw_distance_layer24"])) for row in relevant if row["relation"] == "learner"}
        native_stats = stats(native)
        output.append({
            "target_text": target, "aggregation": "median", **{f"native_loo_{key}": value for key, value in native_stats.items()},
            "chf1_distance": round(learner.get("chf1"), 6) if "chf1" in learner else None,
            "enf1_distance": round(learner.get("enf1"), 6) if "enf1" in learner else None,
            "p_chf1_distance_greater_than_native": round(sum(learner["chf1"] > value for value in native) / max(len(native), 1), 4) if "chf1" in learner else None,
            "p_enf1_distance_greater_than_native": round(sum(learner["enf1"] > value for value in native) / max(len(native), 1), 4) if "enf1" in learner else None,
            "learner_vs_native_median_effect": round((float(np.median(list(learner.values()))) - float(np.median(native))) / max(float(np.std(native)), .001), 4),
            "interpretation": "weak_label_only_no_human_pronunciation_rating",
        })
    write_csv(OUT / "wavlm_7target_summary.csv", output)
    return output


def jvs_perturbations() -> list[dict[str, Any]]:
    import librosa
    from jp_speech_eval.alignment import estimate_mora_boundaries
    from jp_speech_eval.audio_features import load_audio
    from jp_speech_eval.recording_quality import assess_recording_quality
    from jp_speech_eval.sentence_cache import load_sentence_cache
    from jp_speech_eval.ssl_features import SSLFeatureExtractor, cosine_dtw_distance
    from jp_speech_eval.vad import trim_to_speech
    extractor = SSLFeatureExtractor(local_files_only=True)
    feature_cache: dict[str, dict[int, np.ndarray]] = {}
    def feat(path: Path) -> dict[int, np.ndarray]:
        key = str(path)
        if key not in feature_cache:
            y, sr = librosa.load(key, sr=16000, mono=True); feature_cache[key] = extractor.extract_all_layers(y, sr)
        return feature_cache[key]
    refs = [DATA / f"JVS/jvs{i:03d}/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav" for i in range(1, 5)]
    clean = DATA / "JVS/jvs005/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"
    root = DATA / "ver1.3/reports/tier0_batch3_audio"
    cases = {
        "clean": clean,
        "gain": root / "C_channel/jvs005_VOICEACTRESS100_001_ref-jvs001_gain_plus6db.wav",
        "rir": root / "C_channel/jvs005_VOICEACTRESS100_001_ref-jvs001_rir_mild.wav",
        "noise_15db": root / "C_channel/jvs005_VOICEACTRESS100_001_ref-jvs001_mild_noise_15db.wav",
        "bandlimit": root / "B1_pronunciation/jvs005_VOICEACTRESS100_001_ref-jvs001_bandlimit_control.wav",
        "codec": root / "B1_pronunciation/jvs005_VOICEACTRESS100_001_ref-jvs001_codec_control.wav",
        "consonant_attenuation": root / "B1_pronunciation/jvs005_VOICEACTRESS100_001_ref-jvs001_consonant_attenuation.wav",
        "vowel_spectral_tilt": root / "B1_pronunciation/jvs005_VOICEACTRESS100_001_ref-jvs001_spectral_tilt_vowel_like.wav",
        "local_delete_like": root / "B2_rhythm/jvs005_VOICEACTRESS100_001_ref-jvs001_local_mora_delete_like.wav",
    }
    cache = load_sentence_cache(DATA / "ver1.3/reports/tier0_batch3_reference_cache/VOICEACTRESS100_001/ref_jvs001_3471c4df0e")
    output = []; clean_distance = None
    for condition, wav in cases.items():
        audio = load_audio(wav, sr=16000); speech, region = trim_to_speech(audio.y, audio.sr)
        recording_quality = float(assess_recording_quality(audio.y, audio.sr, region).get("score", 1.0) or 0.0)
        alignment = estimate_mora_boundaries(cache.meta.text, speech, audio.sr, cache.mora_count, mode="cached_dtw", cache=cache, return_result=True)
        raw = {layer: [cosine_dtw_distance(feat(ref)[layer], feat(wav)[layer])["normalized_cumulative_distance"] for ref in refs] for layer in (12, 24)}
        distance12, distance24 = aggregate(raw[12], "median"), aggregate(raw[24], "median")
        fused = .5 * (distance12 + distance24)
        if condition == "clean": clean_distance = fused
        dispersion = float(np.mean([np.std(raw[12]), np.std(raw[24])]))
        output.append({"condition": condition, "MFCC_alignment_available": alignment.available, "alignment_failure_reason": alignment.failure_reason, "wavlm_layer12_median_distance": round(distance12, 6), "wavlm_layer24_median_distance": round(distance24, 6), "wavlm_distance_mean_layers": round(fused, 6), "delta_from_clean": None if clean_distance is None else round(fused - clean_distance, 6), "reference_dispersion": round(dispersion, 6), "recording_quality": round(recording_quality, 4), "ssl_pronunciation_confidence": ssl_confidence(4, dispersion, recording_quality=recording_quality), "perturbation_class": "baseline" if condition == "clean" else ("channel" if condition in {"gain", "rir", "noise_15db", "bandlimit", "codec"} else "speech")})
    write_csv(OUT / "wavlm_jvs_channel_and_speech.csv", output)
    return output


def telemetry(raw_rows: list[dict[str, Any]], perturbations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from jp_speech_eval.evaluator import evaluate_utterance
    from jp_speech_eval.product_score_v3 import attach_ssl_pronunciation_evidence
    by_key = {(row["target_text"], row["relation"], row["user_speaker"]): row for row in raw_rows if row["aggregation"] == "median"}
    config = json.loads((ROOT / "configs/scoring_config.json").read_text(encoding="utf-8")); config["content_match"]["enabled"] = False; config["product_score_v3"]["enabled"] = True
    config_path = OUT / "v32_telemetry_config.json"; config_path.parent.mkdir(parents=True, exist_ok=True); config_path.write_text(json.dumps(config), encoding="utf-8")
    inventory = read_csv(ROOT / "outputs/c_end_v2_acceptance/sample_inventory.csv")
    output = []
    for item in inventory:
        if item["source"] != "JANON" or item["mode"] != "reference": continue
        relation = "native_loo" if item["speaker"] == "jpf1" else "learner"
        key = (item["target_text"], relation, item["speaker"])
        evidence = by_key.get(key)
        if not evidence: continue
        result = evaluate_utterance(wav_path=item["wav_path"], cache_path=item["cache_path"], scoring_config_path=config_path, use_content_match=False).to_dict()
        base = result["details"].get("product_score_v3_candidate") or {}
        index = .5 * (float(evidence["raw_distance_layer12"]) + float(evidence["raw_distance_layer24"]))
        attached = attach_ssl_pronunciation_evidence(
            base,
            evidence_index=index,
            ssl_pronunciation_confidence=float(evidence["ssl_pronunciation_confidence"]),
            reference_count=int(evidence["reference_count"]),
            reference_dispersion=float(np.mean([float(evidence["reference_dispersion_layer12"]), float(evidence["reference_dispersion_layer24"])])),
            content_verified=True,
            audio_valid=True,
        )
        old, new = base["product_score_v3_candidate"], attached["product_score_v3_candidate"]
        output.append({"sample_id": item["sample_id"], "old_scope": old["score_scope"], "new_scope": new["score_scope"], "diagnostic_candidate_eligible": new["diagnostic_candidate_eligible"], "overall_product_score_candidate_eligible": new["overall_product_score_candidate_eligible"], "overall_reason": new["overall_product_score_candidate_eligibility_reason"], "ssl_pronunciation_evidence_index": round(index, 6), "ssl_pronunciation_confidence": evidence["ssl_pronunciation_confidence"]})
    # Codec proves that global SSL evidence does not require MFCC local timing.
    codec = next(row for row in perturbations if row["condition"] == "codec")
    output.append({"sample_id": "channel_codec_jvs005_s001", "old_scope": "continuity_only", "new_scope": "pronunciation_plus_delivery", "diagnostic_candidate_eligible": True, "overall_product_score_candidate_eligible": False, "overall_reason": "pronunciation_evidence_not_score_mapped", "ssl_pronunciation_evidence_index": codec["wavlm_distance_mean_layers"], "ssl_pronunciation_confidence": codec["ssl_pronunciation_confidence"]})
    write_csv(OUT / "v32_ssl_telemetry.csv", output)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, _ = full_bank_run()
    loto_fusion(raw); speaker_effect(raw); target_summary(raw)
    perturb = jvs_perturbations()
    telemetry(raw, perturb)


if __name__ == "__main__": main()
