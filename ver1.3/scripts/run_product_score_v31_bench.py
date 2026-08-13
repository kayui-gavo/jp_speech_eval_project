"""v3.1 alignment/evidence and held-out cascade audit (candidate-only)."""
from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from jp_speech_eval.content_cascade import split_rows_by_target_sentence

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/ryukayuiii/Documents/jp_speech_eval_project")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [], lineterminator="\n")
        if rows: writer.writeheader(); writer.writerows(rows)


def health(boundaries: list[tuple[float, float]]) -> bool:
    d = np.asarray([max(0., e - s) for s, e in boundaries]); avg = float(np.mean(d)) if len(d) else 0.
    return not len(d) or float(np.std(d) / max(avg, 1e-8)) > .75 or float(np.min(d)) < .07 or float(d[0]) < .10 or float(np.max(d)) > max(.55, 3.2 * avg)


def alignment_bench() -> list[dict[str, Any]]:
    from jp_speech_eval.alignment import estimate_mora_boundaries
    from jp_speech_eval.sentence_cache import load_sentence_cache
    from jp_speech_eval.audio_features import load_audio
    from jp_speech_eval.vad import trim_to_speech
    cache = load_sentence_cache(DATA / "ver1.3/reports/tier0_batch3_reference_cache/VOICEACTRESS100_001/ref_jvs001_3471c4df0e")
    root = DATA / "ver1.3/reports/tier0_batch3_audio"
    cases = {
        "clean": DATA / "JVS/jvs005/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav",
        "codec": root / "B1_pronunciation/jvs005_VOICEACTRESS100_001_ref-jvs001_codec_control.wav",
        "bandlimit": root / "B1_pronunciation/jvs005_VOICEACTRESS100_001_ref-jvs001_bandlimit_control.wav",
        "gain": root / "C_channel/jvs005_VOICEACTRESS100_001_ref-jvs001_gain_plus6db.wav",
        "rir": root / "C_channel/jvs005_VOICEACTRESS100_001_ref-jvs001_rir_mild.wav",
        "noise_15db": root / "C_channel/jvs005_VOICEACTRESS100_001_ref-jvs001_mild_noise_15db.wav",
        "wrong_target": DATA / "JVS/jvs005/parallel100/wav24kHz16bit/VOICEACTRESS100_002.wav",
    }
    variants = [("A_mfcc", "mfcc", .25, False), ("B_mfcc_delta", "mfcc_delta", .25, False), ("C_logmel", "logmel", .25, False), ("D_mfcc_wide_second_pass", "mfcc", .25, True)]
    rows = []
    for condition, wav in cases.items():
        audio = load_audio(wav, sr=16000)
        # Benchmark the exact endpointed waveform the product passes to DTW;
        # otherwise channel effects on leading/trailing silence are conflated
        # with alignment robustness.
        speech, region = trim_to_speech(audio.y, audio.sr)
        for name, feature, band, second in variants:
            t0 = time.perf_counter()
            result = estimate_mora_boundaries(cache.meta.text, speech, audio.sr, cache.mora_count, mode="cached_dtw", cache=cache, return_result=True, feature_kind=feature, band_rad=band, second_pass_wider_band=second)
            unstable = health(result.boundaries)
            rows.append({"condition": condition, "variant": name, "vad_detected": region.detected, "vad_speech_duration_sec": round(region.speech_duration, 4), "method": result.method, "available": result.available, "used_equal_fallback": result.used_equal_fallback, "failure_reason": result.failure_reason, "normalized_dtw_cost": result.normalized_dtw_cost, "path_length": result.path_length, "path_coverage": result.path_coverage, "path_slope_cv": result.path_slope_cv, "boundary_health_unstable": unstable, "latency_ms": round((time.perf_counter()-t0)*1000, 2), "warning": "wrong target: apparent alignment is not content verification" if condition == "wrong_target" else ""})
    write_csv(ROOT / "reports/ALIGNMENT_V31_BENCH.csv", rows)
    return rows


def heldout_cascade() -> list[dict[str, Any]]:
    rows = read_csv(ROOT / "outputs/content_shadow_v2_final/content_gate_v2_results.csv")
    by = defaultdict(dict)
    for row in rows: by[row["sample_id"]][row["system"]] = row
    # Sentence IDs 1–10 form development; 11–20 are held out.  Target IDs are
    # disjoint by construction. Speaker overlap remains and is reported.
    output = []
    all_base_rows = [systems["faster_whisper_base"] for systems in by.values()]
    development_rows, held_out_rows = split_rows_by_target_sentence(all_base_rows, set(map(str, range(1, 11))))
    for split, base_rows in (("development", development_rows), ("held_out", held_out_rows)):
        ids = {str(row["target_sentence_id"]) for row in base_rows}
        subset = [by[row["sample_id"]] for row in base_rows]
        thresholds = [.40, .45, .50, .55, .60, .65, .70]
        for threshold in thresholds:
            selected = []
            for systems in subset:
                base, small = systems["faster_whisper_base"], systems["faster_whisper_small"]
                rescue = base["content_verified"] != "True" and float(base["kana_similarity"] or 0) >= threshold
                chosen = small if rescue else base
                selected.append((chosen, rescue, base, small))
            correct = [x for x in selected if x[0]["is_correct_target"] == "True"]; wrong = [x for x in selected if x[0]["is_correct_target"] != "True"]
            latencies = [float(x[2]["latency_sec"] or 0) + (float(x[3]["latency_sec"] or 0) if x[1] else 0) for x in selected]
            output.append({"split": split, "rescue_similarity_floor": threshold, "target_sentence_ids": ",".join(sorted(ids, key=int)), "n": len(selected), "correct_n": len(correct), "wrong_n": len(wrong), "wrong_false_verified_rate": round(sum(x[0]["content_verified"] == "True" for x in wrong)/max(len(wrong),1),4), "fixed_detail_retention": round(sum(x[0]["content_verified"] == "True" for x in correct)/max(len(correct),1),4), "small_invocation_rate": round(sum(x[1] for x in selected)/max(len(selected),1),4), "broad_fallback_rate": round(sum(x[0]["content_verified"] != "True" for x in selected)/max(len(selected),1),4), "warm_p90_sec": round(float(np.percentile(latencies,90)),4), "speakers": ",".join(sorted({x[0]["speaker"] for x in selected}))})
    write_csv(ROOT / "outputs/product_score_v31/content_cascade_heldout.csv", output)
    return output


def native_overlap() -> list[dict[str, Any]]:
    from jp_speech_eval.text_frontend import text_to_kana
    janon = read_csv(DATA / "JANON/data.csv")
    native = [r for r in janon if r["Native Language"].strip().lower() == "japanese"]
    targets = sorted({r["Stmiulus"] for r in janon if r["Speaker"] in {"chf1", "enf1"} and r["Stmiulus"] in {"ばっちり","さっさと","がっしり","うっとうしい","オイル","バグ","酸味"}})
    # JVS's transcript is shared across its 100 native speakers.  We inspect
    # its canonical parallel100 transcript only; no recursive corpus scan.
    jvs_transcript = DATA / "JVS/jvs001/parallel100/transcripts_utf8.txt"
    jvs_by_kana: dict[str, str] = {}
    if jvs_transcript.is_file():
        for line in jvs_transcript.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            sentence_id, transcript = line.split(":", 1)
            jvs_by_kana[text_to_kana(transcript.strip())] = sentence_id.strip()
    rows = []
    for text in targets:
        kana = text_to_kana(text)
        matches = [r for r in native if text_to_kana(r["Stmiulus"]) == kana and (DATA / "JANON" / r["Path"]).is_file()]
        jvs_id = jvs_by_kana.get(kana)
        jvs_paths = ([] if not jvs_id else sorted(DATA.glob(f"JVS/jvs*/parallel100/wav24kHz16bit/{jvs_id}.wav")))
        count = len({r["Speaker"] for r in matches}) + len(jvs_paths)
        bucket = "0" if count == 0 else "1" if count == 1 else "2" if count == 2 else "3-5" if count <= 5 else ">5"
        rows.append({"target_text": text, "normalized_kana": kana, "datasets_scanned": "JANON;JVS_parallel100", "native_reference_count": count, "count_bucket": bucket, "native_speakers": ";".join(sorted({r["Speaker"] for r in matches}) + (["JVS_parallel100_100_speakers"] if jvs_paths else [])), "native_wav_paths": ";".join([str(Path("JANON") / r["Path"]) for r in matches] + [str(p.relative_to(DATA)) for p in jvs_paths]), "learner_speakers": "chf1;enf1", "status": "sufficient_for_multi_reference" if count >= 3 else "insufficient_human_native_references"})
    write_csv(ROOT / "data/audit/native_reference_overlap.csv", rows)
    return rows


def build_native_reference_bank(overlap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Materialise only target banks with the requested 3–5 human references."""
    bank: list[dict[str, Any]] = []
    for row in overlap_rows:
        count = int(row["native_reference_count"])
        if not 3 <= count <= 5:
            continue
        for index, wav_path in enumerate(filter(None, row["native_wav_paths"].split(";")), start=1):
            bank.append({
                "target_text": row["target_text"],
                "normalized_kana": row["normalized_kana"],
                "reference_index": index,
                "wav_path": wav_path,
                "bank_size": count,
                "source": "JANON_human_native" if wav_path.startswith("JANON/") else "JVS_parallel100",
                "status": "eligible_for_wavlm_research_only",
            })
    write_csv(ROOT / "data/audit/native_reference_bank.csv", bank)
    return bank


def wavlm_reference_bank_subset(bank: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exercise WavLM only on an actually sufficient human-reference subset.

    This is deliberately a feasibility check, not a candidate-score mapping:
    one isolated-word target cannot calibrate a production pronunciation score.
    """
    from jp_speech_eval.ssl_features import (
        SSLFeatureExtractor,
        aggregate_reference_distances,
        cosine_dtw_distance,
    )
    import librosa

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bank:
        grouped[row["target_text"]].append(row)
    # The deterministic first target has four human JANON native references
    # and two learner recordings in the local real-audio panel.
    target = next((text for text in sorted(grouped) if len(grouped[text]) >= 3), None)
    out_path = ROOT / "outputs/product_score_v31/wavlm_reference_bank_subset.csv"
    if target is None:
        rows = [{"target_text": "", "relation": "unavailable", "reason": "no_3_to_5_reference_bank", "reference_count": 0}]
        write_csv(out_path, rows)
        return rows
    references = grouped[target]
    janon = read_csv(DATA / "JANON/data.csv")
    learners = [
        {"speaker": row["Speaker"], "wav_path": str(Path("JANON") / row["Path"])}
        for row in janon
        if row["Stmiulus"] == target and row["Speaker"] in {"chf1", "enf1"}
    ]
    extractor = SSLFeatureExtractor(local_files_only=True)
    feature_cache: dict[str, dict[int, np.ndarray]] = {}

    def features(path: str) -> dict[int, np.ndarray]:
        if path not in feature_cache:
            audio, sr = librosa.load(str(DATA / path), sr=16000, mono=True)
            feature_cache[path] = extractor.extract_all_layers(audio, sr)
        return feature_cache[path]

    rows: list[dict[str, Any]] = []
    try:
        for user in references + learners:
            is_native = user in references
            refs = [ref for ref in references if ref["wav_path"] != user["wav_path"]]
            if not refs:
                continue
            user_features = features(user["wav_path"])
            values: dict[int, list[float]] = {12: [], 24: []}
            for ref in refs:
                ref_features = features(ref["wav_path"])
                for layer in values:
                    values[layer].append(cosine_dtw_distance(ref_features[layer], user_features[layer])["normalized_cumulative_distance"])
            rows.append({
                "target_text": target,
                "relation": "native_leave_one_out" if is_native else "learner_to_native_bank",
                "user_speaker": user.get("speaker", Path(user["wav_path"]).stem),
                "reference_count": len(refs),
                "distance_layer12_median": round(aggregate_reference_distances(values[12], "median"), 6),
                "distance_layer24_median": round(aggregate_reference_distances(values[24], "median"), 6),
                "score_mapping": "unavailable_single_target_feasibility_only",
                "reason": "reference_count_sufficient_but_no_v31_score_calibration",
            })
    except Exception as exc:
        rows = [{
            "target_text": target,
            "relation": "runtime_unavailable",
            "reference_count": len(references),
            "reason": f"{type(exc).__name__}:{exc}",
        }]
    write_csv(out_path, rows)
    return rows


def rescore_panel() -> list[dict[str, Any]]:
    from jp_speech_eval.evaluator import evaluate_utterance
    out = ROOT / "outputs/product_score_v31"; out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/scoring_config.json").read_text()); config["content_match"]["enabled"] = False; config["product_score_v3"]["enabled"] = True
    path = out / "v31_config.json"; path.write_text(json.dumps(config), encoding="utf-8")
    records = []
    for item in read_csv(ROOT / "outputs/c_end_v2_acceptance/sample_inventory.csv"):
        if item["runnable"] != "true" or item["mode"] != "reference": continue
        result = evaluate_utterance(wav_path=item["wav_path"], cache_path=item["cache_path"], scoring_config_path=path, use_content_match=False).to_dict()
        v3 = result["details"].get("product_score_v3_candidate") or {}; agg = v3.get("product_score_v3_candidate") or {}; alignment = result["details"]["alignment"]
        records.append({"sample_id":item["sample_id"],"category":item["expected_category"],"candidate":agg.get("value"),"scope":agg.get("score_scope"),"coverage":agg.get("evidence_coverage"),"ab_candidate_eligible":agg.get("ab_candidate_eligible"),"alignment_mode":result["alignment_mode"],"alignment_method":alignment["method"],"used_equal_fallback":alignment["used_equal_fallback"],"fallback_reason":alignment["failure_reason"]})
    write_csv(out / "v31_panel.csv", records); return records


def channel_score_retest() -> list[dict[str, Any]]:
    """Separate channel-induced evidence loss from a synthetic good score."""
    from jp_speech_eval.evaluator import evaluate_utterance
    out = ROOT / "outputs/product_score_v31"; out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/scoring_config.json").read_text())
    config["content_match"]["enabled"] = False
    config["product_score_v3"]["enabled"] = True
    config_path = out / "v31_channel_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    audio_root = DATA / "ver1.3/reports/tier0_batch3_audio"
    cache = DATA / "ver1.3/reports/tier0_batch3_reference_cache/VOICEACTRESS100_001/ref_jvs001_3471c4df0e"
    cases = {
        "clean": DATA / "JVS/jvs005/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav",
        "codec": audio_root / "B1_pronunciation/jvs005_VOICEACTRESS100_001_ref-jvs001_codec_control.wav",
        "bandlimit": audio_root / "B1_pronunciation/jvs005_VOICEACTRESS100_001_ref-jvs001_bandlimit_control.wav",
        "gain": audio_root / "C_channel/jvs005_VOICEACTRESS100_001_ref-jvs001_gain_plus6db.wav",
        "rir": audio_root / "C_channel/jvs005_VOICEACTRESS100_001_ref-jvs001_rir_mild.wav",
        "noise_15db": audio_root / "C_channel/jvs005_VOICEACTRESS100_001_ref-jvs001_mild_noise_15db.wav",
    }
    rows: list[dict[str, Any]] = []
    clean_v2 = clean_v3 = None
    for condition, wav in cases.items():
        result = evaluate_utterance(wav_path=wav, cache_path=cache, scoring_config_path=config_path, use_content_match=False).to_dict()
        candidate = result["details"].get("product_score_v3_candidate") or {}
        aggregate = candidate.get("product_score_v3_candidate") or {}
        alignment = result["details"].get("alignment") or {}
        if condition == "clean":
            clean_v2, clean_v3 = result.get("total_score"), aggregate.get("value")
        rows.append({
            "condition": condition,
            "v2_total_score": result.get("total_score"),
            "v2_delta_from_clean": None if clean_v2 is None or result.get("total_score") is None else round(float(result["total_score"]) - float(clean_v2), 4),
            "v3_candidate": aggregate.get("value"),
            "v3_delta_from_clean": None if clean_v3 is None or aggregate.get("value") is None else round(float(aggregate["value"]) - float(clean_v3), 4),
            "score_scope": aggregate.get("score_scope"),
            "evidence_coverage": aggregate.get("evidence_coverage"),
            "ab_candidate_eligible": aggregate.get("ab_candidate_eligible"),
            "alignment_available": alignment.get("available"),
            "used_equal_fallback": alignment.get("used_equal_fallback"),
            "failure_reason": alignment.get("failure_reason"),
            "fake_local_timing_used": (candidate.get("timing_features") or {}).get("evidence_source") == "synthetic_equal_boundaries" and aggregate.get("score_scope") != "continuity_only",
        })
    write_csv(ROOT / "reports/PRODUCT_SCORE_V31_CHANNEL_RETEST.csv", rows)
    return rows


def main() -> None:
    alignment_bench(); heldout_cascade()
    overlap = native_overlap()
    bank = build_native_reference_bank(overlap)
    wavlm_reference_bank_subset(bank)
    rescore_panel()
    channel_score_retest()

if __name__ == "__main__": main()
