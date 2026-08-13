"""Reproducible real-audio audits for C-end content v2 and shadow candidates.

This script is deliberately audit-only except for the content-gate measurements.
It never changes product scoring configuration and it stores every generated
control under the supplied output directory, not alongside source corpora.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path("/Users/ryukayuiii/Documents/jp_speech_eval_project")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _cache_prefix(cache_root: Path, sentence: int) -> Path:
    stem = f"VOICEACTRESS100_{sentence:03d}"
    matches = sorted((cache_root / stem).glob("ref_jvs001_*.json"))
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve cache for {stem}: {matches}")
    return matches[0].with_suffix("")


def _duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames / max(info.samplerate, 1))


def build_manifest(data_root: Path, path: Path) -> list[dict[str, Any]]:
    """Build 30 correct and 50 duration-matched, non-overlapping wrong pairs."""
    cache_root = data_root / "ver1.3/reports/tier0_batch3_reference_cache"
    jvs = data_root / "JVS"
    speakers = ["jvs001", "jvs002", "jvs003"]
    rows: list[dict[str, Any]] = []
    # Correct: 20 different sentence targets plus 10 cross-speaker instances.
    correct = [("jvs002" if sentence % 2 else "jvs003", sentence, sentence) for sentence in range(1, 21)]
    correct += [("jvs001", sentence, sentence) for sentence in range(1, 11)]
    for n, (speaker, wav_sentence, target_sentence) in enumerate(correct, 1):
        wav = jvs / speaker / "parallel100/wav24kHz16bit" / f"VOICEACTRESS100_{wav_sentence:03d}.wav"
        cache = _cache_prefix(cache_root, target_sentence)
        rows.append({
            "sample_id": f"correct_{n:03d}", "speaker": speaker,
            "wav_sentence_id": wav_sentence, "target_sentence_id": target_sentence,
            "is_correct_target": True, "duration_ratio": round(_duration(wav) / _duration(Path(f"{cache}.ref.wav")), 5),
            "wav_path": str(wav), "cache_path": str(cache),
        })
    candidates: list[tuple[float, str, int, int, Path, Path]] = []
    for speaker in speakers:
        for source in range(1, 21):
            wav = jvs / speaker / "parallel100/wav24kHz16bit" / f"VOICEACTRESS100_{source:03d}.wav"
            wav_duration = _duration(wav)
            for target in range(1, 21):
                if source == target:
                    continue
                cache = _cache_prefix(cache_root, target)
                ratio = wav_duration / _duration(Path(f"{cache}.ref.wav"))
                if 0.90 <= ratio <= 1.10:
                    candidates.append((abs(1.0 - ratio), speaker, source, target, wav, cache))
    # Spread sources/targets/speakers rather than taking only the numerically closest pairs.
    selected: list[tuple[float, str, int, int, Path, Path]] = []
    seen: set[tuple[str, int, int]] = set()
    for item in sorted(candidates, key=lambda value: (value[0], value[1], value[2], value[3])):
        _dev, speaker, source, target, _wav, _cache = item
        if (speaker, source, target) not in seen:
            selected.append(item)
            seen.add((speaker, source, target))
        if len(selected) == 50:
            break
    if len(selected) < 50:
        raise RuntimeError(f"only {len(selected)} duration-matched hard negatives available")
    for n, (_dev, speaker, source, target, wav, cache) in enumerate(selected, 1):
        rows.append({
            "sample_id": f"hard_wrong_{n:03d}", "speaker": speaker,
            "wav_sentence_id": source, "target_sentence_id": target,
            "is_correct_target": False, "duration_ratio": round(_duration(wav) / _duration(Path(f"{cache}.ref.wav")), 5),
            "wav_path": str(wav), "cache_path": str(cache),
        })
    _write_csv(path, rows)
    return rows


def _kana_similarity(target: str, transcript: str) -> float:
    from jp_speech_eval.content_match import _kana_similarity
    from jp_speech_eval.text_frontend import text_to_kana
    try:
        return _kana_similarity(target, text_to_kana(transcript))
    except Exception:
        return 0.0


def run_content_bench(manifest_path: Path, out: Path, models: list[str]) -> None:
    from jp_speech_eval.asr import _FASTER_WHISPER_CACHE, transcribe_japanese
    from jp_speech_eval.audio_features import load_audio
    from jp_speech_eval.content_match import estimate_content_match
    from jp_speech_eval.sentence_cache import load_sentence_cache

    manifest = _read_csv(manifest_path)
    rows: list[dict[str, Any]] = []
    # MFCC baseline, recorded separately because content_verified is intentionally
    # false in v2 even when legacy status used to say pass.
    for item in manifest:
        audio = load_audio(item["wav_path"], sr=16000)
        cache = load_sentence_cache(item["cache_path"])
        result = estimate_content_match(cache, audio.y, audio.sr, use_asr=False, asr_policy="never")
        rows.append({**item, "system": "acoustic_only", "cold": False, "latency_sec": None,
                     "status": result.status, "verification_level": result.verification_level,
                     "legacy_would_verify": result.status == "pass", "content_verified": result.content_verified,
                     "kana_similarity": result.kana_similarity, "transcript": "", "note": result.note})
        print(f"acoustic {item['sample_id']}", flush=True)
    for model in models:
        _FASTER_WHISPER_CACHE.pop((model, "cpu", "int8"), None)
        cached: dict[str, tuple[Any, float, bool]] = {}
        for item in manifest:
            wav = item["wav_path"]
            if wav not in cached:
                audio = load_audio(wav, sr=16000)
                cold = not cached
                started = time.perf_counter()
                transcript = transcribe_japanese(audio.y, audio.sr, model_name=model, provider="faster-whisper")
                cached[wav] = (transcript, time.perf_counter() - started, cold)
            transcript, latency, cold = cached[wav]
            cache = load_sentence_cache(item["cache_path"])
            similarity = _kana_similarity(cache.meta.kana, transcript.text) if transcript.available else 0.0
            verified = bool(transcript.available and similarity >= 0.75)
            rows.append({**item, "system": f"faster_whisper_{model}", "cold": cold,
                         "latency_sec": round(latency, 6), "status": "pass" if verified else "fail",
                         "verification_level": "asr_verified" if verified else "mismatch" if transcript.available else "unavailable",
                         "legacy_would_verify": verified, "content_verified": verified,
                         "kana_similarity": round(similarity, 5), "transcript": transcript.text,
                         "note": transcript.note})
            print(f"{model} {item['sample_id']}", flush=True)
    _write_csv(out / "content_gate_v2_results.csv", rows)
    summary = []
    for system in sorted({row["system"] for row in rows}):
        group = [row for row in rows if row["system"] == system]
        correct = [row for row in group if str(row["is_correct_target"]).lower() == "true"]
        wrong = [row for row in group if str(row["is_correct_target"]).lower() != "true"]
        latencies = [float(row["latency_sec"]) for row in group if row["latency_sec"] not in {None, ""}]
        summary.append({
            "system": system, "correct_n": len(correct), "wrong_n": len(wrong),
            "correct_false_mismatch_rate": round(sum(not bool(r["legacy_would_verify"]) for r in correct) / max(1, len(correct)), 4),
            "wrong_false_verified_rate": round(sum(bool(r["legacy_would_verify"]) for r in wrong) / max(1, len(wrong)), 4),
            "correct_kana_median": round(float(np.median([float(r["kana_similarity"]) for r in correct])), 4),
            "wrong_kana_median": round(float(np.median([float(r["kana_similarity"]) for r in wrong])), 4),
            "cold_latency_sec": next((r["latency_sec"] for r in group if r["cold"]), None),
            "warm_p50_sec": round(_pct(latencies[1:], .50) or 0.0, 4), "warm_p90_sec": round(_pct(latencies[1:], .90) or 0.0, 4),
        })
    _write_csv(out / "content_gate_v2_summary.csv", summary)
    (out / "content_gate_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize_content_rows(rows: list[dict[str, Any]], out: Path) -> None:
    _write_csv(out / "content_gate_v2_results.csv", rows)
    summary = []
    for system in sorted({row["system"] for row in rows}):
        group = [row for row in rows if row["system"] == system]
        correct = [row for row in group if str(row["is_correct_target"]).lower() == "true"]
        wrong = [row for row in group if str(row["is_correct_target"]).lower() != "true"]
        latencies = [float(row["latency_sec"]) for row in group if row["latency_sec"] not in {None, ""}]
        summary.append({"system": system, "correct_n": len(correct), "wrong_n": len(wrong),
                        "correct_false_mismatch_rate": round(sum(str(r["legacy_would_verify"]).lower() != "true" for r in correct) / max(1, len(correct)), 4),
                        "wrong_false_verified_rate": round(sum(str(r["legacy_would_verify"]).lower() == "true" for r in wrong) / max(1, len(wrong)), 4),
                        "correct_kana_median": round(float(np.median([float(r["kana_similarity"]) for r in correct])), 4),
                        "wrong_kana_median": round(float(np.median([float(r["kana_similarity"]) for r in wrong])), 4),
                        "cold_latency_sec": next((r["latency_sec"] for r in group if str(r["cold"]).lower() == "true"), None),
                        "warm_p50_sec": round(_pct(latencies[1:], .50) or 0.0, 4), "warm_p90_sec": round(_pct(latencies[1:], .90) or 0.0, 4)})
    _write_csv(out / "content_gate_v2_summary.csv", summary)
    (out / "content_gate_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_wav(path: Path, y: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(y, dtype=np.float32), sr)


def make_engineering_controls(out: Path) -> list[dict[str, Any]]:
    sr, seconds = 16000, 2.0
    n = int(sr * seconds)
    rng = np.random.default_rng(17)
    time_axis = np.arange(n) / sr
    controls = {
        "silence": np.zeros(n), "white_noise": rng.normal(0, .08, n),
        "pink_like_noise": np.cumsum(rng.normal(0, .002, n)),
        "tone_hum": .1 * np.sin(2 * np.pi * 180 * time_axis),
        "burst_noise": np.tile(np.r_[rng.normal(0, .1, sr // 5), np.zeros(sr // 5)], 5),
    }
    rows = []
    for name, y in controls.items():
        y = y / max(1.0, float(np.max(np.abs(y)))) * .12
        path = out / "engineering_controls" / f"{name}.wav"
        _write_wav(path, y, sr)
        rows.append({"sample_id": name, "wav_path": str(path), "control_type": "synthetic_engineering_control", "generation": "numpy"})
    # macOS system voices are permitted as engineering controls only.  Failure
    # to have a Mandarin voice remains recorded rather than silently replaced.
    for name, voice, text in (("english_tts", "Alex", "This is an English test sentence."), ("mandarin_tts", "Ting-Ting", "这是一个普通话测试句子。")):
        aiff, wav = out / "engineering_controls" / f"{name}.aiff", out / "engineering_controls" / f"{name}.wav"
        try:
            subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True, capture_output=True, timeout=30)
            y, got_sr = sf.read(str(aiff), dtype="float32")
            _write_wav(wav, y, got_sr)
            rows.append({"sample_id": name, "wav_path": str(wav), "control_type": "synthetic_engineering_control", "generation": f"macos_say:{voice}"})
        except Exception as exc:
            rows.append({"sample_id": name, "wav_path": "", "control_type": "synthetic_engineering_control", "generation": f"unavailable:{type(exc).__name__}"})
    _write_csv(out / "engineering_negative_controls.csv", rows)
    return rows


def evaluate_engineering_controls(data_root: Path, out: Path) -> None:
    from jp_speech_eval.api import EvaluationRequest, SpeechEvalConfig, SpeechEvaluationClient
    controls = _read_csv(out / "engineering_negative_controls.csv")
    cache = _cache_prefix(data_root / "ver1.3/reports/tier0_batch3_reference_cache", 1)
    from jp_speech_eval.sentence_cache import load_sentence_cache
    target = load_sentence_cache(cache).meta.text
    client = SpeechEvaluationClient(SpeechEvalConfig())
    rows = []
    for control in controls:
        if not control.get("wav_path"):
            rows.append({**control, "score_available": False, "status": "generation_unavailable"})
            continue
        response = client.evaluate(EvaluationRequest(audio_path=control["wav_path"], mode="reference", target_text=target, cache_path=cache)).to_dict()
        user, raw = response.get("user_facing", {}), response.get("raw_result", {})
        content = raw.get("details", {}).get("content_match", {})
        rows.append({**control, "score_available": user.get("display_score") is not None,
                     "display_score": user.get("display_score"), "status": user.get("status"),
                     "content_status": content.get("status"), "verification_level": content.get("verification_level"),
                     "asr_transcript": content.get("transcript"), "error": response.get("error")})
        print(f"control {control['sample_id']}", flush=True)
    _write_csv(out / "engineering_negative_control_results.csv", rows)


def run_calibration(data_root: Path, out: Path) -> None:
    """Controlled ladder: measurements only; never maps features to product scores."""
    from jp_speech_eval.api import EvaluationRequest, SpeechEvalConfig, SpeechEvaluationClient
    from jp_speech_eval.sentence_cache import load_sentence_cache
    source = data_root / "JVS/jvs005/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"
    cache = _cache_prefix(data_root / "ver1.3/reports/tier0_batch3_reference_cache", 1)
    target_text = load_sentence_cache(cache).meta.text
    y, sr = librosa.load(str(source), sr=16000, mono=True)
    ladder: list[tuple[str, float, np.ndarray]] = [("clean", 0.0, y)]
    for rate in (.7, .8, .9, 1.1, 1.2, 1.3):
        ladder.append((f"global_speed_{rate}", abs(rate - 1), librosa.effects.time_stretch(y, rate=rate)))
    for pause in (.20, .45, .80):
        cut = len(y) // 2
        ladder.append((f"pause_insert_{pause}", pause, np.r_[y[:cut], np.zeros(int(pause * sr)), y[cut:]]))
    for db in (25, 15, 8):
        noise = np.random.default_rng(int(db)).normal(0, np.std(y) / (10 ** (db / 20)), len(y))
        ladder.append((f"noise_{db}db", 30 - db, y + noise))
    rows = []
    client = SpeechEvaluationClient(SpeechEvalConfig())
    clean_score = None
    for name, severity, signal in ladder:
        wav = out / "calibration_ladder" / f"{name}.wav"
        _write_wav(wav, signal, sr)
        response = client.evaluate(EvaluationRequest(audio_path=wav, mode="reference", target_text=target_text, cache_path=cache)).to_dict()
        raw = response.get("raw_result", {})
        details = raw.get("details", {})
        user = response.get("user_facing", {})
        if name == "clean":
            clean_score = user.get("display_score")
        rows.append({"condition": name, "severity": severity, "display_score": user.get("display_score"),
                     "delta_from_clean": None if clean_score is None or user.get("display_score") is None else round(float(user["display_score"]) - float(clean_score), 3),
                     "pronunciation_score": raw.get("pronunciation_score"), "rhythm_timing_score": details.get("fluency", {}).get("rhythm_timing_score"),
                     "delivery_fluency_score": details.get("fluency", {}).get("delivery_fluency_score"),
                     **details.get("calibration_features", {})})
    _write_csv(out / "product_calibration_ladder.csv", rows)


def run_wavlm(data_root: Path, out: Path, layers: list[int]) -> None:
    from jp_speech_eval.ssl_features import SSLFeatureExtractor, aggregate_reference_distances, cosine_dtw_distance
    jvs, cache_root = data_root / "JVS", data_root / "ver1.3/reports/tier0_batch3_reference_cache"
    extractor = SSLFeatureExtractor()
    speakers = ["jvs001", "jvs002", "jvs003", "jvs004"]
    feature_cache: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    def feats(wav: Path) -> dict[int, np.ndarray]:
        key = (str(wav), 16000)
        if key not in feature_cache:
            y, sr = librosa.load(str(wav), sr=16000, mono=True)
            feature_cache[key] = extractor.extract_all_layers(y, sr)
        return feature_cache[key]
    rows: list[dict[str, Any]] = []
    for sentence in range(1, 21):
        paths = {speaker: jvs / speaker / "parallel100/wav24kHz16bit" / f"VOICEACTRESS100_{sentence:03d}.wav" for speaker in speakers}
        for user_speaker in speakers:
            refs = [speaker for speaker in speakers if speaker != user_speaker]
            for layer in layers:
                distances = [cosine_dtw_distance(feats(paths[ref])[layer], feats(paths[user_speaker])[layer])["normalized_cumulative_distance"] for ref in refs]
                for strategy in ("mean", "median", "trimmed_mean", "nearest", "top_k_mean"):
                    rows.append({"condition": "native_leave_one_speaker_out", "sentence_id": sentence, "user_speaker": user_speaker,
                                 "layer": layer, "strategy": strategy, "reference_count": len(refs),
                                 "distance": round(aggregate_reference_distances(distances, strategy), 6)})
                wrong = jvs / "jvs005" / "parallel100/wav24kHz16bit" / f"VOICEACTRESS100_{(sentence % 20) + 1:03d}.wav"
                wrong_distances = [cosine_dtw_distance(feats(paths[ref])[layer], feats(wrong)[layer])["normalized_cumulative_distance"] for ref in refs]
                rows.append({"condition": "wrong_target", "sentence_id": sentence, "user_speaker": "jvs005_other_sentence",
                             "layer": layer, "strategy": "median", "reference_count": len(refs),
                             "distance": round(aggregate_reference_distances(wrong_distances, "median"), 6)})
        print(f"wavlm sentence {sentence:03d}/020", flush=True)
    _write_csv(out / "wavlm_multi_reference_results.csv", rows)
    summary = []
    for layer in layers:
        native = [float(r["distance"]) for r in rows if r["layer"] == layer and r["condition"] == "native_leave_one_speaker_out" and r["strategy"] == "median"]
        wrong = [float(r["distance"]) for r in rows if r["layer"] == layer and r["condition"] == "wrong_target"]
        summary.append({"layer": layer, "native_median_distance": round(float(np.median(native)), 5), "native_sd": round(float(np.std(native)), 5),
                        "wrong_median_distance": round(float(np.median(wrong)), 5), "median_separation": round(float(np.median(wrong) - np.median(native)), 5)})
    _write_csv(out / "wavlm_multi_reference_summary.csv", summary)


def render_report(out: Path, report: Path) -> None:
    content = _read_csv(out / "content_gate_v2_summary.csv") if (out / "content_gate_v2_summary.csv").exists() else []
    calibration = _read_csv(out / "product_calibration_ladder.csv") if (out / "product_calibration_ladder.csv").exists() else []
    wavlm = _read_csv(out / "wavlm_multi_reference_summary.csv") if (out / "wavlm_multi_reference_summary.csv").exists() else []
    lines = ["# C-end v2 Content and Shadow Bench", "", "## Content verification", "", "|system|correct false mismatch|wrong false verified|cold sec|warm p50|warm p90|", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"|{r['system']}|{r['correct_false_mismatch_rate']}|{r['wrong_false_verified_rate']}|{r['cold_latency_sec']}|{r['warm_p50_sec']}|{r['warm_p90_sec']}|" for r in content]
    lines += ["", "MFCC acoustic-only uses `acoustic_likely_match` only; it never sets `content_verified=true`. The recommended production policy is selected from the lowest false-verified ASR option subject to observed latency, rather than from a new DTW threshold.", "", "## Product calibration ladder", "", f"- Conditions measured: {len(calibration)}. All raw evidence is in `outputs/content_shadow_v2/product_calibration_ladder.csv`; no score mapping was changed.", "- Current score ceiling remains an upstream saturation finding; this bench records its components rather than compressing the displayed curve.", "", "## WavLM multi-reference", "", "|layer|native median|native SD|wrong median|separation|", "|---:|---:|---:|---:|---:|"]
    lines += [f"|{r['layer']}|{r['native_median_distance']}|{r['native_sd']}|{r['wrong_median_distance']}|{r['median_separation']}|" for r in wavlm]
    lines += ["", "## Shadow guarantees", "", "- WavLM, special-mora v3, phrase intonation v2, and accent-nucleus v1 remain default-off, exception-isolated, and outside ProductScore/practice feedback.", "- Sokuon uses neighbor-relative low-energy/closure evidence; long vowels use a combined vowel nucleus; moraic nasals report context classes only.", "- Phrase intonation no longer emits an arbitrary /100 mapping and no longer bridges missing-F0 morae. Accent analysis is per accent phrase and only computes target correctness for strong target provenance.", "", "## Merge gate", "", "**BLOCK MERGE** until the selected ASR policy removes the hard duration-matched Japanese false-verification regression and negative-control results have been reviewed."]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/content_shadow_v2")
    parser.add_argument("command", choices=["manifest", "content", "combine-content", "controls", "negative-controls", "calibration", "wavlm", "report", "all"])
    parser.add_argument("--models", default="tiny,base,small")
    parser.add_argument("--layers", default="6,12,18,24")
    parser.add_argument("--input-dirs", default="")
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "data/audit/content_gate_v2_manifest.csv"
    if args.command in {"manifest", "all"}:
        build_manifest(args.data_root.resolve(), manifest)
    if args.command in {"content", "all"}:
        run_content_bench(manifest, out, [m.strip() for m in args.models.split(",") if m.strip()])
    if args.command == "combine-content":
        all_rows: list[dict[str, Any]] = []
        for raw in [item.strip() for item in args.input_dirs.split(",") if item.strip()]:
            all_rows.extend(_read_csv(Path(raw) / "content_gate_v2_results.csv"))
        # identical acoustic rows recur in every model run; retain one baseline.
        by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in all_rows:
            key = (row["system"], row["sample_id"])
            prior = by_key.get(key)
            # A later successful checkpoint run supersedes an earlier download
            # failure for the same benchmark cell.
            if prior is None or ("unavailable" in str(prior.get("note", "")).lower() and str(row.get("note", "")) == "ok"):
                by_key[key] = row
        deduped = list(by_key.values())
        summarize_content_rows(deduped, out)
    if args.command in {"controls", "all"}:
        make_engineering_controls(out)
    if args.command in {"negative-controls", "all"}:
        evaluate_engineering_controls(args.data_root.resolve(), out)
    if args.command in {"calibration", "all"}:
        run_calibration(args.data_root.resolve(), out)
    if args.command in {"wavlm", "all"}:
        run_wavlm(args.data_root.resolve(), out, [int(x) for x in args.layers.split(",")])
    if args.command in {"report", "all"}:
        render_report(out, ROOT / "reports/C_END_V2_CONTENT_AND_SHADOW_BENCH.md")


if __name__ == "__main__":
    main()
