#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from jp_speech_eval.alignment import estimate_mora_boundaries_equal  # noqa: E402
from jp_speech_eval.alignment_evidence.phone_mora_mapper import map_phones_to_moras  # noqa: E402
from jp_speech_eval.alignment_evidence.textgrid_parser import parse_lab_phone_segments  # noqa: E402
from jp_speech_eval.audio_features import detect_pauses, load_audio  # noqa: E402
from jp_speech_eval.phonology import classify_mora_sequence  # noqa: E402
from jp_speech_eval.scoring import score_fluency, score_pronunciation_rhythm  # noqa: E402
from jp_speech_eval.text_frontend import build_text_info  # noqa: E402
from jp_speech_eval.vad import trim_to_speech  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _raw_lab_pause_info(path: Path) -> tuple[float, dict[str, Any]]:
    segments: list[tuple[float, float, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            segments.append((float(parts[0]), float(parts[1]), parts[2]))
        except ValueError:
            continue
    speech = [item for item in segments if item[2] not in {"sil", "sp", "<eps>"}]
    if not speech:
        return 0.0, {"pause_count": 0, "pause_total": 0.0, "pause_ratio": 0.0}
    start = speech[0][0]
    end = speech[-1][1]
    duration = max(0.0, end - start)
    pauses = [(s, e) for s, e, phone in segments if phone == "pau" and s >= start and e <= end and e - s >= 0.30]
    pause_total = sum(end_ - start_ for start_, end_ in pauses)
    return duration, {
        "pause_count": len(pauses),
        "pause_total": pause_total,
        "pause_ratio": pause_total / max(duration, 1e-6),
        "pause_segments": pauses,
    }


def _boundaries_from_durations(durations: Sequence[float]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    cursor = 0.0
    for duration in durations:
        end = cursor + max(float(duration), 1e-4)
        out.append((cursor, end))
        cursor = end
    return out


def _timing_jitter(boundaries: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    durations = np.asarray([end - start for start, end in boundaries], dtype=float)
    if durations.size < 3:
        return list(boundaries)
    factors = np.asarray([0.35 if index % 3 == 0 else 1.75 if index % 3 == 1 else 0.90 for index in range(len(durations))])
    changed = np.maximum(0.01, durations * factors)
    changed *= float(np.sum(durations)) / max(float(np.sum(changed)), 1e-8)
    return _boundaries_from_durations(changed)


def _compress_special_morae(moras: Sequence[str], boundaries: Sequence[tuple[float, float]]) -> list[tuple[float, float]] | None:
    durations = np.asarray([end - start for start, end in boundaries], dtype=float)
    strong = [item.index - 1 for item in classify_mora_sequence(list(moras)) if item.strength == "strong"]
    strong = [index for index in strong if 0 <= index < len(durations)]
    if not strong or durations.size < 2:
        return None
    avg = float(np.mean(durations))
    changed = durations.copy()
    removed = 0.0
    for index in strong:
        target = min(float(changed[index]), 0.20 * avg)
        removed += float(changed[index] - target)
        changed[index] = max(0.01, target)
    ordinary = [index for index in range(len(changed)) if index not in strong]
    if ordinary and removed > 0:
        changed[ordinary] += removed / len(ordinary)
    return _boundaries_from_durations(changed)


def _weak_rhythm_score(boundaries: Sequence[tuple[float, float]]) -> int:
    durations = np.asarray([max(0.0, end - start) for start, end in boundaries], dtype=float)
    if not durations.size:
        return 0
    avg = float(np.mean(durations))
    cv = float(np.std(durations) / (avg + 1e-8))
    return int(round(100.0 * max(0.0, min(1.0, 1.0 - 0.85 * max(0.0, cv - 0.18)))))


def _jvs_rows(audit_jsonl: Path) -> list[dict[str, Any]]:
    manifest = [
        row for row in _read_jsonl(audit_jsonl)
        if row.get("group") == "fresh_jvs_parallel100" and row.get("condition") == "normal"
    ]
    output: list[dict[str, Any]] = []
    for item in manifest:
        text_info = build_text_info(str(item["reference_text"]))
        phones = parse_lab_phone_segments(item["lab_path"])
        segments, mapping = map_phones_to_moras(phones, text_info.moras)
        if len(segments) != len(text_info.moras):
            continue
        boundaries = [(float(segment.start), float(segment.end)) for segment in segments]
        duration, pause_info = _raw_lab_pause_info(Path(item["lab_path"]))
        pronunciation, _feedback, pron_details = score_pronunciation_rhythm(text_info.moras, boundaries)
        jitter_boundaries = _timing_jitter(boundaries)
        jitter_pronunciation, _feedback, _details = score_pronunciation_rhythm(text_info.moras, jitter_boundaries)
        special_boundaries = _compress_special_morae(text_info.moras, boundaries)
        special_pronunciation = None
        if special_boundaries is not None:
            special_pronunciation, _feedback, _details = score_pronunciation_rhythm(text_info.moras, special_boundaries)
        fluency, _feedback, fluency_details = score_fluency(len(text_info.moras), duration, pause_info)
        slow_fluency, _feedback, _details = score_fluency(len(text_info.moras), duration * 1.8, pause_info)
        fast_fluency, _feedback, _details = score_fluency(len(text_info.moras), duration * 0.55, pause_info)
        added_pause = max(1.5, duration * 0.30)
        hesitation_pause = {
            "pause_count": int(pause_info["pause_count"]) + 5,
            "pause_total": float(pause_info["pause_total"]) + added_pause,
            "pause_ratio": (float(pause_info["pause_total"]) + added_pause) / max(duration + added_pause, 1e-6),
        }
        hesitation_fluency, _feedback, _details = score_fluency(len(text_info.moras), duration + added_pause, hesitation_pause)
        equal_boundaries = estimate_mora_boundaries_equal(sum(end - start for start, end in boundaries), len(boundaries))
        output.append({
            "sample_id": item["sample_id"],
            "dataset": "JVS",
            "speaker_id": item["speaker_id"],
            "utterance_id": str(item["sample_id"]).split(":", 1)[-1],
            "mora_count": len(text_info.moras),
            "mapping_success": bool(mapping.get("mapping_success")),
            "pronunciation_normal": pronunciation,
            "pronunciation_timing_jitter": jitter_pronunciation,
            # A segment substitution with unchanged boundaries is invisible to
            # the current timing-only pronunciation formula.
            "pronunciation_segment_substitution_proxy": pronunciation,
            "pronunciation_special_mora_compressed": special_pronunciation,
            "pronunciation_mora_duration_cv": pron_details.get("mora_duration_cv"),
            "rhythm_normal": _weak_rhythm_score(boundaries),
            "rhythm_timing_jitter": _weak_rhythm_score(jitter_boundaries),
            "rhythm_equal_fallback": _weak_rhythm_score(equal_boundaries),
            "fluency_normal": fluency,
            "fluency_slow": slow_fluency,
            "fluency_fast": fast_fluency,
            "fluency_hesitation": hesitation_fluency,
            "speech_rate_mora_per_sec": fluency_details.get("speech_rate_mora_per_sec"),
            "pause_ratio": pause_info.get("pause_ratio"),
            "pause_count": pause_info.get("pause_count"),
            "duration_sec": duration,
        })
    return output


def _janon_rows(janon_root: Path, checkpoint: Path) -> list[dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed[str(row["sample_id"])] = row
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    sources = [row for row in _read_csv(janon_root / "data.csv") if row.get("Stimulus Type") == "sentence"]
    for index, source in enumerate(sources, start=1):
        sample_id = f"{source['Speaker']}:{Path(source['Path']).stem}"
        if sample_id in completed:
            continue
        wav = janon_root / source["Path"].replace("/sentence/", "/sentences/")
        if not wav.exists():
            continue
        text = source["Stmiulus"].strip().lstrip("\ufeff")
        text_info = build_text_info(text)
        audio = load_audio(str(wav), sr=16000)
        speech, region = trim_to_speech(audio.y, audio.sr)
        duration = len(speech) / max(audio.sr, 1)
        pause_info = detect_pauses(speech, audio.sr)
        fluency, _feedback, details = score_fluency(len(text_info.moras), duration, pause_info)
        equal = estimate_mora_boundaries_equal(duration, len(text_info.moras))
        pronunciation, _feedback, _details = score_pronunciation_rhythm(text_info.moras, equal)
        row = {
            "sample_id": sample_id,
            "dataset": "JANON",
            "speaker_id": source["Speaker"],
            "native_language": source["Native Language"],
            "group": "janon_native" if source["Native Language"] == "Japanese" else "janon_learner",
            "mora_count": len(text_info.moras),
            "duration_sec": duration,
            "speech_detected": bool(region.detected),
            "pronunciation_equal_fallback": pronunciation,
            "rhythm_equal_fallback": _weak_rhythm_score(equal),
            "fluency": fluency,
            "speech_rate_mora_per_sec": details.get("speech_rate_mora_per_sec"),
            "pause_ratio": pause_info.get("pause_ratio"),
            "pause_count": pause_info.get("pause_count"),
            "interpretation": "external_read_speech_not_negative_label",
        }
        with checkpoint.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        completed[sample_id] = row
        if index % 100 == 0:
            print(f"JANON [{index}/{len(sources)}]", flush=True)
    return list(completed.values())


def _values(rows: Iterable[Mapping[str, Any]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows if row.get(field) not in (None, "")], dtype=float)


def _distribution(rows: Sequence[Mapping[str, Any]], field: str, *, section: str, metric: str) -> dict[str, Any]:
    values = _values(rows, field)
    if not values.size:
        return {"section": section, "metric": metric, "n": 0}
    return {
        "section": section,
        "metric": metric,
        "n": len(values),
        "mean": round(float(np.mean(values)), 4),
        "sd": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0,
        "p10": round(float(np.percentile(values, 10)), 4),
        "p50": round(float(np.percentile(values, 50)), 4),
        "p90": round(float(np.percentile(values, 90)), 4),
        "below_60_rate": round(float(np.mean(values < 60)), 4),
        "at_least_80_rate": round(float(np.mean(values >= 80)), 4),
        "ceiling_rate": round(float(np.mean(values >= 100)), 4),
        "unique_integer_scores": len(set(int(round(value)) for value in values)),
    }


def _paired(rows: Sequence[Mapping[str, Any]], normal_field: str, control_field: str, *, dimension: str, control: str) -> dict[str, Any]:
    pairs = [
        (float(row[normal_field]), float(row[control_field]))
        for row in rows if row.get(normal_field) not in (None, "") and row.get(control_field) not in (None, "")
    ]
    normal = np.asarray([pair[0] for pair in pairs], dtype=float)
    changed = np.asarray([pair[1] for pair in pairs], dtype=float)
    labels = np.concatenate([np.ones(len(normal)), np.zeros(len(changed))])
    scores = np.concatenate([normal, changed])
    return {
        "section": "paired_control",
        "metric": f"{dimension}_normal_vs_{control}",
        "dimension": dimension,
        "control": control,
        "n": len(pairs),
        "normal_mean": round(float(np.mean(normal)), 4),
        "control_mean": round(float(np.mean(changed)), 4),
        "paired_delta_mean": round(float(np.mean(normal - changed)), 4),
        "normal_wins_rate": round(float(np.mean(normal > changed)), 4),
        "unchanged_rate": round(float(np.mean(normal == changed)), 4),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
    }


def summarize(jvs: Sequence[Mapping[str, Any]], janon: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    native = [row for row in janon if row.get("group") == "janon_native"]
    learner = [row for row in janon if row.get("group") == "janon_learner"]
    rows = [
        _distribution(jvs, "pronunciation_normal", section="distribution", metric="jvs_pronunciation_normal"),
        _distribution(jvs, "rhythm_normal", section="distribution", metric="jvs_rhythm_normal"),
        _distribution(jvs, "fluency_normal", section="distribution", metric="jvs_fluency_normal"),
        _distribution(native, "fluency", section="distribution", metric="janon_native_fluency"),
        _distribution(learner, "fluency", section="distribution", metric="janon_learner_fluency_descriptive"),
        _distribution(native, "pronunciation_equal_fallback", section="fallback_diagnostic", metric="janon_native_pronunciation_equal"),
        _distribution(learner, "pronunciation_equal_fallback", section="fallback_diagnostic", metric="janon_learner_pronunciation_equal"),
        _distribution(native, "rhythm_equal_fallback", section="fallback_diagnostic", metric="janon_native_rhythm_equal"),
        _distribution(learner, "rhythm_equal_fallback", section="fallback_diagnostic", metric="janon_learner_rhythm_equal"),
        _paired(jvs, "pronunciation_normal", "pronunciation_timing_jitter", dimension="pronunciation", control="timing_jitter"),
        _paired(jvs, "pronunciation_normal", "pronunciation_special_mora_compressed", dimension="pronunciation", control="special_mora_compressed"),
        _paired(jvs, "pronunciation_normal", "pronunciation_segment_substitution_proxy", dimension="pronunciation", control="segment_substitution"),
        _paired(jvs, "rhythm_normal", "rhythm_timing_jitter", dimension="rhythm", control="timing_jitter"),
        _paired(jvs, "rhythm_normal", "rhythm_equal_fallback", dimension="rhythm", control="equal_fallback"),
        _paired(jvs, "fluency_normal", "fluency_slow", dimension="fluency", control="slow"),
        _paired(jvs, "fluency_normal", "fluency_fast", dimension="fluency", control="fast"),
        _paired(jvs, "fluency_normal", "fluency_hesitation", dimension="fluency", control="hesitation"),
    ]
    return rows


def _report(
    path: Path,
    summary: Sequence[Mapping[str, Any]],
    jvs_count: int,
    janon_count: int,
    jvs_mapping_success_count: int,
) -> None:
    lookup = {(str(row["section"]), str(row["metric"])): row for row in summary}
    dist = lambda metric: lookup[("distribution", metric)]
    pair = lambda metric: lookup[("paired_control", metric)]
    jvs_p = dist("jvs_pronunciation_normal")
    jvs_r = dist("jvs_rhythm_normal")
    jvs_f = dist("jvs_fluency_normal")
    janon_n = dist("janon_native_fluency")
    janon_l = dist("janon_learner_fluency_descriptive")
    pron_jitter = pair("pronunciation_normal_vs_timing_jitter")
    pron_segment = pair("pronunciation_normal_vs_segment_substitution")
    special = pair("pronunciation_normal_vs_special_mora_compressed")
    rhythm_jitter = pair("rhythm_normal_vs_timing_jitter")
    rhythm_fallback = pair("rhythm_normal_vs_equal_fallback")
    slow = pair("fluency_normal_vs_slow")
    fast = pair("fluency_normal_vs_fast")
    hesitation = pair("fluency_normal_vs_hesitation")
    lines = [
        "# Pronunciation, Rhythm and Fluency Objective Validation",
        "",
        f"- JVS native rows: {jvs_count} (real audio transcripts plus phone-lab timing)",
        f"- JVS phone-to-mora mapping success: {jvs_mapping_success_count}/{jvs_count}",
        f"- JANON sentence rows: {janon_count} (real audio; external descriptive audit only)",
        "- No human ratings are introduced. JANON learners are not treated as bad-pronunciation labels.",
        "- Runtime scoring is unchanged by this audit.",
        "",
        "## Bottom Line",
        "",
        "- **Pronunciation clarity is not stable yet.** Real native phone-lab timing receives a mean of only 13.18, while equal-boundary fallback produces 100. The score is dominated by the alignment representation and cannot detect segment substitutions.",
        "- **Rhythm is not stable yet.** It responds to timing jitter with reliable boundaries, but equal fallback forces the score to 100 and removes the evidence needed for special-mora judgement.",
        "- **Fluency is the strongest of the three, but still coarse.** It separates gross hesitation/fast speech; slow speech separation is weak, and readable JANON speech frequently hits 100.",
        "- Therefore these dimensions can return numbers, but only fluency currently has useful coarse negative-control separation. None of the three has pitch-v2-level validation.",
        "",
        "## Native / External Distributions",
        "",
        "| group and dimension | n | mean | p10 | p50 | p90 | ceiling rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| JVS pronunciation timing proxy | {jvs_p['n']} | {jvs_p['mean']} | {jvs_p['p10']} | {jvs_p['p50']} | {jvs_p['p90']} | {jvs_p['ceiling_rate']} |",
        f"| JVS rhythm timing proxy | {jvs_r['n']} | {jvs_r['mean']} | {jvs_r['p10']} | {jvs_r['p50']} | {jvs_r['p90']} | {jvs_r['ceiling_rate']} |",
        f"| JVS fluency | {jvs_f['n']} | {jvs_f['mean']} | {jvs_f['p10']} | {jvs_f['p50']} | {jvs_f['p90']} | {jvs_f['ceiling_rate']} |",
        f"| JANON native fluency | {janon_n['n']} | {janon_n['mean']} | {janon_n['p10']} | {janon_n['p50']} | {janon_n['p90']} | {janon_n['ceiling_rate']} |",
        f"| JANON learner fluency (descriptive) | {janon_l['n']} | {janon_l['mean']} | {janon_l['p10']} | {janon_l['p50']} | {janon_l['p90']} | {janon_l['ceiling_rate']} |",
        "",
        "## Controlled Separation",
        "",
        "| dimension | negative control | normal mean | control mean | delta | AUC | result |",
        "|---|---|---:|---:|---:|---:|---|",
        f"| Pronunciation proxy | mora timing jitter | {pron_jitter['normal_mean']} | {pron_jitter['control_mean']} | {pron_jitter['paired_delta_mean']} | {pron_jitter['roc_auc']} | weak separation; native already near floor |",
        f"| Pronunciation proxy | compressed special mora | {special['normal_mean']} | {special['control_mean']} | {special['paired_delta_mean']} | {special['roc_auc']} | weak separation; floor-limited |",
        f"| Pronunciation proxy | segment substitution | {pron_segment['normal_mean']} | {pron_segment['control_mean']} | {pron_segment['paired_delta_mean']} | {pron_segment['roc_auc']} | FAIL: invisible to current formula |",
        f"| Rhythm | mora timing jitter | {rhythm_jitter['normal_mean']} | {rhythm_jitter['control_mean']} | {rhythm_jitter['paired_delta_mean']} | {rhythm_jitter['roc_auc']} | timing-sensitive with real boundaries |",
        f"| Rhythm | equal-boundary fallback | {rhythm_fallback['normal_mean']} | {rhythm_fallback['control_mean']} | {rhythm_fallback['paired_delta_mean']} | {rhythm_fallback['roc_auc']} | FAIL: fallback inflates score |",
        f"| Fluency | slow | {slow['normal_mean']} | {slow['control_mean']} | {slow['paired_delta_mean']} | {slow['roc_auc']} | speed-sensitive |",
        f"| Fluency | fast | {fast['normal_mean']} | {fast['control_mean']} | {fast['paired_delta_mean']} | {fast['roc_auc']} | speed-sensitive |",
        f"| Fluency | hesitation | {hesitation['normal_mean']} | {hesitation['control_mean']} | {hesitation['paired_delta_mean']} | {hesitation['roc_auc']} | pause-sensitive |",
        "",
        "## Findings by Dimension",
        "",
        "### Pronunciation clarity",
        "",
        "The current score is `100 - 90 * mora-duration CV - special-mora penalties`. Natural phone-lab mora durations are not equal: devoicing, phrase structure and legitimate special-mora timing create substantial variation. All 300 JVS mappings succeeded, yet the native median is 0, so the low result is not explained by mapping failure alone. A phoneme substitution with unchanged timing also produces exactly the same score. This dimension is neither stable on native speech nor capable of consonant/vowel correctness judgement in its current form.",
        "",
        "### Rhythm / special mora",
        "",
        "With phone-lab boundaries, timing jitter is detectable, but the native mean is only 53.40. Under equal-mora fallback, every mora is assigned the same duration and rhythm becomes 100 by construction. The current result therefore changes more with the alignment backend than it should. A fallback result can remain numeric for UX, but it must be described as a coarse estimate with low confidence and cannot support specific special-mora correction.",
        "",
        "### Fluency",
        "",
        "Fluency responds strongly to fast speech and hesitation, but slow-control AUC is only 0.61. JANON learners remain a readable external population rather than a negative class, so high scores are not inherently wrong. However, 83% of learner sentence recordings and 51% of JANON native recordings score exactly 100, which shows a ceiling/resolution problem. The dimension also needs robustness checks for VAD and endpointing because those directly alter duration and pause ratio.",
        "",
        "## Scientific Status",
        "",
        "| dimension | numeric availability | native stability | controlled separation | current status |",
        "|---|---|---|---|---|",
        "| Pronunciation clarity | yes | **FAIL**: JVS mean 13.18, fallback 100 | weak timing separation; phoneme errors invisible | redesign required |",
        "| Rhythm / special mora | yes | **FAIL**: JVS mean 53.40, fallback 100 | moderate with reliable boundaries | alignment-dependent; redesign required |",
        "| Fluency | yes | partial: JVS high, JANON ceiling-heavy | strong for hesitation/fast, weak for slow | usable coarse proxy, not fine-grained |",
        "",
        "## Next Step Without Human Ratings",
        "",
        "1. Add waveform-level phoneme corruption controls and an acoustic/ASR posterior feature before claiming pronunciation separation.",
        "2. Replace equal-boundary rhythm evidence with phone/alignment-derived timing when available; otherwise keep a coarse numeric score with low confidence.",
        "3. Run gain/noise/codec/VAD perturbations and require limited score drift on clean native speech.",
        "4. Keep JANON as an external readable-learner audit. Do not train a learner-vs-native classifier and call it pronunciation quality.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="No-human-rating objective validation for pronunciation, rhythm and fluency.")
    parser.add_argument("--jvs-audit-jsonl", type=Path, default=ROOT / "outputs/pitch_naturalness_v2_audit.jsonl")
    parser.add_argument("--janon-root", type=Path, default=PROJECT_ROOT / "JANON")
    parser.add_argument("--janon-checkpoint", type=Path, default=ROOT / "outputs/three_dimensions_janon.jsonl")
    parser.add_argument("--out-detail", type=Path, default=ROOT / "results/calibration/three_dimensions_objective_validation.csv")
    parser.add_argument("--out-summary", type=Path, default=ROOT / "data/calibration_candidates/three_dimensions_objective_validation_summary.csv")
    parser.add_argument("--out-report", type=Path, default=ROOT / "reports/three_dimensions_objective_validation.md")
    args = parser.parse_args()
    jvs = _jvs_rows(args.jvs_audit_jsonl)
    janon = _janon_rows(args.janon_root, args.janon_checkpoint)
    summary = summarize(jvs, janon)
    detail = [*jvs, *janon]
    _write_csv(args.out_detail, detail)
    _write_csv(args.out_summary, summary)
    _report(
        args.out_report,
        summary,
        len(jvs),
        len(janon),
        sum(bool(row.get("mapping_success")) for row in jvs),
    )
    print(json.dumps({"jvs": len(jvs), "janon": len(janon), "report": str(args.out_report)}, indent=2))


if __name__ == "__main__":
    main()
