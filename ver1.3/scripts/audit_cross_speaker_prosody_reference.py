#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.alignment_evidence.phone_mora_mapper import map_phones_to_moras  # noqa: E402
from jp_speech_eval.alignment_evidence.textgrid_parser import parse_lab_phone_segments  # noqa: E402
from jp_speech_eval.audio_features import extract_f0, load_audio, log_f0_normalize, median_f0_by_mora  # noqa: E402
from jp_speech_eval.prosody_reference_cache import smooth_f0_by_mora  # noqa: E402
from jp_speech_eval.scoring import score_prosody  # noqa: E402
from jp_speech_eval.text_frontend import build_text_info  # noqa: E402


def pattern_from_f0(f0_by_mora: Sequence[float]) -> List[str]:
    z = log_f0_normalize(np.asarray(f0_by_mora, dtype=float))
    return ["H" if np.isfinite(v) and v >= 0 else "L" if np.isfinite(v) else "?" for v in z]


def flat_f0(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr) & (arr > 0)
    out = np.full_like(arr, np.nan, dtype=float)
    if int(np.sum(valid)):
        out[valid] = float(np.median(arr[valid]))
    return out.tolist()


def shuffled_f0(values: Sequence[float], *, seed: int = 0) -> List[float]:
    arr = np.asarray(values, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(arr) & (arr > 0))
    out = arr.copy()
    if valid_idx.size >= 2:
        rng = np.random.default_rng(seed)
        shuffled = out[valid_idx].copy()
        for _ in range(8):
            rng.shuffle(shuffled)
            if not np.allclose(shuffled, out[valid_idx], equal_nan=True):
                break
        out[valid_idx] = shuffled
    return out.tolist()


def wrong_drop_f0(values: Sequence[float], reference: Sequence[float], accent_phrases: Sequence[Mapping[str, Any]]) -> List[float]:
    out = np.asarray(values, dtype=float).copy()
    ref = np.asarray(reference, dtype=float)
    if out.size < 2:
        return out.tolist()
    drop_indices: List[int] = []
    offset = 0
    for phrase in accent_phrases:
        phrase_moras = list(phrase.get("moras") or [])
        n = len(phrase_moras)
        accent_position = int(phrase.get("accent_position", 0) or 0)
        if accent_position > 0:
            idx = offset + accent_position - 1
            if 0 <= idx < out.size - 1:
                drop_indices.append(idx)
        offset += n
    if not drop_indices:
        ref_z = log_f0_normalize(ref)
        for idx in range(min(len(ref_z) - 1, len(out) - 1)):
            if np.isfinite(ref_z[idx]) and np.isfinite(ref_z[idx + 1]) and ref_z[idx + 1] < ref_z[idx] - 0.20:
                drop_indices.append(idx)
                break
    if not drop_indices:
        return shuffled_f0(values, seed=13)
    valid = np.isfinite(out) & (out > 0)
    median = float(np.median(out[valid])) if int(np.sum(valid)) else 120.0
    for idx in drop_indices:
        if not np.isfinite(out[idx]) or out[idx] <= 0:
            out[idx] = median
        out[idx + 1] = max(float(out[idx]) * 1.08, median * 1.04)
    return out.tolist()


def low_f0_coverage(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(arr) & (arr > 0))
    if valid.size:
        out[valid[0]] = arr[valid[0]]
    if valid.size > 1:
        out[valid[-1]] = arr[valid[-1]]
    return out.tolist()


def f0_coverage(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.mean(np.isfinite(arr) & (arr > 0)))


def _jvs_by_utterance(jvs_root: Path, *, max_speakers: int | None) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    speakers = sorted(jvs_root.glob("jvs*"))
    if max_speakers is not None:
        speakers = speakers[:max_speakers]
    for speaker_dir in speakers:
        transcript_path = speaker_dir / "parallel100" / "transcripts_utf8.txt"
        wav_dir = speaker_dir / "parallel100" / "wav24kHz16bit"
        lab_dir = speaker_dir / "parallel100" / "lab" / "mon"
        if not transcript_path.exists() or not wav_dir.exists() or not lab_dir.exists():
            continue
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            utt_id, text = line.split(":", 1)
            wav = wav_dir / f"{utt_id}.wav"
            lab = lab_dir / f"{utt_id}.lab"
            if wav.exists() and lab.exists():
                grouped[utt_id].append({
                    "speaker_id": speaker_dir.name,
                    "utterance_id": utt_id,
                    "target_text": text.strip(),
                    "audio_path": str(wav),
                    "lab_path": str(lab),
                })
    return grouped


def _select_pairs(jvs_root: Path, *, max_speakers: int | None, max_pairs: int) -> List[Dict[str, Any]]:
    grouped = _jvs_by_utterance(jvs_root, max_speakers=max_speakers)
    pairs: List[Dict[str, Any]] = []
    for utt_id in sorted(grouped):
        items = sorted(grouped[utt_id], key=lambda item: item["speaker_id"])
        if len(items) < 2:
            continue
        pairs.append({
            "pair_id": f"{items[0]['speaker_id']}->{items[1]['speaker_id']}:{utt_id}",
            "reference": items[0],
            "user": items[1],
        })
        if len(pairs) >= max_pairs:
            break
    return pairs


def _lab_f0(item: Mapping[str, str], moras: Sequence[str], *, sample_rate: int) -> tuple[List[float] | None, Dict[str, Any]]:
    lab_path = Path(item.get("lab_path") or "")
    if not lab_path.exists():
        return None, {"ok": False, "reason": "missing_lab"}
    phones = parse_lab_phone_segments(lab_path)
    segments, mapping = map_phones_to_moras(phones, moras)
    if len(segments) != len(moras):
        return None, {
            "ok": False,
            "reason": "phone_mora_mapping_failed",
            "mapping_warning_flags": ";".join(str(x) for x in mapping.get("mapping_warning_flags") or []),
        }
    audio = load_audio(str(item["audio_path"]), sr=sample_rate)
    times, f0, method = extract_f0(audio.y, audio.sr)
    boundaries = [(float(seg.start), float(seg.end)) for seg in segments]
    return median_f0_by_mora(times, f0, boundaries), {
        "ok": True,
        "f0_method": method,
        "mapping_success": bool(mapping.get("mapping_success")),
        "mapping_warning_flags": ";".join(str(x) for x in mapping.get("mapping_warning_flags") or []),
        "boundary_min_sec": round(min(e - s for s, e in boundaries), 4) if boundaries else None,
        "boundary_max_sec": round(max(e - s for s, e in boundaries), 4) if boundaries else None,
    }


def _score_row(
    *,
    pair: Mapping[str, Any],
    text_info: Any,
    mode: str,
    user_f0: Sequence[float],
    target_pattern: Sequence[str],
    pitch_target_source: str,
    reference_f0: Sequence[float] | None,
    notes: str,
) -> Dict[str, Any]:
    score, _feedback, details = score_prosody(
        moras=text_info.moras,
        target_pattern=list(target_pattern),
        f0_by_mora=list(user_f0),
        reference_f0_by_mora=list(reference_f0) if reference_f0 is not None else None,
        pitch_target_source=pitch_target_source,
        reference_f0_target_source=pitch_target_source if reference_f0 is not None else None,
        is_question=bool(text_info.is_question),
        accent_phrases=text_info.accent_phrases,
    )
    ref = pair["reference"]
    user = pair["user"]
    components = details.get("prosody_score_components") or {}
    return {
        "pair_id": pair["pair_id"],
        "utterance_id": ref["utterance_id"],
        "target_text": ref["target_text"],
        "target_kana": text_info.kana,
        "mora_count": len(text_info.moras),
        "reference_speaker": ref["speaker_id"],
        "user_speaker": user["speaker_id"],
        "reference_audio_path": ref["audio_path"],
        "user_audio_path": user["audio_path"],
        "mode": mode,
        "alignment_mode": "lab_phone_mora",
        "content_status": "same_transcript_control",
        "prosody_score": score,
        "f0_coverage": round(f0_coverage(user_f0), 4),
        "pitch_target_source": details.get("pitch_target_source"),
        "pitch_target_reliability": "reliable" if pitch_target_source == "reference_audio_f0_cache" else "heuristic",
        "contour_corr": details.get("contour_corr"),
        "transition_agreement": details.get("transition_agreement"),
        "hl_match": details.get("hl_match_rate", details.get("hl_match")),
        "accent_drop_match": details.get("accent_drop_agreement"),
        "final_intonation_score": details.get("final_score"),
        "component_contour": components.get("contour"),
        "component_transition": components.get("transition"),
        "component_hl": components.get("hl"),
        "component_final": components.get("final"),
        "unavailable_reason": details.get("note") or "",
        "notes": notes,
    }


def audit_pairs(jvs_root: Path, *, max_speakers: int | None, max_pairs: int, sample_rate: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pair in _select_pairs(jvs_root, max_speakers=max_speakers, max_pairs=max_pairs):
        text_info = build_text_info(pair["reference"]["target_text"])
        ref_f0, ref_meta = _lab_f0(pair["reference"], text_info.moras, sample_rate=sample_rate)
        user_f0, user_meta = _lab_f0(pair["user"], text_info.moras, sample_rate=sample_rate)
        if ref_f0 is None or user_f0 is None:
            rows.append({
                "pair_id": pair["pair_id"],
                "mode": "alignment_unavailable",
                "reference_speaker": pair["reference"]["speaker_id"],
                "user_speaker": pair["user"]["speaker_id"],
                "utterance_id": pair["reference"]["utterance_id"],
                "target_text": pair["reference"]["target_text"],
                "notes": f"ref={ref_meta.get('reason')}; user={user_meta.get('reason')}",
            })
            continue
        ref_smooth = smooth_f0_by_mora(ref_f0)
        ref_pattern = pattern_from_f0(ref_smooth)
        rows.append(_score_row(
            pair=pair,
            text_info=text_info,
            mode="native_cross_speaker_reference_audio_f0_cache",
            user_f0=user_f0,
            target_pattern=ref_pattern,
            pitch_target_source="reference_audio_f0_cache",
            reference_f0=ref_smooth,
            notes="speaker A lab-timed native F0 contour used as verified reference cache target; speaker B same sentence scored as user",
        ))
        rows.append(_score_row(
            pair=pair,
            text_info=text_info,
            mode="native_cross_speaker_openjtalk_target",
            user_f0=user_f0,
            target_pattern=text_info.target_pitch,
            pitch_target_source=text_info.pitch_target_source,
            reference_f0=None,
            notes="same speaker B native audio scored against OpenJTalk accent phrase target",
        ))
        rows.append(_score_row(
            pair=pair,
            text_info=text_info,
            mode="flat_pitch_correct_content",
            user_f0=flat_f0(user_f0),
            target_pattern=ref_pattern,
            pitch_target_source="reference_audio_f0_cache",
            reference_f0=ref_smooth,
            notes="semi-audio counterfactual: real same-sentence timing/content, flattened user F0",
        ))
        rows.append(_score_row(
            pair=pair,
            text_info=text_info,
            mode="shuffled_random_pitch_correct_content",
            user_f0=shuffled_f0(user_f0, seed=sum(ord(ch) for ch in pair["pair_id"])),
            target_pattern=ref_pattern,
            pitch_target_source="reference_audio_f0_cache",
            reference_f0=ref_smooth,
            notes="semi-audio counterfactual: real same-sentence timing/content, shuffled valid user F0 values",
        ))
        rows.append(_score_row(
            pair=pair,
            text_info=text_info,
            mode="wrong_accent_drop_correct_content",
            user_f0=wrong_drop_f0(user_f0, ref_smooth, text_info.accent_phrases),
            target_pattern=ref_pattern,
            pitch_target_source="reference_audio_f0_cache",
            reference_f0=ref_smooth,
            notes="semi-audio counterfactual: same content/timing with accent-drop transition suppressed or contradicted",
        ))
        rows.append(_score_row(
            pair=pair,
            text_info=text_info,
            mode="low_f0_coverage_correct_content",
            user_f0=low_f0_coverage(user_f0),
            target_pattern=ref_pattern,
            pitch_target_source="reference_audio_f0_cache",
            reference_f0=ref_smooth,
            notes="semi-audio counterfactual: same content/timing with insufficient voiced mora F0",
        ))
    return rows


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(float(statistics.fmean(values)), 4),
        "min": round(float(min(values)), 4),
        "p50": round(float(ordered[len(ordered) // 2]), 4),
        "max": round(float(max(values)), 4),
    }


def _summary_by_mode(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = _num(row.get(field))
        if value is not None:
            grouped[str(row.get("mode") or "unknown")].append(value)
    return {mode: _summary(values) for mode, values in grouped.items()}


def _paired_delta(rows: Sequence[Mapping[str, Any]], reference_mode: str, other_mode: str) -> Dict[str, Any]:
    by_pair: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[str(row.get("pair_id"))][str(row.get("mode"))] = row
    deltas: List[float] = []
    for modes in by_pair.values():
        ref = _num((modes.get(reference_mode) or {}).get("prosody_score"))
        other = _num((modes.get(other_mode) or {}).get("prosody_score"))
        if ref is not None and other is not None:
            deltas.append(ref - other)
    out = _summary(deltas)
    out["comparison"] = f"{reference_mode} - {other_mode}"
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _report_table(summary: Mapping[str, Mapping[str, Any]]) -> List[str]:
    lines = [
        "| mode | n | mean | min | p50 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, row in sorted(summary.items()):
        lines.append(
            f"| {mode} | {row.get('n', 0)} | {row.get('mean', '')} | {row.get('min', '')} | {row.get('p50', '')} | {row.get('max', '')} |"
        )
    return lines


def _write_report(path: Path, rows: Sequence[Mapping[str, Any]], *, jvs_root: Path, max_pairs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    score_summary = _summary_by_mode(rows, "prosody_score")
    comparisons = [
        _paired_delta(rows, "native_cross_speaker_reference_audio_f0_cache", mode)
        for mode in [
            "native_cross_speaker_openjtalk_target",
            "flat_pitch_correct_content",
            "shuffled_random_pitch_correct_content",
            "wrong_accent_drop_correct_content",
            "low_f0_coverage_correct_content",
        ]
    ]
    ref_mean = score_summary.get("native_cross_speaker_reference_audio_f0_cache", {}).get("mean")
    openjtalk_mean = score_summary.get("native_cross_speaker_openjtalk_target", {}).get("mean")
    flat_mean = score_summary.get("flat_pitch_correct_content", {}).get("mean")
    shuffled_mean = score_summary.get("shuffled_random_pitch_correct_content", {}).get("mean")
    wrong_drop_mean = score_summary.get("wrong_accent_drop_correct_content", {}).get("mean")
    lines = [
        "# Cross-speaker prosody reference sanity",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- jvs_root: `{jvs_root}`",
        f"- requested_pairs: {max_pairs}",
        f"- rows: {len(rows)}",
        "- method: semi-audio; real JVS audio and lab timing provide mora-level F0, then counterfactuals replace F0 only.",
        "",
        "## Prosody Score by Mode",
        "",
        *_report_table(score_summary),
        "",
        "## Paired Deltas",
        "",
        "| comparison | n | mean | min | p50 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row.get('comparison')} | {row.get('n', 0)} | {row.get('mean', '')} | {row.get('min', '')} | {row.get('p50', '')} | {row.get('max', '')} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if isinstance(ref_mean, (int, float)) and isinstance(openjtalk_mean, (int, float)):
        if ref_mean > openjtalk_mean:
            lines.append("- Cross-speaker reference-audio F0 target scores higher than OpenJTalk target in this sample.")
        else:
            lines.append("- Cross-speaker reference-audio F0 target does not beat OpenJTalk in this sample; inspect reference timing and speaker pitch-normalization.")
    if isinstance(ref_mean, (int, float)) and isinstance(flat_mean, (int, float)):
        lines.append("- Flat counterfactual is lower than native reference target." if ref_mean > flat_mean else "- Flat counterfactual is not lower; pitch discrimination is insufficient here.")
    if isinstance(ref_mean, (int, float)) and isinstance(shuffled_mean, (int, float)):
        lines.append("- Shuffled/random counterfactual is lower than native reference target." if ref_mean > shuffled_mean else "- Shuffled/random counterfactual is not lower; pitch discrimination is insufficient here.")
    if isinstance(ref_mean, (int, float)) and isinstance(wrong_drop_mean, (int, float)):
        lines.append("- Wrong-drop counterfactual is lower than native reference target." if ref_mean > wrong_drop_mean else "- Wrong-drop counterfactual is not lower; accent-drop sensitivity needs more work.")
    lines.extend([
        "- Content is controlled by same JVS transcript; no ASR/TTS/content-gate behavior is changed by this audit.",
        "- This audit does not make calibration active.",
        "",
        "## Calibration Readiness",
        "",
        "Not ready. This sanity check is stronger than component-only tests, but calibration should wait for verified packaged reference audio and broader cross-speaker/audio-level negative controls.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cross-speaker same-sentence prosody reference targets on JVS.")
    parser.add_argument("--jvs-root", default=str(ROOT.parent / "JVS"))
    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--out-csv", default="results/calibration/cross_speaker_prosody_reference_sanity.csv")
    parser.add_argument("--out-report", default="reports/cross_speaker_prosody_reference_sanity.md")
    args = parser.parse_args()

    jvs_root = Path(args.jvs_root)
    rows = audit_pairs(
        jvs_root,
        max_speakers=args.max_speakers,
        max_pairs=args.max_pairs,
        sample_rate=args.sample_rate,
    )
    _write_csv(ROOT / args.out_csv, rows)
    _write_report(ROOT / args.out_report, rows, jvs_root=jvs_root, max_pairs=args.max_pairs)
    print(f"wrote {ROOT / args.out_csv}")
    print(f"wrote {ROOT / args.out_report}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
