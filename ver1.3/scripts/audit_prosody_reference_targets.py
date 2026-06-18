#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.alignment_evidence.phone_mora_mapper import map_phones_to_moras
from jp_speech_eval.alignment_evidence.textgrid_parser import parse_lab_phone_segments
from jp_speech_eval.audio_features import extract_f0, load_audio, log_f0_normalize, median_f0_by_mora
from jp_speech_eval.evaluator import evaluate_utterance
from jp_speech_eval.scoring import score_prosody
from jp_speech_eval.text_frontend import build_text_info


def _jvs_items(
    jvs_root: Path,
    *,
    speakers: int,
    utterances_per_speaker: int,
    limit: int | None,
) -> Iterable[Dict[str, str]]:
    emitted = 0
    for speaker_dir in sorted(jvs_root.glob("jvs*"))[:speakers]:
        transcript_path = speaker_dir / "parallel100" / "transcripts_utf8.txt"
        wav_dir = speaker_dir / "parallel100" / "wav24kHz16bit"
        lab_dir = speaker_dir / "parallel100" / "lab" / "mon"
        if not transcript_path.exists() or not wav_dir.exists():
            continue
        count = 0
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            utt_id, text = line.split(":", 1)
            wav = wav_dir / f"{utt_id}.wav"
            lab = lab_dir / f"{utt_id}.lab"
            if not wav.exists():
                continue
            yield {
                "speaker_id": speaker_dir.name,
                "utterance_id": utt_id,
                "audio_path": str(wav),
                "lab_path": str(lab) if lab.exists() else "",
                "target_text": text.strip(),
            }
            emitted += 1
            count += 1
            if limit is not None and emitted >= limit:
                return
            if count >= utterances_per_speaker:
                break


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _pct(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 4)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return round(float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac), 4)


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(float(statistics.fmean(values)), 4),
        "min": round(float(min(values)), 4),
        "p10": _pct(values, 0.10),
        "p50": _pct(values, 0.50),
        "p90": _pct(values, 0.90),
        "max": round(float(max(values)), 4),
    }


def pattern_from_f0(f0_by_mora: Sequence[float]) -> List[str]:
    z = log_f0_normalize(np.asarray(f0_by_mora, dtype=float))
    return ["H" if np.isfinite(v) and v >= 0 else "L" if np.isfinite(v) else "?" for v in z]


def smoothed_f0(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr) & (arr > 0)
    if int(np.sum(valid)) < 2:
        return arr.tolist()
    x = np.arange(arr.size)
    filled = arr.copy()
    filled[~valid] = np.interp(x[~valid], x[valid], arr[valid])
    if arr.size < 3:
        return filled.tolist()
    padded = np.pad(filled, (1, 1), mode="edge")
    smooth = 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
    smooth[~valid] = np.nan
    return smooth.tolist()


def shifted_f0(values: Sequence[float], steps: int) -> List[float]:
    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    if steps > 0:
        out[steps:] = arr[:-steps]
    elif steps < 0:
        out[:steps] = arr[-steps:]
    else:
        out = arr.copy()
    return out.tolist()


def flat_f0(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr) & (arr > 0)
    out = np.full_like(arr, np.nan, dtype=float)
    if int(np.sum(valid)) > 0:
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


def _coverage(f0_by_mora: Sequence[float]) -> float:
    if not f0_by_mora:
        return 0.0
    arr = np.asarray(f0_by_mora, dtype=float)
    return float(np.mean(np.isfinite(arr) & (arr > 0)))


def _score_row(
    *,
    base: Mapping[str, Any],
    target_mode: str,
    alignment_mode: str,
    user_f0: Sequence[float],
    target_pattern: Sequence[str],
    pitch_target_source: str,
    reference_f0: Sequence[float] | None,
    accent_phrases: Sequence[Mapping[str, Any]] | None,
    notes: str,
) -> Dict[str, Any]:
    score, _feedback, details = score_prosody(
        moras=list(base["moras"]),
        target_pattern=list(target_pattern),
        f0_by_mora=list(user_f0),
        reference_f0_by_mora=list(reference_f0) if reference_f0 is not None else None,
        pitch_target_source=pitch_target_source,
        is_question=bool(base.get("is_question")),
        accent_phrases=list(accent_phrases or []),
    )
    components = details.get("prosody_score_components") or {}
    weights = components.get("weights") or {}
    valid_count = int(details.get("valid_mora_count") or 0)
    mora_count = int(details.get("mora_count") or len(base["moras"]))
    return {
        "sample_id": base["sample_id"],
        "speaker_id": base["speaker_id"],
        "utterance_id": base["utterance_id"],
        "target_text": base["target_text"],
        "target_kana": base["target_kana"],
        "target_mora_count": mora_count,
        "audio_path": base["audio_path"],
        "lab_path": base.get("lab_path") or "",
        "target_mode": target_mode,
        "alignment_mode": alignment_mode,
        "prosody_score": score,
        "f0_coverage": round(_coverage(user_f0), 4),
        "contour_corr": details.get("contour_corr"),
        "contour_rmse": details.get("contour_rmse"),
        "transition_agreement": details.get("transition_agreement"),
        "hl_match": details.get("hl_match_rate", details.get("hl_match")),
        "accent_drop_match": details.get("accent_drop_agreement"),
        "final_intonation_score": details.get("final_score"),
        "voiced_mora_count": valid_count,
        "valid_f0_mora_count": valid_count,
        "pitch_target_source": details.get("pitch_target_source"),
        "hl_target_source": details.get("hl_target_source"),
        "pitch_target_consistency": details.get("pitch_target_consistency"),
        "gate_reason": "",
        "unavailable_reason": details.get("note") or "",
        "component_contour": components.get("contour"),
        "component_transition": components.get("transition"),
        "component_final": components.get("final"),
        "component_hl": components.get("hl"),
        "weight_contour": weights.get("contour"),
        "weight_transition": weights.get("transition"),
        "weight_final": weights.get("final"),
        "weight_hl": weights.get("hl"),
        "notes": notes,
    }


def _current_openjtalk_equal_row(item: Mapping[str, str], *, sample_rate: int, config: str | None) -> Dict[str, Any]:
    result = evaluate_utterance(
        item["target_text"],
        item["audio_path"],
        alignment_mode="equal",
        sample_rate=sample_rate,
        scoring_config_path=config,
        use_content_match=False,
    )
    data = result.to_dict()
    details = data.get("details") or {}
    prosody = details.get("prosody") or {}
    reliability = details.get("reliability") or {}
    components = prosody.get("prosody_score_components") or {}
    weights = components.get("weights") or {}
    valid_count = int(prosody.get("valid_mora_count") or 0)
    mora_count = int(prosody.get("mora_count") or len(data.get("moras") or []))
    return {
        "sample_id": f"{item['speaker_id']}:{item['utterance_id']}",
        "speaker_id": item["speaker_id"],
        "utterance_id": item["utterance_id"],
        "target_text": item["target_text"],
        "target_kana": data.get("kana"),
        "target_mora_count": mora_count,
        "audio_path": item["audio_path"],
        "lab_path": item.get("lab_path") or "",
        "target_mode": "current_openjtalk_target",
        "alignment_mode": data.get("alignment_mode") or "equal",
        "prosody_score": data.get("prosody_score"),
        "f0_coverage": reliability.get("f0_coverage"),
        "contour_corr": prosody.get("contour_corr"),
        "contour_rmse": prosody.get("contour_rmse"),
        "transition_agreement": prosody.get("transition_agreement"),
        "hl_match": prosody.get("hl_match_rate", prosody.get("hl_match")),
        "accent_drop_match": prosody.get("accent_drop_agreement"),
        "final_intonation_score": prosody.get("final_score"),
        "voiced_mora_count": valid_count,
        "valid_f0_mora_count": valid_count,
        "pitch_target_source": prosody.get("pitch_target_source"),
        "hl_target_source": prosody.get("hl_target_source"),
        "pitch_target_consistency": prosody.get("pitch_target_consistency"),
        "gate_reason": "",
        "unavailable_reason": prosody.get("note") or "",
        "component_contour": components.get("contour"),
        "component_transition": components.get("transition"),
        "component_final": components.get("final"),
        "component_hl": components.get("hl"),
        "weight_contour": weights.get("contour"),
        "weight_transition": weights.get("transition"),
        "weight_final": weights.get("final"),
        "weight_hl": weights.get("hl"),
        "notes": "existing runtime path with trimmed audio and equal mora boundaries",
    }


def _lab_f0_by_mora(item: Mapping[str, str], moras: Sequence[str], *, sample_rate: int) -> tuple[List[float] | None, Dict[str, Any]]:
    lab_path = Path(item.get("lab_path") or "")
    if not lab_path.exists():
        return None, {"lab_available": False, "notes": "missing_lab_file"}
    phones = parse_lab_phone_segments(lab_path)
    segments, mapping = map_phones_to_moras(phones, moras)
    if len(segments) != len(moras):
        return None, {
            "lab_available": True,
            "mapping_success": False,
            "notes": "lab_phone_to_mora_mapping_failed",
            "mapping_warning_flags": ";".join(str(x) for x in mapping.get("mapping_warning_flags") or []),
        }
    audio = load_audio(str(item["audio_path"]), sr=sample_rate)
    times, f0, method = extract_f0(audio.y, audio.sr)
    boundaries = [(float(seg.start), float(seg.end)) for seg in segments]
    f0_by_mora = median_f0_by_mora(times, f0, boundaries)
    return f0_by_mora, {
        "lab_available": True,
        "mapping_success": bool(mapping.get("mapping_success")),
        "special_mora_mapping_success": bool(mapping.get("special_mora_mapping_success")),
        "mapping_warning_flags": ";".join(str(x) for x in mapping.get("mapping_warning_flags") or []),
        "f0_method": method,
        "lab_boundary_min_sec": round(min(e - s for s, e in boundaries), 4) if boundaries else None,
        "lab_boundary_max_sec": round(max(e - s for s, e in boundaries), 4) if boundaries else None,
        "notes": "JVS monophone lab timing mapped sequentially to target moras",
    }


def diagnose_item(item: Mapping[str, str], *, sample_rate: int, config: str | None) -> List[Dict[str, Any]]:
    text_info = build_text_info(item["target_text"])
    base = {
        "sample_id": f"{item['speaker_id']}:{item['utterance_id']}",
        "speaker_id": item["speaker_id"],
        "utterance_id": item["utterance_id"],
        "target_text": item["target_text"],
        "target_kana": text_info.kana,
        "moras": text_info.moras,
        "is_question": text_info.is_question,
        "audio_path": item["audio_path"],
        "lab_path": item.get("lab_path") or "",
    }
    rows = [_current_openjtalk_equal_row(item, sample_rate=sample_rate, config=config)]
    lab_f0, lab_meta = _lab_f0_by_mora(item, text_info.moras, sample_rate=sample_rate)
    if lab_f0 is None:
        row = dict(rows[0])
        row["target_mode"] = "lab_alignment_unavailable"
        row["alignment_mode"] = "lab_or_reference_alignment"
        row["prosody_score"] = ""
        row["notes"] = lab_meta.get("notes", "lab_alignment_unavailable")
        rows.append(row)
        return rows

    lab_notes = "; ".join(
        str(part)
        for part in [
            lab_meta.get("notes"),
            f"f0_method={lab_meta.get('f0_method')}",
            f"mapping_warnings={lab_meta.get('mapping_warning_flags') or 'none'}",
        ]
        if part
    )
    rows.append(_score_row(
        base=base,
        target_mode="current_openjtalk_target_lab_alignment",
        alignment_mode="lab_or_reference_alignment",
        user_f0=lab_f0,
        target_pattern=text_info.target_pitch,
        pitch_target_source=text_info.pitch_target_source,
        reference_f0=None,
        accent_phrases=text_info.accent_phrases,
        notes=lab_notes,
    ))

    self_pattern = pattern_from_f0(lab_f0)
    smooth = smoothed_f0(lab_f0)
    rows.append(_score_row(
        base=base,
        target_mode="self_oracle_contour_target",
        alignment_mode="lab_or_reference_alignment",
        user_f0=lab_f0,
        target_pattern=self_pattern,
        pitch_target_source="manual",
        reference_f0=lab_f0,
        accent_phrases=text_info.accent_phrases,
        notes="same native mora-level F0 used as both user contour and reference target",
    ))
    rows.append(_score_row(
        base=base,
        target_mode="smoothed_self_target",
        alignment_mode="lab_or_reference_alignment",
        user_f0=lab_f0,
        target_pattern=pattern_from_f0(smooth),
        pitch_target_source="manual",
        reference_f0=smooth,
        accent_phrases=text_info.accent_phrases,
        notes="same native contour smoothed with a 3-point mora-level kernel as reference target",
    ))
    for steps in (-1, 1):
        shifted = shifted_f0(lab_f0, steps)
        rows.append(_score_row(
            base=base,
            target_mode=f"shifted_self_target_{steps:+d}_mora",
            alignment_mode="lab_or_reference_alignment",
            user_f0=lab_f0,
            target_pattern=pattern_from_f0(shifted),
            pitch_target_source="manual",
            reference_f0=shifted,
            accent_phrases=text_info.accent_phrases,
            notes="self reference shifted by one mora to quantify alignment sensitivity",
        ))
    rows.append(_score_row(
        base=base,
        target_mode="flat_user_contour_vs_self_target",
        alignment_mode="lab_or_reference_alignment",
        user_f0=flat_f0(lab_f0),
        target_pattern=self_pattern,
        pitch_target_source="manual",
        reference_f0=lab_f0,
        accent_phrases=text_info.accent_phrases,
        notes="user contour flattened while target/reference remains native self contour",
    ))
    rows.append(_score_row(
        base=base,
        target_mode="shuffled_user_contour_vs_self_target",
        alignment_mode="lab_or_reference_alignment",
        user_f0=shuffled_f0(lab_f0, seed=sum(ord(ch) for ch in str(base["sample_id"]))),
        target_pattern=self_pattern,
        pitch_target_source="manual",
        reference_f0=lab_f0,
        accent_phrases=text_info.accent_phrases,
        notes="valid user F0 values shuffled while target/reference remains native self contour",
    ))
    return rows


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


def _by_mode_summary(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = _num(row.get(field))
        if value is not None:
            grouped[str(row.get("target_mode") or "unknown")].append(value)
    for mode, values in grouped.items():
        out[mode] = _summary(values)
    return out


def _mean_by_mode(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, float]:
    out = {}
    for mode, summary in _by_mode_summary(rows, field).items():
        if summary.get("n"):
            out[mode] = float(summary["mean"])
    return out


def _paired_delta(rows: Sequence[Mapping[str, Any]], *, reference_mode: str, other_mode: str, field: str = "prosody_score") -> Dict[str, Any]:
    by_sample: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_sample[str(row.get("sample_id"))][str(row.get("target_mode"))] = row
    deltas: List[float] = []
    for modes in by_sample.values():
        ref = _num((modes.get(reference_mode) or {}).get(field))
        other = _num((modes.get(other_mode) or {}).get(field))
        if ref is not None and other is not None:
            deltas.append(ref - other)
    summary = _summary(deltas)
    summary["reference_minus_other"] = f"{reference_mode} - {other_mode}"
    return summary


def _markdown_table_from_summary(summary: Mapping[str, Mapping[str, Any]]) -> List[str]:
    lines = [
        "| target_mode | n | mean | min | p10 | p50 | p90 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, row in sorted(summary.items()):
        lines.append(
            f"| {mode} | {row.get('n', 0)} | {row.get('mean', '')} | {row.get('min', '')} | "
            f"{row.get('p10', '')} | {row.get('p50', '')} | {row.get('p90', '')} | {row.get('max', '')} |"
        )
    return lines


def _write_report(path: Path, rows: Sequence[Mapping[str, Any]], *, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    score_summary = _by_mode_summary(rows, "prosody_score")
    contour_summary = _by_mode_summary(rows, "contour_corr")
    transition_summary = _by_mode_summary(rows, "transition_agreement")
    hl_summary = _by_mode_summary(rows, "hl_match")
    f0_summary = _by_mode_summary(rows, "f0_coverage")
    score_means = _mean_by_mode(rows, "prosody_score")
    target_sources = Counter(str(row.get("pitch_target_source") or "unknown") for row in rows)
    alignment_modes = Counter(str(row.get("alignment_mode") or "unknown") for row in rows)
    low_f0_count = sum(1 for row in rows if (_num(row.get("f0_coverage")) or 0.0) < 0.50)

    paired_modes = [
        "current_openjtalk_target",
        "current_openjtalk_target_lab_alignment",
        "smoothed_self_target",
        "shifted_self_target_-1_mora",
        "shifted_self_target_+1_mora",
        "flat_user_contour_vs_self_target",
        "shuffled_user_contour_vs_self_target",
    ]
    deltas = [
        _paired_delta(rows, reference_mode="self_oracle_contour_target", other_mode=mode)
        for mode in paired_modes
    ]
    self_mean = score_means.get("self_oracle_contour_target")
    current_mean = score_means.get("current_openjtalk_target")
    lab_current_mean = score_means.get("current_openjtalk_target_lab_alignment")
    flat_mean = score_means.get("flat_user_contour_vs_self_target")
    shuffled_mean = score_means.get("shuffled_user_contour_vs_self_target")
    shifted_means = [
        score_means.get("shifted_self_target_-1_mora"),
        score_means.get("shifted_self_target_+1_mora"),
    ]
    shifted_available = [v for v in shifted_means if v is not None]

    interpretations: List[str] = []
    if self_mean is not None and self_mean >= 90:
        interpretations.append("self-oracle scores are high, so score_prosody can reward a matching mora-level F0 contour.")
    elif self_mean is not None:
        interpretations.append("self-oracle is not high enough; inspect F0 aggregation or component math before calibration.")
    if self_mean is not None and current_mean is not None and self_mean - current_mean >= 15:
        interpretations.append("OpenJTalk/tool-generated pitch targets are a major limiter versus self-reference.")
    if current_mean is not None and lab_current_mean is not None:
        diff = lab_current_mean - current_mean
        if diff >= 8:
            interpretations.append("lab/reference mora timing improves current OpenJTalk-target scoring, so equal alignment is a material risk.")
        elif abs(diff) < 5:
            interpretations.append("lab/reference mora timing does not materially improve OpenJTalk-target scoring in this audit.")
        else:
            interpretations.append("lab/reference mora timing changes scores, but the direction is mixed enough to treat alignment as a risk rather than the sole cause.")
    if shifted_available and self_mean is not None:
        shifted_gap = self_mean - statistics.fmean(shifted_available)
        if shifted_gap >= 20:
            interpretations.append("one-mora shifts heavily reduce score, so pitch scoring is alignment-sensitive.")
        elif shifted_gap >= 8:
            interpretations.append("one-mora shifts reduce score moderately; alignment precision still matters.")
        else:
            interpretations.append("one-mora shifts do not strongly reduce score in this audit; alignment sensitivity looks limited.")
    if self_mean is not None and flat_mean is not None and self_mean - flat_mean >= 10:
        interpretations.append("flat pitch counterfactuals score lower than self-oracle.")
    if self_mean is not None and shuffled_mean is not None and self_mean - shuffled_mean >= 10:
        interpretations.append("shuffled/randomized pitch counterfactuals score lower than self-oracle.")
    if low_f0_count == 0:
        interpretations.append("low F0 coverage is not the main explanation for these rows.")
    interpretations.append("This report is diagnostic only; it does not change runtime scoring, UI, aggregate, or content gates.")

    lines = [
        "# Prosody reference/target/alignment audit: JVS native",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- jvs_root: `{args.jvs_root}`",
        f"- rows: {len(rows)}",
        f"- speakers: {args.jvs_speakers}",
        f"- utterances_per_speaker: {args.jvs_utterances_per_speaker}",
        f"- sample_rate: {args.sample_rate}",
        f"- pitch_target_source counts: `{dict(target_sources)}`",
        f"- alignment_mode counts: `{dict(alignment_modes)}`",
        "",
        "## Prosody Score by Target Mode",
        "",
        *_markdown_table_from_summary(score_summary),
        "",
        "## F0 Coverage by Target Mode",
        "",
        *_markdown_table_from_summary(f0_summary),
        "",
        "## Contour Correlation by Target Mode",
        "",
        *_markdown_table_from_summary(contour_summary),
        "",
        "## Transition Agreement by Target Mode",
        "",
        *_markdown_table_from_summary(transition_summary),
        "",
        "## HL Match by Target Mode",
        "",
        *_markdown_table_from_summary(hl_summary),
        "",
        "## Paired Score Deltas",
        "",
        "| comparison | n | mean | min | p50 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for delta in deltas:
        lines.append(
            f"| {delta.get('reference_minus_other')} | {delta.get('n', 0)} | {delta.get('mean', '')} | "
            f"{delta.get('min', '')} | {delta.get('p50', '')} | {delta.get('max', '')} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        *[f"- {item}" for item in interpretations],
        "",
        "## Calibration Readiness",
        "",
    ])
    if (
        self_mean is not None
        and self_mean >= 90
        and current_mean is not None
        and self_mean - current_mean >= 15
        and flat_mean is not None
        and shuffled_mean is not None
        and self_mean > flat_mean
        and self_mean > shuffled_mean
    ):
        lines.append(
            "Not ready for a score calibration transform yet. The audit supports target/reference and alignment work first, because self-reference behaves well while the current OpenJTalk target remains much lower."
        )
    else:
        lines.append(
            "Not ready. The audit does not yet provide enough stable evidence to apply calibration safely."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit prosody target/reference/alignment behavior on JVS native audio.")
    parser.add_argument("--jvs-root", type=Path, default=PROJECT_ROOT / "JVS")
    parser.add_argument("--jvs-speakers", type=int, default=1)
    parser.add_argument("--jvs-utterances-per-speaker", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "calibration" / "prosody_reference_target_audit_jvs_native.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "prosody_reference_target_audit_jvs_native.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for item in _jvs_items(
        args.jvs_root,
        speakers=args.jvs_speakers,
        utterances_per_speaker=args.jvs_utterances_per_speaker,
        limit=args.limit,
    ):
        rows.extend(diagnose_item(item, sample_rate=args.sample_rate, config=args.config))
    _write_csv(args.csv, rows)
    _write_report(args.report, rows, args=args)
    print(f"wrote {len(rows)} rows to {args.csv}")
    print(f"wrote report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
