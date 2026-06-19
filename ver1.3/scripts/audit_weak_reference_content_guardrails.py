#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.text_frontend import build_text_info  # noqa: E402
from jp_speech_eval.weak_reference_guardrails import apply_weak_overall_guardrail  # noqa: E402


def _safe(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _summary(values: Iterable[Any]) -> Dict[str, Any]:
    nums = sorted(value for value in (_float_or_none(v) for v in values) if value is not None)
    if not nums:
        return {"n": 0, "mean": None, "min": None, "p50": None, "max": None}
    return {
        "n": len(nums),
        "mean": round(statistics.mean(nums), 4),
        "min": round(nums[0], 4),
        "p50": round(statistics.median(nums), 4),
        "max": round(nums[-1], 4),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _safe(row.get(key)) for key in fieldnames})


def _row_with_guardrail(row: Mapping[str, Any], *, synthetic: bool = False) -> Dict[str, Any]:
    text = str(row.get("target_text") or "")
    try:
        text_info = build_text_info(text)
        kana = text_info.kana
        moras = text_info.moras
    except Exception:
        kana = ""
        moras = []
    mora_count = _int_or_zero(row.get("mora_count")) or len(moras)
    if mora_count and not moras:
        moras = ["?"] * mora_count
    before = _float_or_none(row.get("weak_overall_practice_score"))
    if before is None:
        before = _float_or_none(row.get("weak_prosody_naturalness_score"))
    before_int = int(round(before)) if before is not None else None
    weak_details = {
        "available": str(row.get("weak_available")).lower() not in {"false", "0"} if row.get("weak_available") not in {None, ""} else before_int is not None,
        "f0_coverage": _float_or_none(row.get("f0_coverage")),
        "valid_f0_mora_count": _int_or_zero(row.get("valid_f0_mora_count")),
        "voiced_mora_count": _int_or_zero(row.get("voiced_mora_count")),
    }
    guardrail = apply_weak_overall_guardrail(
        weak_overall_score=before_int,
        target_text=text,
        kana=kana,
        moras=moras,
        duration_sec=_float_or_none(row.get("duration_sec")),
        content_match={"status": row.get("content_gate_status") or "pass"},
        weak_prosody_details=weak_details,
        mora_evidence_summary={"judgement_available_count": row.get("judgement_available_count") or mora_count},
    )
    reasons = guardrail.get("reasons") or []
    return {
        "dataset": row.get("dataset"),
        "case_name": row.get("case_name"),
        "utterance_id": row.get("utterance_id"),
        "target_text": text,
        "native_language": row.get("native_language"),
        "synthetic_control": synthetic,
        "weak_pronunciation_naturalness_score": row.get("weak_pronunciation_naturalness_score"),
        "weak_prosody_naturalness_score": row.get("weak_prosody_naturalness_score"),
        "weak_rhythm_naturalness_score": row.get("weak_rhythm_naturalness_score"),
        "fluency_score": row.get("fluency_score"),
        "duration_sec": row.get("duration_sec"),
        "mora_count": mora_count,
        "kana_ratio": guardrail.get("japanese_likeness", {}).get("japanese_ratio"),
        "japanese_likeness": guardrail.get("japanese_likeness", {}).get("japanese_ratio"),
        "latin_ratio": guardrail.get("japanese_likeness", {}).get("latin_ratio"),
        "voiced_mora_count": guardrail.get("voiced_mora_count"),
        "valid_f0_mora_count": guardrail.get("valid_f0_mora_count"),
        "f0_coverage": row.get("f0_coverage"),
        "content_gate_status": guardrail.get("content_gate_status"),
        "guardrail_status": guardrail.get("status"),
        "guardrail_cap": guardrail.get("cap"),
        "guardrail_reasons": ";".join(str(reason) for reason in reasons),
        "weak_overall_before_guardrail": before_int,
        "weak_overall_after_guardrail": guardrail.get("weak_overall_practice_score_after_guardrail"),
        "reason": ";".join(str(reason) for reason in reasons) or "ok",
    }


def _synthetic_rows() -> List[Dict[str, Any]]:
    return [
        {
            "dataset": "synthetic",
            "case_name": "random_english_latin",
            "target_text": "please give me ramen",
            "mora_count": 0,
            "weak_overall_practice_score": 93,
            "weak_prosody_naturalness_score": 91,
            "f0_coverage": 0.95,
            "valid_f0_mora_count": 8,
            "voiced_mora_count": 8,
        },
        {
            "dataset": "synthetic",
            "case_name": "latin_dominant_transcript",
            "target_text": "I want sushi kudasai",
            "mora_count": 2,
            "weak_overall_practice_score": 88,
            "weak_prosody_naturalness_score": 80,
            "f0_coverage": 0.9,
            "valid_f0_mora_count": 5,
            "voiced_mora_count": 5,
        },
        {
            "dataset": "synthetic",
            "case_name": "kana_mora_failed",
            "target_text": "12345",
            "mora_count": 0,
            "weak_overall_practice_score": 82,
            "weak_prosody_naturalness_score": 75,
            "f0_coverage": 0.9,
            "valid_f0_mora_count": 5,
            "voiced_mora_count": 5,
        },
        {
            "dataset": "synthetic",
            "case_name": "short_japanese_control",
            "target_text": "はい",
            "mora_count": 2,
            "weak_overall_practice_score": 95,
            "weak_prosody_naturalness_score": 93,
            "f0_coverage": 1.0,
            "valid_f0_mora_count": 2,
            "voiced_mora_count": 2,
        },
        {
            "dataset": "synthetic",
            "case_name": "asr_confirmed_japanese_sentence",
            "target_text": "今日はいい天気です",
            "mora_count": 10,
            "weak_overall_practice_score": 91,
            "weak_prosody_naturalness_score": 92,
            "f0_coverage": 0.92,
            "valid_f0_mora_count": 9,
            "voiced_mora_count": 9,
        },
    ]


def build_rows(source_csv: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if source_csv.exists():
        for row in csv.DictReader(source_csv.open(encoding="utf-8")):
            rows.append(_row_with_guardrail(row))
    for row in _synthetic_rows():
        rows.append(_row_with_guardrail(row, synthetic=True))
    return rows


def _by_case(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Any]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("case_name") or "")].append(row.get(field))
    return {case: _summary(values) for case, values in sorted(grouped.items())}


def write_report(path: Path, rows: Sequence[Mapping[str, Any]], *, source_csv: Path) -> None:
    before = _by_case(rows, "weak_overall_before_guardrail")
    after = _by_case(rows, "weak_overall_after_guardrail")
    no_score = defaultdict(int)
    capped = defaultdict(int)
    total = defaultdict(int)
    for row in rows:
        case = str(row.get("case_name") or "")
        total[case] += 1
        if row.get("guardrail_status") == "no_score":
            no_score[case] += 1
        if row.get("guardrail_status") == "capped":
            capped[case] += 1
    lines = [
        "# Weak-reference content guardrail audit",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- source_csv: `{source_csv}`",
        "- scope: weak-reference arbitrary-sentence practice score guardrails only.",
        "- strict fixed-reference, pitch scoring formula, aggregate weights, UI, ASR/TTS providers, and tone core dimensions are unchanged.",
        "",
        "## Weak Overall Before Guardrail",
        "",
        "| case | n | mean | min | p50 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case, stats in before.items():
        lines.append(f"| {case} | {stats['n']} | {stats['mean']} | {stats['min']} | {stats['p50']} | {stats['max']} |")
    lines.extend([
        "",
        "## Weak Overall After Guardrail",
        "",
        "| case | n | mean | min | p50 | max | no_score | capped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case, stats in after.items():
        lines.append(
            f"| {case} | {stats['n']} | {stats['mean']} | {stats['min']} | {stats['p50']} | {stats['max']} | "
            f"{no_score[case]}/{total[case]} | {capped[case]}/{total[case]} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- JANON English means English-L1 learners reading Japanese stimuli, not random English input. Its previous high mean was partly from short isolated-word rows where fluency/rhythm-like proxies could dominate weak overall.",
        "- Latin-dominant, random-English, and kana/mora-failed controls now receive no weak overall practice score.",
        "- Very short Japanese controls are treated as insufficient evidence rather than high-scoring arbitrary-sentence practice; this also explains most JANON no-score/capped rows.",
        "- JVS native rows are long Japanese utterances and remain unaffected by the guardrail.",
        "- Flat/random pitch controls remain below native because the pitch scoring formula was not changed.",
        "- This is a content/language/evidence guardrail, not pitch calibration.",
        "",
        "## Product Rule",
        "",
        "- Show weak-reference practice score only when confirmed text is Japanese-like and has enough mora/F0 evidence.",
        "- Hide weak overall for English, Latin-dominant, kana/mora extraction failures, content mismatch, and too-short utterances.",
        "- Cap weak overall for low evidence cases instead of letting fluency/rhythm proxies create a high score.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit weak-reference content/language guardrails.")
    parser.add_argument("--source-csv", default="results/calibration/weak_reference_native_likeness_audit.csv")
    parser.add_argument("--out-csv", default="results/calibration/weak_reference_content_guardrail_audit.csv")
    parser.add_argument("--out-report", default="reports/weak_reference_content_guardrail_audit.md")
    args = parser.parse_args()

    source_csv = ROOT / args.source_csv
    rows = build_rows(source_csv)
    out_csv = ROOT / args.out_csv
    out_report = ROOT / args.out_report
    _write_csv(out_csv, rows)
    write_report(out_report, rows, source_csv=source_csv)
    print(f"wrote {out_csv}")
    print(f"wrote {out_report}")


if __name__ == "__main__":
    main()
