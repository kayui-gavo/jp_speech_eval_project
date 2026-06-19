#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.content_match import _kana_similarity
from jp_speech_eval.text_frontend import build_text_info, kata_normalize
from jp_speech_eval.transcript_sanity import check_asr_transcript_sanity
from jp_speech_eval.weak_reference_guardrails import japanese_likeness


PROVIDERS = {
    "current_asr": {"column": "current_asr", "language_column": "current_language", "timestamps": False, "confirmed": False},
    "oracle_transcript": {"column": "oracle_transcript", "language": "oracle", "timestamps": False, "confirmed": False},
    "whisper_large_or_faster_whisper": {"column": "whisper_large_candidate", "language": "auto", "timestamps": False, "confirmed": False},
    "whisperx_alignment": {"column": "whisperx_candidate", "language": "auto", "timestamps": True, "confirmed": False},
    "openai_transcribe_candidate": {"column": "openai_transcribe_candidate", "language": "auto", "timestamps": True, "confirmed": False},
    "manual_transcript_confirmed": {"column": "manual_transcript_confirmed", "language": "manual", "timestamps": False, "confirmed": True},
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _normalized(text: str) -> str:
    return kata_normalize(str(text or "").replace(" ", ""))


def _kana_and_moras(text: str) -> tuple[str, list[str]]:
    if not text:
        return "", []
    try:
        info = build_text_info(text)
        return info.kana, info.moras
    except Exception:
        return "", []


def _language_for(row: Mapping[str, str], spec: Mapping[str, Any], transcript: str) -> str:
    column = spec.get("language_column")
    if column:
        return str(row.get(str(column)) or "unknown")
    fixed = str(spec.get("language") or "auto")
    if fixed in {"oracle", "manual", "auto"}:
        likeness = japanese_likeness(text=transcript)
        return "ja" if likeness["japanese_ratio"] >= 0.55 else "en" if transcript else "unknown"
    return fixed


def _evaluate(row: Mapping[str, str], provider: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    transcript = str(row.get(str(spec["column"])) or "")
    language = _language_for(row, spec, transcript)
    expected = str(row.get("expected_spoken_text") or "")
    target = str(row.get("target_text") or "")
    expected_good = str(row.get("expected_good_japanese") or "false").lower() == "true"
    sanity = check_asr_transcript_sanity(transcript) if transcript else None
    transcript_kana, moras = _kana_and_moras(transcript)
    target_kana, _target_moras = _kana_and_moras(target)
    likeness = japanese_likeness(text=transcript, kana=transcript_kana, moras=moras)
    kana_similarity = _kana_similarity(target_kana, transcript_kana) if target_kana and transcript_kana else 0.0
    exact = bool(transcript) and _normalized(transcript) == _normalized(expected)

    if not transcript:
        content_gate = "unavailable"
        reason = "missing_transcript"
    elif language not in {"ja", "jp", "jpn", "japanese", "manual", "oracle"}:
        content_gate = "reject"
        reason = "non_japanese_language"
    elif sanity is not None and not sanity.ok:
        content_gate = "reject"
        reason = sanity.reason
    elif likeness["latin_dominant"]:
        content_gate = "reject"
        reason = "latin_dominant"
    elif target_kana and transcript_kana and kana_similarity < 0.35:
        content_gate = "reject"
        reason = "low_kana_similarity"
    else:
        content_gate = "pass"
        reason = "candidate_japanese_content"

    enough_sentence_evidence = len(moras) >= 4
    visible_if_confirmed = content_gate == "pass" and enough_sentence_evidence
    currently_visible = bool(spec.get("confirmed")) and visible_if_confirmed
    hallucination_risk = not expected_good and content_gate == "pass" and bool(transcript)
    false_reject = expected_good and row.get("input_class") != "low_evidence_japanese" and content_gate != "pass"
    false_accept = not expected_good and visible_if_confirmed
    no_score_correct = (not expected_good or row.get("input_class") == "low_evidence_japanese") and not currently_visible
    timestamp_available = bool(spec.get("timestamps")) and bool(transcript)
    return {
        "case_id": row.get("case_id"),
        "input_class": row.get("input_class"),
        "provider_slot": provider,
        "result_source": "offline_plan_replay",
        "external_provider_called": False,
        "target_text": target,
        "expected_spoken_text": expected,
        "transcript": transcript,
        "language": language,
        "asr_confidence": None,
        "transcript_exact_normalized": exact,
        "transcript_kana": transcript_kana,
        "kana_similarity": round(float(kana_similarity), 4),
        "japanese_likeness": likeness["japanese_ratio"],
        "mora_count": len(moras),
        "content_gate_status": content_gate,
        "content_gate_reason": reason,
        "weak_score_visibility_now": currently_visible,
        "weak_score_visibility_if_user_confirms_unchanged": visible_if_confirmed,
        "no_score_correct": no_score_correct,
        "false_accept_bad_input_if_confirmed": false_accept,
        "false_reject_good_japanese": false_reject,
        "hallucination_risk": hallucination_risk,
        "word_timestamps_available": timestamp_available,
        "alignment_usefulness": "word_or_segment_anchor_available" if timestamp_available else "none",
        "feedback_location_may_improve": timestamp_available and expected_good,
        "notes": row.get("notes"),
    }


def audit_rows(plan_path: Path) -> list[dict[str, Any]]:
    return [
        _evaluate(row, provider, spec)
        for row in _read(plan_path)
        for provider, spec in PROVIDERS.items()
    ]


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(statistics.mean(values), 4) if values else None


def _provider_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["provider_slot"])].append(row)
    out: list[dict[str, Any]] = []
    for provider, items in grouped.items():
        good = [row for row in items if row["input_class"] == "good_japanese"]
        bad = [row for row in items if str(row["input_class"]).startswith("bad_")]
        kana = [float(row["kana_similarity"]) for row in good if row["transcript"]]
        out.append({
            "provider": provider,
            "good_japanese_accept_rate": _mean(float(row["content_gate_status"] == "pass") for row in good),
            "bad_input_reject_rate": _mean(float(row["content_gate_status"] != "pass") for row in bad),
            "hallucination_risk_rate": _mean(float(row["hallucination_risk"]) for row in bad),
            "good_japanese_mean_kana_similarity": _mean(kana),
            "false_reject_count": sum(bool(row["false_reject_good_japanese"]) for row in items),
            "false_accept_if_confirmed_count": sum(bool(row["false_accept_bad_input_if_confirmed"]) for row in items),
            "timestamp_case_count": sum(bool(row["word_timestamps_available"]) for row in items),
        })
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, Any]], plan_path: Path) -> None:
    summary = _provider_summary(rows)
    lines = [
        "# ASR bottleneck audit",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- plan: `{plan_path}`",
        f"- replay rows: {len(rows)}",
        "- external provider calls: 0",
        "- status: offline planning replay, not an empirical provider benchmark.",
        "",
        "## Provider-slot summary",
        "",
        "| provider | good accept | bad reject | hallucination risk | mean kana similarity | false reject | false accept if blindly confirmed | timestamp cases |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['provider']} | {item['good_japanese_accept_rate']} | {item['bad_input_reject_rate']} | "
            f"{item['hallucination_risk_rate']} | {item['good_japanese_mean_kana_similarity']} | "
            f"{item['false_reject_count']} | {item['false_accept_if_confirmed_count']} | {item['timestamp_case_count']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Better ASR can improve candidate transcript quality, kana/mora stability, content-gate decisions, and the amount of manual correction needed.",
        "- Word/segment timestamps may improve coarse feedback location and alignment initialization, especially for pauses and special-mora sentences.",
        "- User confirmation remains the product safety boundary. A Japanese-looking hallucination must not become a formal score merely because an ASR provider emitted it.",
        "- Oracle/manual transcript replay improves noisy or pronunciation-variant content eligibility, but it does not improve F0 extraction or pitch naturalness features.",
        "- This replay cannot rank real providers until identical audio is transcribed offline by each candidate and stored with language/confidence/timestamps.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an offline ASR bottleneck benchmark plan.")
    parser.add_argument("--plan", default="data/asr_benchmark_plan.csv")
    parser.add_argument("--out-csv", default="results/calibration/asr_bottleneck_audit.csv")
    parser.add_argument("--out-report", default="reports/asr_bottleneck_audit.md")
    args = parser.parse_args()
    plan = ROOT / args.plan
    rows = audit_rows(plan)
    _write_csv(ROOT / args.out_csv, rows)
    _write_report(ROOT / args.out_report, rows, plan)
    print(f"wrote {ROOT / args.out_csv}")
    print(f"wrote {ROOT / args.out_report}")


if __name__ == "__main__":
    main()
