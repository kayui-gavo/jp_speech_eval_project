from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.evaluator import evaluate_utterance
from jp_speech_eval.sentence_cache import build_sentence_cache


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _pick_rows(
    rows: Iterable[Dict[str, str]],
    *,
    stimulus_type: str,
    native_language: str | None,
    max_rows: int,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        if stimulus_type != "all" and row.get("Stimulus Type") != stimulus_type:
            continue
        if native_language and row.get("Native Language") != native_language:
            continue
        out.append(row)
        if max_rows > 0 and len(out) >= max_rows:
            break
    return out


def _safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _summary_row(row: Dict[str, str], wav_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    details = result.get("details") or {}
    reliability = details.get("reliability") or {}
    prosody = details.get("prosody") or {}
    fluency = details.get("fluency") or {}
    pronunciation = details.get("pronunciation") or {}
    evidence = details.get("mora_evidence_summary") or {}
    alignment = details.get("alignment") or {}
    content = details.get("content_match") or {}
    score_adjustments = details.get("score_adjustments") or []

    return {
        "speaker": row.get("Speaker"),
        "speaker_id": row.get("Speaker ID"),
        "native_language": row.get("Native Language"),
        "nationality": row.get("Nationality"),
        "sex": row.get("Sex"),
        "stimulus_type": row.get("Stimulus Type"),
        "stimulus": row.get("Stmiulus"),
        "audio_path": str(wav_path),
        "target_kana": result.get("kana"),
        "mora_count": len(result.get("moras") or []),
        "duration_sec": result.get("duration_sec"),
        "alignment_mode": result.get("alignment_mode"),
        "content_status": content.get("status"),
        "total_score": result.get("total_score"),
        "pronunciation_score": result.get("pronunciation_score"),
        "prosody_score": result.get("prosody_score"),
        "fluency_score": result.get("fluency_score"),
        "tone_score": result.get("tone_score"),
        "reliability_overall": reliability.get("overall"),
        "reliability_level": reliability.get("level"),
        "reliability_alignment": reliability.get("alignment"),
        "reliability_f0_coverage": reliability.get("f0_coverage"),
        "reliability_mora_evidence": reliability.get("mora_evidence"),
        "score_is_diagnostic": reliability.get("score_is_diagnostic"),
        "judgement_available_count": evidence.get("judgement_available_count"),
        "prosody_available_count": evidence.get("prosody_available_count"),
        "mean_boundary_confidence": evidence.get("mean_boundary_confidence"),
        "speech_rate_mora_per_sec": fluency.get("speech_rate_mora_per_sec"),
        "avg_mora_duration_sec": fluency.get("avg_mora_duration_sec"),
        "mora_duration_cv": pronunciation.get("mora_duration_cv"),
        "special_mora_penalty": pronunciation.get("special_mora_penalty"),
        "contour_corr": prosody.get("contour_corr"),
        "transition_agreement": prosody.get("transition_agreement"),
        "accent_drop_agreement": prosody.get("accent_drop_agreement"),
        "accent_phrase_count": prosody.get("accent_phrase_count"),
        "pitch_target_source": prosody.get("pitch_target_source"),
        "pitch_target_consistency": prosody.get("pitch_target_consistency"),
        "boundary_cv": alignment.get("boundary_duration_cv"),
        "score_adjustments": " | ".join(str(x) for x in score_adjustments),
        "warnings": " | ".join(str(x) for x in reliability.get("warnings", []) or []),
        "feedback": " | ".join(str(x) for x in result.get("feedback", []) or []),
    }


def _resolve_audio_path(janon_root: Path, relative_path: str) -> Path:
    wav_path = janon_root / relative_path
    if wav_path.exists():
        return wav_path
    alternatives = []
    if "/sentence/" in relative_path:
        alternatives.append(relative_path.replace("/sentence/", "/sentences/"))
    if "/isolated/" in relative_path:
        alternatives.append(relative_path.replace("/isolated/", "/isolated_words/"))
    for alt in alternatives:
        alt_path = janon_root / alt
        if alt_path.exists():
            return alt_path
    return wav_path


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _safe(v) for k, v in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run jp_speech_eval over JANON-SPEECH for calibration analysis.")
    parser.add_argument("--janon-root", default="../JANON")
    parser.add_argument("--stimulus-type", default="isolated", choices=["isolated", "sentence", "all"])
    parser.add_argument("--native-language", default=None)
    parser.add_argument("--max-rows", type=int, default=30, help="Use 0 for all matching rows.")
    parser.add_argument("--out-csv", default="outputs/janon_analysis.csv")
    parser.add_argument("--cache-dir", default="outputs/janon_cache")
    parser.add_argument("--jsonl", default=None, help="Optional full raw result JSONL.")
    parser.add_argument("--content-match", action="store_true", help="Run content-match gate. Off by default for JANON calibration because targets are known.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    janon_root = Path(args.janon_root)
    if not janon_root.is_absolute():
        janon_root = (ROOT / janon_root).resolve()
    rows = _pick_rows(
        _read_rows(janon_root / "data.csv"),
        stimulus_type=args.stimulus_type,
        native_language=args.native_language,
        max_rows=args.max_rows,
    )
    if not rows:
        parser.error("No JANON rows matched the filters.")

    out_rows: List[Dict[str, Any]] = []
    jsonl_path = Path(args.jsonl) if args.jsonl else None
    if jsonl_path and not jsonl_path.is_absolute():
        jsonl_path = ROOT / jsonl_path
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(rows, start=1):
        text = row.get("Stmiulus") or ""
        wav_path = _resolve_audio_path(janon_root, row.get("Path") or "")
        try:
            text_digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
            cache_prefix = ROOT / args.cache_dir / f"txt_{text_digest}"
            if not cache_prefix.with_suffix(".json").exists() or not cache_prefix.with_suffix(".npz").exists():
                build_sentence_cache(text, cache_prefix, save_reference_wav=False)
            result = evaluate_utterance(
                wav_path=wav_path,
                cache_path=cache_prefix,
                alignment_mode="cached_dtw",
                scoring_config_path=args.config,
                profile=False,
                use_content_match=bool(args.content_match),
            ).to_dict()
            out_rows.append(_summary_row(row, wav_path, result))
            if jsonl_path:
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"metadata": row, "result": result}, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(rows)}] ok {row.get('Speaker')} {row.get('Stimulus Type')} {text}")
        except Exception as exc:
            out_rows.append({
                "speaker": row.get("Speaker"),
                "stimulus_type": row.get("Stimulus Type"),
                "stimulus": text,
                "audio_path": str(wav_path),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{idx}/{len(rows)}] error {text}: {type(exc).__name__}: {exc}")

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = ROOT / out_csv
    _write_csv(out_csv, out_rows)
    print(f"\nSaved CSV: {out_csv}")
    if jsonl_path:
        print(f"Saved JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
