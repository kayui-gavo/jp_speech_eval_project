from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


FIELDNAMES = [
    "case_id",
    "audio_path",
    "label",
    "asr_provider",
    "asr_model",
    "asr_available",
    "asr_text",
    "asr_language",
    "asr_note",
    "asr_sanity_ok",
    "asr_sanity_reason",
    "tts_backend",
    "confirmed_text_source",
    "weak_reference_evaluated",
    "display_score",
    "debug_total_score",
    "debug_prosody_score",
    "pitch_correctness_available",
    "weak_reference",
    "reference_source",
    "latency_ms",
    "error",
]


def _load_cases(path: str | Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("case_id", f"case_{idx:03d}")
            row.setdefault("label", "")
            row.setdefault("confirmed_text", "")
            cases.append(row)
    return cases


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _run_asr(audio_path: str | Path, *, provider: str, model: str) -> Dict[str, Any]:
    if provider.lower().strip() in {"none", "off", "skip"}:
        return {
            "asr_available": False,
            "asr_text": "",
            "asr_language": "",
            "asr_note": "asr_skipped",
            "asr_sanity_ok": False,
            "asr_sanity_reason": "asr_skipped",
        }
    from jp_speech_eval.asr import transcribe_japanese
    from jp_speech_eval.audio_features import load_audio
    from jp_speech_eval.transcript_sanity import check_asr_transcript_sanity
    from jp_speech_eval.vad import trim_to_speech

    audio = load_audio(str(audio_path))
    y_trim, _ = trim_to_speech(audio.y, audio.sr)
    transcript = transcribe_japanese(y_trim, audio.sr, model_name=model, provider=provider)
    sanity = check_asr_transcript_sanity(transcript.text)
    return {
        "asr_available": transcript.available,
        "asr_text": transcript.text,
        "asr_language": transcript.language,
        "asr_note": transcript.note,
        "asr_sanity_ok": sanity.ok,
        "asr_sanity_reason": sanity.reason,
    }


def _evaluate_confirmed(
    *,
    audio_path: str | Path,
    cache_path: str | Path,
    confirmed_text: str,
    tts_backend: str,
) -> Dict[str, Any]:
    from jp_speech_eval.eval_modes import evaluate_mode
    from jp_speech_eval.feedback_renderer import render_user_facing_result

    result = evaluate_mode(
        "asr_confirmed_weak_reference",
        audio_path,
        cache_path=cache_path,
        user_confirmed_text=confirmed_text,
        tts_backend=tts_backend,
    )
    rendered = render_user_facing_result(result, mode="asr_confirmed_weak_reference")
    details = result.get("details", {})
    prosody = details.get("prosody", {})
    return {
        "weak_reference_evaluated": True,
        "display_score": rendered.get("display_score"),
        "debug_total_score": result.get("total_score"),
        "debug_prosody_score": result.get("prosody_score"),
        "pitch_correctness_available": prosody.get("pitch_correctness_available"),
        "weak_reference": details.get("weak_reference"),
        "reference_source": details.get("reference_source"),
    }


def _base_row(case: Dict[str, Any], *, asr_provider: str, asr_model: str, tts_backend: str) -> Dict[str, Any]:
    return {
        "case_id": case.get("case_id", ""),
        "audio_path": case.get("audio_path", ""),
        "label": case.get("label", ""),
        "asr_provider": asr_provider,
        "asr_model": asr_model,
        "asr_available": "",
        "asr_text": "",
        "asr_language": "",
        "asr_note": "",
        "asr_sanity_ok": "",
        "asr_sanity_reason": "",
        "tts_backend": tts_backend,
        "confirmed_text_source": "",
        "weak_reference_evaluated": False,
        "display_score": "",
        "debug_total_score": "",
        "debug_prosody_score": "",
        "pitch_correctness_available": "",
        "weak_reference": "",
        "reference_source": "",
        "latency_ms": "",
        "error": "",
    }


def run_matrix(
    cases: Iterable[Dict[str, Any]],
    *,
    cache_path: str | Path,
    asr_providers: List[str],
    asr_models: List[str],
    tts_backends: List[str],
    auto_confirm_asr: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        for asr_provider in asr_providers:
            for asr_model in asr_models:
                start = time.perf_counter()
                try:
                    asr = _run_asr(case["audio_path"], provider=asr_provider, model=asr_model)
                    asr_error = ""
                except Exception as exc:
                    asr = {}
                    asr_error = f"{type(exc).__name__}: {exc}"
                for tts_backend in tts_backends:
                    row = _base_row(case, asr_provider=asr_provider, asr_model=asr_model, tts_backend=tts_backend)
                    row.update(asr)
                    row["error"] = asr_error
                    confirmed_text = str(case.get("confirmed_text") or "").strip()
                    confirmed_source = "case_confirmed_text" if confirmed_text else ""
                    if not confirmed_text and auto_confirm_asr and row.get("asr_sanity_ok"):
                        confirmed_text = str(row.get("asr_text") or "").strip()
                        confirmed_source = "auto_confirmed_asr"
                    row["confirmed_text_source"] = confirmed_source
                    if confirmed_text:
                        try:
                            row.update(
                                _evaluate_confirmed(
                                    audio_path=case["audio_path"],
                                    cache_path=cache_path,
                                    confirmed_text=confirmed_text,
                                    tts_backend=tts_backend,
                                )
                            )
                        except Exception as exc:
                            row["error"] = f"{row['error']} | {type(exc).__name__}: {exc}".strip(" |")
                    row["latency_ms"] = int((time.perf_counter() - start) * 1000)
                    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ASR/TTS backends without treating raw ASR as ground truth. "
            "Weak-reference scoring only runs for case confirmed_text or --auto-confirm-asr."
        )
    )
    parser.add_argument("--case-jsonl", required=True, help="JSONL with audio_path, optional confirmed_text, case_id, label.")
    parser.add_argument("--cache", default="cache/ramen_kudasai", help="Base cache for sample rate/config context.")
    parser.add_argument("--out", default="outputs/asr_tts_backend_comparison.csv")
    parser.add_argument("--asr-providers", default="auto", help="Comma-separated: auto,faster-whisper,openai-whisper")
    parser.add_argument("--asr-models", default="small", help="Comma-separated Whisper model names, e.g. small,medium")
    parser.add_argument("--tts-backends", default="pyopenjtalk", help="Comma-separated: pyopenjtalk,aivis_http,google")
    parser.add_argument("--auto-confirm-asr", action="store_true", help="Use sane ASR text as confirmed weak target for experiments only.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = _load_cases(args.case_jsonl)
    if args.limit is not None:
        cases = cases[: args.limit]
    rows = run_matrix(
        cases,
        cache_path=args.cache,
        asr_providers=_split_csv(args.asr_providers),
        asr_models=_split_csv(args.asr_models),
        tts_backends=_split_csv(args.tts_backends),
        auto_confirm_asr=args.auto_confirm_asr,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")
    print("注意: display_score 为空是正常的。ASR-generated weak reference 不再展示综合分或音调正确性。")


if __name__ == "__main__":
    main()
