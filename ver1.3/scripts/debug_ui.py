from __future__ import annotations

import argparse
import cgi
import hashlib
import io
import json
import os
import sys
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.audio_features import load_audio
from jp_speech_eval.audio_features import median_f0_by_mora
from jp_speech_eval.acoustic_evaluator import evaluate_reference_free_acoustic
from jp_speech_eval.asr_confirmation import build_asr_confirmation_prompt
from jp_speech_eval.config import load_scoring_config
from jp_speech_eval.eval_modes import (
    evaluate_asr_confirmed_weak_reference,
    evaluate_asr_pseudo_reference,
    evaluate_kanade_asr_voice_reference,
    evaluate_kanade_voice_reference,
    evaluate_mode,
)
from jp_speech_eval.evaluation_log import append_jsonl, export_feature_table
from jp_speech_eval.evaluator import evaluate_utterance
from jp_speech_eval.realtime_evaluator import RealtimeEvaluator
from jp_speech_eval.sentence_cache import build_sentence_cache, load_sentence_cache
from jp_speech_eval.streaming_features import StreamingFeatureExtractor
from jp_speech_eval.transcript_assisted import evaluate_transcript_assisted_light
from jp_speech_eval.unified_result import unify_evaluation_result
from jp_speech_eval.vad import detect_speech_region
from jp_speech_eval.feedback_renderer import render_user_facing_result


def _json_response(handler: SimpleHTTPRequestHandler, payload: Dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _error_response(handler: SimpleHTTPRequestHandler, message: str, status: int = 400) -> None:
    _json_response(handler, {"ok": False, "error": message}, status=status)


def _none_if_nan(value: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _reference_payload(cache_prefix: Path) -> Dict[str, Any]:
    cache = load_sentence_cache(cache_prefix)
    ref_f0 = median_f0_by_mora(cache.ref_f0_times, cache.ref_f0, cache.meta.ref_mora_boundaries)
    return {
        "target_text": cache.meta.text,
        "kana": cache.meta.kana,
        "moras": cache.meta.moras,
        "target_pitch": cache.meta.target_pitch,
        "pitch_target_source": cache.meta.pitch_target_source,
        "reference_text": cache.meta.reference_text,
        "reference_source": cache.meta.reference_source,
        "ref_boundary_method": cache.meta.ref_boundary_method,
        "ref_duration_sec": cache.meta.ref_duration_sec,
        "ref_mora_boundaries": cache.meta.ref_mora_boundaries,
        "ref_f0_by_mora": [_none_if_nan(v) for v in ref_f0],
    }


def _realtime_rows(wav_path: Path, cache_prefix: Path, config_path: str | None, chunk_ms: float) -> List[Dict[str, Any]]:
    config = load_scoring_config(config_path)
    cache = load_sentence_cache(cache_prefix)
    sr = cache.meta.sr
    audio = load_audio(str(wav_path), sr=sr)
    speech_region = detect_speech_region(audio.y, sr)
    expected_duration = speech_region.speech_duration if speech_region.detected else len(audio.y) / sr

    chunk_samples = max(1, int(sr * chunk_ms / 1000.0))
    extractor = StreamingFeatureExtractor.from_config(config, sr=sr)
    evaluator = RealtimeEvaluator.from_config(cache, config, expected_user_duration_sec=expected_duration)

    rows: List[Dict[str, Any]] = []
    for i in range(0, len(audio.y), chunk_samples):
        chunk = audio.y[i : i + chunk_samples]
        features = extractor.process_chunk(chunk)
        feedback = evaluator.update(features)
        row = feedback.to_dict()
        row.update({
            "rms": float(features.rms),
            "dbfs": float(features.dbfs),
            "is_voiced": bool(features.is_voiced),
        })
        rows.append(row)
    return rows


def _field_value(form: cgi.FieldStorage, name: str, default: str) -> str:
    item = form[name] if name in form else None
    if item is None:
        return default
    value = getattr(item, "value", default)
    return str(value or default)


CORE_MODES = [
    "reference",
    "asr_pseudo_reference",
    "transcript_assisted_light",
    "acoustic",
]

PUBLIC_DEMO_MODES = [
    "reference",
    "asr_pseudo_reference",
    "kanade_asr_voice_reference",
    "transcript_assisted_light",
    "acoustic",
]

EXPERIMENTAL_MODES = [
    "kanade_voice_reference",
    "kanade_asr_voice_reference",
]

ALL_MODES = CORE_MODES + EXPERIMENTAL_MODES


def _mode_labels() -> Dict[str, str]:
    return {
        "reference": "Reference fixed-sentence scoring",
        "asr_pseudo_reference": "Free speech: ASR-generated pseudo-reference",
        "asr_confirmed_weak_reference": "Free speech: confirmed weak-reference practice",
        "transcript_assisted_light": "Free speech: transcript-assisted light diagnosis",
        "acoustic": "Recording/acoustic quality diagnosis",
        "kanade_voice_reference": "Experimental: voice-conditioned fixed-sentence reference",
        "kanade_asr_voice_reference": "Experimental: ASR pseudo-reference with voice playback",
    }


class DebugUiHandler(SimpleHTTPRequestHandler):
    server_version = "JpSpeechEvalDebugUI/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT / "debug_ui"), **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/config":
            cache = load_sentence_cache(self.server.cache_prefix)  # type: ignore[attr-defined]
            payload = {
                "ok": True,
                "target_text": cache.meta.text,
                "kana": cache.meta.kana,
                "moras": cache.meta.moras,
                "target_pitch": cache.meta.target_pitch,
                "pitch_target_source": cache.meta.pitch_target_source,
                "cache_prefix": str(cache.prefix),
                "eval_mode": self.server.eval_mode,  # type: ignore[attr-defined]
                "sample_wav": str(self.server.sample_wav.relative_to(ROOT)),  # type: ignore[attr-defined]
                "chunk_ms": self.server.chunk_ms,  # type: ignore[attr-defined]
                "log_jsonl": str(self.server.log_jsonl.relative_to(ROOT)),  # type: ignore[attr-defined]
                "feature_csv": str(self.server.feature_csv.relative_to(ROOT)),  # type: ignore[attr-defined]
                "reference": _reference_payload(self.server.cache_prefix),  # type: ignore[attr-defined]
                "reference_audio_url": "/api/reference.wav",
                "available_modes": self.server.available_modes,  # type: ignore[attr-defined]
                "mode_labels": _mode_labels(),
                "server_label": self.server.server_label,  # type: ignore[attr-defined]
                "tts_backend": self.server.tts_backend,  # type: ignore[attr-defined]
                "tts_model": self.server.tts_model,  # type: ignore[attr-defined]
                "tts_voice": self.server.tts_voice,  # type: ignore[attr-defined]
            }
            _json_response(self, payload)
            return
        if self.path == "/api/reference.wav":
            self._reference_wav()
            return
        if self.path.startswith("/api/latest-reference.wav"):
            self._serve_wav_file(self.server.latest_reference_wav)  # type: ignore[attr-defined]
            return
        if self.path == "/api/sample.wav":
            self._serve_wav_file(self.server.sample_wav)  # type: ignore[attr-defined]
            return
        if self.path.startswith("/api/evaluate-sample"):
            from urllib.parse import parse_qs, urlparse

            query = parse_qs(urlparse(self.path).query)
            mode = query.get("mode", [self.server.eval_mode])[0]  # type: ignore[attr-defined]
            self._evaluate_wav(self.server.sample_wav, mode=mode)  # type: ignore[attr-defined]
            return
        if self.path.startswith("/api/asr-confirm-sample"):
            prompt = build_asr_confirmation_prompt(self.server.sample_wav)  # type: ignore[attr-defined]
            self.server.asr_confirmation_sessions[prompt.session_id] = str(self.server.sample_wav)  # type: ignore[attr-defined]
            _json_response(self, {"ok": True, **prompt.to_dict()})
            return
        if self.path.startswith("/api/evaluate-confirmed-sample"):
            from urllib.parse import parse_qs, urlparse

            query = parse_qs(urlparse(self.path).query)
            text = query.get("user_confirmed_text", query.get("text", [""]))[0]
            self._evaluate_confirmed_asr(self.server.sample_wav, text)  # type: ignore[attr-defined]
            return
        if self.path.startswith("/api/export-features"):
            count = export_feature_table([self.server.log_jsonl], self.server.feature_csv)  # type: ignore[attr-defined]
            _json_response(self, {
                "ok": True,
                "rows": count,
                "jsonl": str(self.server.log_jsonl.relative_to(ROOT)),  # type: ignore[attr-defined]
                "csv": str(self.server.feature_csv.relative_to(ROOT)),  # type: ignore[attr-defined]
            })
            return
        if self.path.startswith("/api/compare-sample"):
            self._compare_sample()
            return
        return super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/api/evaluate-confirmed-asr":
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                data = json.loads(self.rfile.read(length) or b"{}")
                session_id = str(data.get("session_id") or "").strip()
                text = str(data.get("user_confirmed_text") or data.get("text") or "").strip()
                if not session_id:
                    _error_response(self, "session_id is required")
                    return
                if not text:
                    _error_response(self, "user_confirmed_text is required")
                    return
                wav_raw = self.server.asr_confirmation_sessions.get(session_id)  # type: ignore[attr-defined]
                if not wav_raw:
                    _error_response(self, "ASR confirmation session expired. Please record or upload again.", status=404)
                    return
                wav_path = Path(wav_raw)
                if not wav_path.exists():
                    _error_response(self, "Confirmed audio file is no longer available. Please record or upload again.", status=404)
                    return
                self._evaluate_confirmed_asr(wav_path, text)
            except Exception as exc:
                _error_response(self, f"{type(exc).__name__}: {exc}", status=500)
            return

        if self.path != "/api/evaluate":
            _error_response(self, "Unknown endpoint", status=404)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )
        file_item = form["audio"] if "audio" in form else None
        if file_item is None or not getattr(file_item, "file", None):
            _error_response(self, "Missing audio file field named 'audio'")
            return
        mode = _field_value(form, "mode", self.server.eval_mode)  # type: ignore[attr-defined]
        if mode not in self.server.available_modes:  # type: ignore[attr-defined]
            _error_response(self, f"Mode is not enabled in this deployment: {mode}")
            return

        upload_dir = ROOT / "outputs" / "debug_ui"
        upload_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="recording_", suffix=".wav", dir=upload_dir, delete=False) as f:
            wav_path = Path(f.name)
            f.write(file_item.file.read())
        try:
            self._evaluate_wav(wav_path, mode=mode)
        finally:
            keep_for_confirmation = mode == "asr_pseudo_reference"
            if not self.server.retain_uploads and not keep_for_confirmation:  # type: ignore[attr-defined]
                wav_path.unlink(missing_ok=True)

    def _evaluate_wav(self, wav_path: Path, mode: str | None = None) -> None:
        try:
            mode = (mode or self.server.eval_mode).strip()  # type: ignore[attr-defined]
            if mode not in self.server.available_modes:  # type: ignore[attr-defined]
                raise ValueError(f"Mode is not enabled in this deployment: {mode}")
            if mode == "acoustic":
                result = evaluate_reference_free_acoustic(wav_path)
                realtime = []
                reference = None
                reference_audio_url = None
            elif mode == "transcript_assisted_light":
                cache = load_sentence_cache(self.server.cache_prefix)  # type: ignore[attr-defined]
                result = evaluate_transcript_assisted_light(
                    wav_path,
                    sample_rate=cache.meta.sr,
                    scoring_config_path=self.server.config_path,  # type: ignore[attr-defined]
                )
                realtime = []
                reference = None
                reference_audio_url = None
            elif mode == "asr_pseudo_reference":
                prompt = build_asr_confirmation_prompt(wav_path)
                self.server.asr_confirmation_sessions[prompt.session_id] = str(wav_path)  # type: ignore[attr-defined]
                _json_response(self, {"ok": True, **prompt.to_dict(), "requires_user_confirmation": True})
                return
            elif mode == "kanade_voice_reference":
                result = evaluate_kanade_voice_reference(
                    wav_path,
                    base_cache_path=self.server.cache_prefix,  # type: ignore[attr-defined]
                    scoring_config_path=self.server.config_path,  # type: ignore[attr-defined]
                    generated_cache_dir=ROOT / "outputs" / "debug_ui" / "generated_refs",
                )
                generated_prefix = Path(result.get("cache_prefix", ""))
                reference = _reference_payload(generated_prefix)
                self.server.latest_reference_wav = generated_prefix.with_suffix(".ref.wav")  # type: ignore[attr-defined]
                reference_audio_url = f"/api/latest-reference.wav?mode=kanade_voice_reference&cache={generated_prefix.name}"
                realtime = []
            elif mode == "kanade_asr_voice_reference":
                result = evaluate_kanade_asr_voice_reference(
                    wav_path,
                    base_cache_path=self.server.cache_prefix,  # type: ignore[attr-defined]
                    scoring_config_path=self.server.config_path,  # type: ignore[attr-defined]
                    generated_cache_dir=ROOT / "outputs" / "debug_ui" / "generated_refs",
                    tts_backend=self.server.tts_backend,  # type: ignore[attr-defined]
                    tts_backend_url=self.server.tts_url,  # type: ignore[attr-defined]
                    tts_speaker=self.server.tts_speaker,  # type: ignore[attr-defined]
                    tts_model=self.server.tts_model,  # type: ignore[attr-defined]
                    tts_voice=self.server.tts_voice,  # type: ignore[attr-defined]
                    tts_speed=self.server.tts_speed,  # type: ignore[attr-defined]
                )
                asr_text = result.get("details", {}).get("asr", {}).get("text", "")
                scoring_prefix = Path(result.get("cache_prefix", ""))
                voice_prefix = Path(result.get("details", {}).get("voice_reference_cache_prefix", ""))
                if result.get("details", {}).get("mode") == "kanade_asr_voice_reference" and asr_text:
                    reference = _reference_payload(scoring_prefix)
                    self.server.latest_reference_wav = voice_prefix.with_suffix(".ref.wav")  # type: ignore[attr-defined]
                    reference_audio_url = (
                        f"/api/latest-reference.wav?mode=kanade_asr_voice_reference"
                        f"&text={hashlib.sha1(asr_text.encode('utf-8')).hexdigest()[:12]}"
                    )
                else:
                    reference = None
                    reference_audio_url = None
                realtime = []
            else:
                eval_result = evaluate_utterance(
                    wav_path=wav_path,
                    alignment_mode=self.server.alignment_mode,  # type: ignore[attr-defined]
                    cache_path=self.server.cache_prefix,  # type: ignore[attr-defined]
                    scoring_config_path=self.server.config_path,  # type: ignore[attr-defined]
                    profile=False,
                )
                result = eval_result.to_dict()
                reference = _reference_payload(self.server.cache_prefix)  # type: ignore[attr-defined]
                self.server.latest_reference_wav = self.server.cache_prefix.with_suffix(".ref.wav")  # type: ignore[attr-defined]
                if not self.server.latest_reference_wav.exists():  # type: ignore[attr-defined]
                    self.server.latest_reference_wav = self.server.sample_wav  # type: ignore[attr-defined]
                reference_audio_url = "/api/reference.wav"
                realtime = _realtime_rows(
                    wav_path=wav_path,
                    cache_prefix=self.server.cache_prefix,  # type: ignore[attr-defined]
                    config_path=self.server.config_path,  # type: ignore[attr-defined]
                    chunk_ms=float(self.server.chunk_ms),  # type: ignore[attr-defined]
                )
            unified = unify_evaluation_result(
                result,
                mode=result.get("details", {}).get("mode") or mode,
                audio_path=wav_path,
            )
            if self.server.enable_logs:  # type: ignore[attr-defined]
                append_jsonl(self.server.log_jsonl, unified)  # type: ignore[attr-defined]
            unified_payload = unified.to_dict()
            unified_payload.pop("raw_metrics", None)
            user_facing = render_user_facing_result(result, mode=result.get("details", {}).get("mode") or mode)
            _json_response(self, {
                "ok": True,
                "mode": mode,
                "wav_path": str(wav_path.relative_to(ROOT)),
                "log_jsonl": str(self.server.log_jsonl.relative_to(ROOT)),  # type: ignore[attr-defined]
                "reference": reference,
                "reference_audio_url": reference_audio_url,
                "result": result,
                "unified": unified_payload,
                "user_facing": user_facing,
                "realtime": realtime,
            })
        except Exception as exc:
            _error_response(self, f"{type(exc).__name__}: {exc}", status=500)

    def _evaluate_confirmed_asr(self, wav_path: Path, user_confirmed_text: str) -> None:
        try:
            result = evaluate_asr_confirmed_weak_reference(
                wav_path,
                user_confirmed_text=user_confirmed_text,
                base_cache_path=self.server.cache_prefix,  # type: ignore[attr-defined]
                scoring_config_path=self.server.config_path,  # type: ignore[attr-defined]
                generated_cache_dir=ROOT / "outputs" / "debug_ui" / "generated_refs",
                tts_backend=self.server.tts_backend,  # type: ignore[attr-defined]
                tts_backend_url=self.server.tts_url,  # type: ignore[attr-defined]
                tts_speaker=self.server.tts_speaker,  # type: ignore[attr-defined]
                tts_model=self.server.tts_model,  # type: ignore[attr-defined]
                tts_voice=self.server.tts_voice,  # type: ignore[attr-defined]
                tts_speed=self.server.tts_speed,  # type: ignore[attr-defined]
            )
            generated_prefix = Path(result.get("cache_prefix", ""))
            reference = _reference_payload(generated_prefix)
            self.server.latest_reference_wav = generated_prefix.with_suffix(".ref.wav")  # type: ignore[attr-defined]
            unified = unify_evaluation_result(result, mode="asr_confirmed_weak_reference", audio_path=wav_path)
            if self.server.enable_logs:  # type: ignore[attr-defined]
                append_jsonl(self.server.log_jsonl, unified)  # type: ignore[attr-defined]
            unified_payload = unified.to_dict()
            unified_payload.pop("raw_metrics", None)
            _json_response(self, {
                "ok": True,
                "mode": "asr_confirmed_weak_reference",
                "wav_path": str(wav_path.relative_to(ROOT)) if wav_path.is_relative_to(ROOT) else str(wav_path),
                "reference": reference,
                "reference_audio_url": f"/api/latest-reference.wav?mode=asr_confirmed_weak_reference&text={hashlib.sha1(user_confirmed_text.encode('utf-8')).hexdigest()[:12]}",
                "result": result,
                "unified": unified_payload,
                "user_facing": render_user_facing_result(result, mode="asr_confirmed_weak_reference"),
            })
        except Exception as exc:
            _error_response(self, f"{type(exc).__name__}: {exc}", status=500)

    def _compare_sample(self) -> None:
        modes = [
            mode
            for mode in ["reference", "acoustic", "transcript_assisted_light", "asr_pseudo_reference"]
            if mode in self.server.available_modes  # type: ignore[attr-defined]
        ]
        rows: List[Dict[str, Any]] = []
        for mode in modes:
            try:
                result = evaluate_mode(
                    mode,
                    self.server.sample_wav,  # type: ignore[attr-defined]
                    cache_path=self.server.cache_prefix if mode in {"reference", "asr_pseudo_reference"} else None,  # type: ignore[attr-defined]
                    scoring_config_path=self.server.config_path,  # type: ignore[attr-defined]
                    tts_backend=self.server.tts_backend,  # type: ignore[attr-defined]
                    tts_backend_url=self.server.tts_url,  # type: ignore[attr-defined]
                    tts_speaker=self.server.tts_speaker,  # type: ignore[attr-defined]
                    tts_model=self.server.tts_model,  # type: ignore[attr-defined]
                    tts_voice=self.server.tts_voice,  # type: ignore[attr-defined]
                    tts_speed=self.server.tts_speed,  # type: ignore[attr-defined]
                )
                unified = unify_evaluation_result(
                    result,
                    mode=result.get("details", {}).get("mode") or mode,
                    audio_path=self.server.sample_wav,  # type: ignore[attr-defined]
                )
                if self.server.enable_logs:  # type: ignore[attr-defined]
                    append_jsonl(self.server.log_jsonl, unified)  # type: ignore[attr-defined]
                rows.append({
                    "mode": unified.mode,
                    "scores": unified.scores,
                    "reliability": unified.reliability,
                    "latency_ms": unified.latency_ms,
                    "warnings": unified.warnings,
                    "feedback": unified.feedback,
                })
            except Exception as exc:
                rows.append({"mode": mode, "error": f"{type(exc).__name__}: {exc}"})
        _json_response(self, {
            "ok": True,
            "wav_path": str(self.server.sample_wav.relative_to(ROOT)),  # type: ignore[attr-defined]
            "log_jsonl": str(self.server.log_jsonl.relative_to(ROOT)),  # type: ignore[attr-defined]
            "rows": rows,
        })

    def _reference_wav(self) -> None:
        try:
            import soundfile as sf

            cache = load_sentence_cache(self.server.cache_prefix)  # type: ignore[attr-defined]
            out = io.BytesIO()
            sf.write(out, cache.ref_y, cache.meta.sr, format="WAV")
            body = out.getvalue()
            self._audio_response(body)
        except Exception as exc:
            _error_response(self, f"{type(exc).__name__}: {exc}", status=500)

    def _serve_wav_file(self, wav_path: Path) -> None:
        try:
            self._audio_response(wav_path.read_bytes())
        except Exception as exc:
            _error_response(self, f"{type(exc).__name__}: {exc}", status=500)

    def _audio_response(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local product-like debug UI for jp_speech_eval")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    parser.add_argument("--cache", default="cache/ramen_kudasai")
    parser.add_argument("--wav", default="data/ramen.wav", help="Sample wav used by the Try sample button")
    parser.add_argument("--alignment", default="cached_dtw", choices=["cached_dtw", "dtw", "equal"])
    parser.add_argument(
        "--mode",
        default="reference",
        choices=ALL_MODES,
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--tts-backend", default="pyopenjtalk", help="pyopenjtalk, voicevox_http, aivis_http, or google for generated references")
    parser.add_argument("--tts-url", default=None)
    parser.add_argument("--tts-speaker", type=int, default=None)
    parser.add_argument("--tts-model", default=None, help="Optional provider model id, e.g. chirp3-hd.")
    parser.add_argument("--tts-voice", default=None, help="Optional provider voice id, e.g. ja-JP-Chirp3-HD-Achernar.")
    parser.add_argument("--tts-speed", type=float, default=None, help="Optional provider speaking speed when supported.")
    parser.add_argument("--chunk-ms", type=float, default=20.0)
    parser.add_argument("--log-jsonl", default="outputs/debug_ui/eval_log.jsonl")
    parser.add_argument("--feature-csv", default="outputs/debug_ui/features.csv")
    parser.add_argument(
        "--available-modes",
        default=None,
        help="Comma-separated modes exposed in the UI. Overrides the stable default set.",
    )
    parser.add_argument(
        "--show-experimental-modes",
        action="store_true",
        help="Expose experimental Kanade voice-reference modes in the local UI.",
    )
    parser.add_argument(
        "--public-demo",
        action="store_true",
        help="Public-safe defaults: stable modes only, no retained uploads, no JSONL logs.",
    )
    args = parser.parse_args()

    if args.available_modes:
        available_modes = [mode.strip() for mode in args.available_modes.split(",") if mode.strip()]
    elif args.public_demo:
        available_modes = list(PUBLIC_DEMO_MODES)
    else:
        available_modes = list(CORE_MODES)
        if args.show_experimental_modes:
            available_modes.extend(EXPERIMENTAL_MODES)
    invalid_modes = sorted(set(available_modes) - set(ALL_MODES))
    if invalid_modes:
        raise ValueError(f"Unknown modes in --available-modes: {', '.join(invalid_modes)}")
    if args.mode not in available_modes:
        args.mode = available_modes[0]

    server = ThreadingHTTPServer((args.host, args.port), DebugUiHandler)
    server.cache_prefix = (ROOT / args.cache).resolve() if not Path(args.cache).is_absolute() else Path(args.cache)
    server.sample_wav = (ROOT / args.wav).resolve() if not Path(args.wav).is_absolute() else Path(args.wav)
    server.alignment_mode = args.alignment
    server.eval_mode = args.mode
    server.config_path = args.config
    server.tts_backend = args.tts_backend
    server.tts_url = args.tts_url
    server.tts_speaker = args.tts_speaker
    server.tts_model = args.tts_model
    server.tts_voice = args.tts_voice
    server.tts_speed = args.tts_speed
    server.chunk_ms = float(args.chunk_ms)
    server.log_jsonl = (ROOT / args.log_jsonl).resolve() if not Path(args.log_jsonl).is_absolute() else Path(args.log_jsonl)
    server.feature_csv = (ROOT / args.feature_csv).resolve() if not Path(args.feature_csv).is_absolute() else Path(args.feature_csv)
    server.available_modes = available_modes
    server.asr_confirmation_sessions = {}
    server.retain_uploads = not args.public_demo
    server.enable_logs = not args.public_demo
    server.server_label = "Public demo" if args.public_demo else "Local debug"
    default_ref = server.cache_prefix.with_suffix(".ref.wav")
    server.latest_reference_wav = default_ref if default_ref.exists() else server.sample_wav

    print(f"Debug UI: http://{args.host}:{args.port}")
    print(f"Cache   : {server.cache_prefix}")
    print(f"Sample  : {server.sample_wav}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped debug UI")


if __name__ == "__main__":
    main()
