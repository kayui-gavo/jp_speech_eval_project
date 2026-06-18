#!/usr/bin/env bash
set -euo pipefail

AIVIS_RUN="${AIVIS_ENGINE_DIR:-/opt/aivis-engine}/Linux-x64/run"
AIVIS_URL="${AIVIS_URL:-http://127.0.0.1:10101}"
AIVIS_SPEAKER="${AIVIS_SPEAKER:-888753760}"
APP_HOST="${HOST:-0.0.0.0}"
APP_PORT="${PORT:-7860}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GOOGLE_TTS_MODEL="${GOOGLE_TTS_MODEL:-chirp3-hd}"
GOOGLE_TTS_VOICE="${GOOGLE_TTS_VOICE:-ja-JP-Chirp3-HD-Achernar}"
ENABLE_AIVIS="${ENABLE_AIVIS:-0}"
PREWARM_REFERENCES="${PREWARM_REFERENCES:-1}"
PUBLIC_DEMO_FAST_START="${PUBLIC_DEMO_FAST_START:-1}"
TTS_BACKEND="pyopenjtalk"
TTS_MODEL=""
TTS_VOICE=""
AIVIS_PID=""

if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS_JSON:-}" && -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  GOOGLE_CREDENTIALS_PATH="/tmp/google-tts-service-account.json"
  printf '%s' "${GOOGLE_APPLICATION_CREDENTIALS_JSON}" > "${GOOGLE_CREDENTIALS_PATH}"
  chmod 600 "${GOOGLE_CREDENTIALS_PATH}"
  export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_CREDENTIALS_PATH}"
fi

if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" && -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]]; then
  TTS_BACKEND="google"
  TTS_MODEL="${GOOGLE_TTS_MODEL}"
  TTS_VOICE="${GOOGLE_TTS_VOICE}"
  echo "[demo] Using Google Cloud TTS reference backend: model=${TTS_MODEL} voice=${TTS_VOICE}"
elif [[ "${ENABLE_AIVIS}" == "1" ]]; then
  TTS_BACKEND="aivis_http"
  echo "[demo] Starting AivisSpeech fallback in background."
  "${AIVIS_RUN}" --host 127.0.0.1 --port 10101 &
  AIVIS_PID=$!
else
  echo "[demo] Google Cloud TTS credentials not configured."
  echo "[demo] AivisSpeech disabled by default for fast public demo startup."
  echo "[demo] Using pyopenjtalk cached/local pseudo-reference fallback."
fi

cleanup() {
  if [[ -n "${AIVIS_PID}" ]]; then
    kill "${AIVIS_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

export TTS_BACKEND TTS_MODEL TTS_VOICE
export AIVIS_URL AIVIS_SPEAKER

if [[ "${PREWARM_REFERENCES}" == "1" ]]; then
  echo "[demo] Starting reference warmup in background."
"${PYTHON_BIN}" - <<'PY' &
import os
import time
from jp_speech_eval.text_frontend import build_text_info
from jp_speech_eval.tts_backends import synthesize_reference

try:
    build_text_info("ラーメンをください")
    backend = os.environ.get("TTS_BACKEND", "pyopenjtalk")
    kwargs = {"backend": backend}
    if backend == "aivis_http":
        from urllib import request

        url = os.environ.get("AIVIS_URL", "http://127.0.0.1:10101").rstrip("/") + "/version"
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                with request.urlopen(url, timeout=3) as resp:
                    print("[demo] AivisSpeech ready:", resp.read().decode("utf-8"))
                    break
            except Exception:
                time.sleep(2)
        else:
            print("[demo] AivisSpeech not ready; skipping Aivis warmup.")
            raise SystemExit(0)
        kwargs.update(
            base_url=os.environ.get("AIVIS_URL", "http://127.0.0.1:10101"),
            speaker=int(os.environ.get("AIVIS_SPEAKER", "888753760")),
        )
    elif backend == "google":
        kwargs.update(
            model=os.environ.get("TTS_MODEL") or None,
            voice=os.environ.get("TTS_VOICE") or None,
        )
    synthesize_reference("ラーメンをください", sr=16000, **kwargs)
    print("[demo] Reference warmup completed.")
except Exception as exc:
    print(f"[demo] Reference warmup skipped: {type(exc).__name__}: {exc}")
PY
fi

CMD=("${PYTHON_BIN}" scripts/debug_ui.py
  --host "${APP_HOST}"
  --port "${APP_PORT}"
  --mode reference
  --wav cache/ramen_kudasai.ref.wav
  --tts-backend "${TTS_BACKEND}"
  --public-demo
  --available-modes reference,asr_pseudo_reference)

if [[ "${TTS_BACKEND}" == "aivis_http" ]]; then
  CMD+=(--tts-url "${AIVIS_URL}" --tts-speaker "${AIVIS_SPEAKER}")
elif [[ "${TTS_BACKEND}" == "google" ]]; then
  CMD+=(--tts-model "${TTS_MODEL}" --tts-voice "${TTS_VOICE}")
fi

echo "[demo] Starting debug UI server..."
exec "${CMD[@]}"
