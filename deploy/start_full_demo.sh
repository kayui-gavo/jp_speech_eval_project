#!/usr/bin/env bash
set -euo pipefail

AIVIS_RUN="${AIVIS_ENGINE_DIR:-/opt/aivis-engine}/Linux-x64/run"
AIVIS_URL="${AIVIS_URL:-http://127.0.0.1:10101}"
AIVIS_SPEAKER="${AIVIS_SPEAKER:-888753760}"
APP_PORT="${PORT:-7860}"
GOOGLE_TTS_MODEL="${GOOGLE_TTS_MODEL:-chirp3-hd}"
GOOGLE_TTS_VOICE="${GOOGLE_TTS_VOICE:-ja-JP-Chirp3-HD-Achernar}"
TTS_BACKEND="aivis_http"
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
  echo "Using Google Cloud TTS reference backend: model=${TTS_MODEL} voice=${TTS_VOICE}"
else
  echo "Google Cloud TTS credentials not configured; using bundled AivisSpeech fallback."
  "${AIVIS_RUN}" --host 127.0.0.1 --port 10101 &
  AIVIS_PID=$!
fi

cleanup() {
  if [[ -n "${AIVIS_PID}" ]]; then
    kill "${AIVIS_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${TTS_BACKEND}" == "aivis_http" ]]; then
python - <<'PY'
import os
import time
from urllib import request

url = os.environ.get("AIVIS_URL", "http://127.0.0.1:10101").rstrip("/") + "/version"
deadline = time.time() + 900
last_error = None
while time.time() < deadline:
    try:
        with request.urlopen(url, timeout=5) as resp:
            print("AivisSpeech ready:", resp.read().decode("utf-8"))
            break
    except Exception as exc:
        last_error = exc
        time.sleep(2)
else:
    raise RuntimeError(f"AivisSpeech did not become ready: {last_error}")
PY
fi

export TTS_BACKEND TTS_MODEL TTS_VOICE

# Prewarm the lazy-loaded Japanese frontend and selected TTS backend before
# the browser sees the app. This keeps the first visible interaction from paying
# those one-time costs.
python - <<'PY'
import os
from jp_speech_eval.text_frontend import build_text_info
from jp_speech_eval.tts_backends import synthesize_reference

build_text_info("ラーメンをください")
backend = os.environ.get("TTS_BACKEND", "aivis_http")
kwargs = {"backend": backend}
if backend == "aivis_http":
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
PY

CMD=(python scripts/debug_ui.py
  --host 0.0.0.0
  --port "${APP_PORT}"
  --mode reference
  --wav cache/ramen_kudasai.ref.wav
  --tts-backend "${TTS_BACKEND}"
  --public-demo
  --available-modes reference,asr_pseudo_reference,kanade_asr_voice_reference,transcript_assisted_light,acoustic)

if [[ "${TTS_BACKEND}" == "aivis_http" ]]; then
  CMD+=(--tts-url "${AIVIS_URL}" --tts-speaker "${AIVIS_SPEAKER}")
elif [[ "${TTS_BACKEND}" == "google" ]]; then
  CMD+=(--tts-model "${TTS_MODEL}" --tts-voice "${TTS_VOICE}")
fi

exec "${CMD[@]}"
