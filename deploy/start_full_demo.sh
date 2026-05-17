#!/usr/bin/env bash
set -euo pipefail

AIVIS_RUN="${AIVIS_ENGINE_DIR:-/opt/aivis-engine}/Linux-x64/run"
AIVIS_URL="${AIVIS_URL:-http://127.0.0.1:10101}"
AIVIS_SPEAKER="${AIVIS_SPEAKER:-888753760}"
APP_PORT="${PORT:-7860}"

"${AIVIS_RUN}" --host 127.0.0.1 --port 10101 &
AIVIS_PID=$!

cleanup() {
  kill "${AIVIS_PID}" 2>/dev/null || true
}
trap cleanup EXIT

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

exec python scripts/debug_ui.py \
  --host 0.0.0.0 \
  --port "${APP_PORT}" \
  --mode asr_pseudo_reference \
  --tts-backend aivis_http \
  --tts-url "${AIVIS_URL}" \
  --tts-speaker "${AIVIS_SPEAKER}"
