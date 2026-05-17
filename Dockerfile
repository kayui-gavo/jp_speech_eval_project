FROM python:3.11-slim

ARG AIVIS_ENGINE_VERSION=1.2.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    AIVIS_ENGINE_DIR=/opt/aivis-engine \
    KANADE_PYTHON=/app/.venv-kanade/bin/python

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      curl \
      ffmpeg \
      git \
      libsndfile1 \
      p7zip-full \
    && rm -rf /var/lib/apt/lists/*

COPY ver1.3/requirements.txt /app/ver1.3/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel uv \
    && python -m pip install -r /app/ver1.3/requirements.txt

# Install the same isolated Python 3.12 Kanade worker used by the local setup.
RUN uv python install 3.12 \
    && uv venv --python 3.12 /app/.venv-kanade \
    && uv pip install --python /app/.venv-kanade/bin/python \
      git+https://github.com/frothywater/kanade-tokenizer

# Bundle the Linux AivisSpeech Engine binary so the public deployment can use
# the same local HTTP TTS backend as the developer machine.
RUN mkdir -p "${AIVIS_ENGINE_DIR}" \
    && curl -L --fail \
      -o /tmp/aivis-engine.7z.001 \
      "https://github.com/Aivis-Project/AivisSpeech-Engine/releases/download/${AIVIS_ENGINE_VERSION}/AivisSpeech-Engine-Linux-x64-${AIVIS_ENGINE_VERSION}.7z.001" \
    && 7z x /tmp/aivis-engine.7z.001 -o"${AIVIS_ENGINE_DIR}" \
    && rm /tmp/aivis-engine.7z.001

COPY ver1.3 /app/ver1.3
RUN python -m pip install -e /app/ver1.3

COPY deploy/start_full_demo.sh /app/deploy/start_full_demo.sh
RUN chmod +x /app/deploy/start_full_demo.sh

WORKDIR /app/ver1.3
EXPOSE 7860

CMD ["/app/deploy/start_full_demo.sh"]
