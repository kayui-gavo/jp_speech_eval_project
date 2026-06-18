FROM python:3.11-slim

ARG AIVIS_ENGINE_VERSION=1.2.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

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
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0" \
    && python -m pip install -r /app/ver1.3/requirements.txt

COPY ver1.3 /app/ver1.3
RUN python -m pip install -e /app/ver1.3

COPY deploy/start_full_demo.sh /app/deploy/start_full_demo.sh
RUN chmod +x /app/deploy/start_full_demo.sh

WORKDIR /app/ver1.3
EXPOSE 7860

CMD ["/app/deploy/start_full_demo.sh"]
