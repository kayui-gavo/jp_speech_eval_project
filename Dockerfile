FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      ffmpeg \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY ver1.3/requirements.txt /app/ver1.3/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/ver1.3/requirements.txt

COPY ver1.3 /app/ver1.3
RUN python -m pip install -e /app/ver1.3

WORKDIR /app/ver1.3
EXPOSE 7860

CMD ["python", "scripts/debug_ui.py", "--host", "0.0.0.0", "--port", "7860", "--public-demo"]
