# Japanese Speech Evaluation Project

Research-oriented prototype for Japanese speaking evaluation, with the current
implementation in [`ver1.3/`](ver1.3/README.md).

Public demo:

```text
https://kayui-gavo-jp-speech-eval-demo.hf.space/
```

For a concise overview of the current research direction and demo workflow:

- [Chinese research overview](docs/research_overview_zh.md)
- [Chinese demo guide](docs/demo_guide_zh.md)

The project focuses on lightweight, inspectable evaluation rather than opaque
end-to-end scoring. Current capabilities include:

- Japanese text frontend with mora and accent-phrase analysis
- sentence-final pronunciation, prosody, fluency, and expression proxies
- realtime lightweight acoustic feedback
- reliability-aware gating for unstable evidence
- optional ASR-assisted content checks
- pluggable pseudo-reference TTS backends, including AivisSpeech-compatible HTTP engines

## Quick Start

```bash
cd ver1.3
python3.11 -m venv ../.venv
source ../.venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/prepare_cache.py \
  --text "ラーメンをください" \
  --out cache/ramen_kudasai \
  --save-ref-wav
python scripts/debug_ui.py
```

Then open `http://127.0.0.1:8765/`.

For the full setup guide, evaluation limitations, and optional AivisSpeech /
Kanade experiments, see [`ver1.3/README.md`](ver1.3/README.md).

## Full Web Demo

The repository includes a Dockerized full-stack web deployment:

```bash
docker build -t jp-speech-eval-demo .
docker run --rm -p 7860:7860 jp-speech-eval-demo
```

This starts the UI on `http://127.0.0.1:7860/` with the same evaluation modes as
the local debug UI, including ASR-generated references, AivisSpeech-backed TTS,
and Kanade voice-reference experiments. The container starts AivisSpeech Engine
internally and installs the isolated Python 3.12 Kanade worker used by the local
setup.

The same `Dockerfile` can be deployed on a container host such as Hugging Face
Docker Spaces or Render. This is intentionally a heavy deployment: first startup
downloads AivisSpeech model assets and runtime latency depends on the host CPU.

For public hosting, the container uses the same TTS backend for the normal
ASR-generated reference mode and the ASR + Kanade voice-reference mode. By
default it falls back to the bundled AivisSpeech engine. To use Google Cloud TTS
for both modes, add a Hugging Face Space secret named
`GOOGLE_APPLICATION_CREDENTIALS_JSON` containing the service-account JSON. The
startup script writes that secret to a temporary credentials file and switches
the dynamic reference backend to Google Chirp 3 HD
(`ja-JP-Chirp3-HD-Achernar`) automatically.

## Important Caveat

This repository is a prototype. Its reference audio is still a
`pseudo-reference`, ASR transcripts are not ground truth, and the current scores
are best treated as evidence-weighted proxies rather than certified language
assessment labels.
