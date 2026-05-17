# Japanese Speech Evaluation Project

Research-oriented prototype for Japanese speaking evaluation, with the current
implementation in [`ver1.3/`](ver1.3/README.md).

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

## Important Caveat

This repository is a prototype. Its reference audio is still a
`pseudo-reference`, ASR transcripts are not ground truth, and the current scores
are best treated as evidence-weighted proxies rather than certified language
assessment labels.
