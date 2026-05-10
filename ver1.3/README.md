# jp_speech_eval_project v1.5 mora-evidence speech-evaluation patch

A lightweight Python prototype for Japanese speaking evaluation.

This version adds:

- sentence cache: precompute target text, kana, mora, pitch pattern, TTS reference, reference MFCC/F0
- faster sentence-final scoring with `cached_dtw`
- realtime simulation: karaoke-like frame-level volume / F0 / pitch feedback
- timing profile for speed analysis
- clearer separation between theory-backed acoustic features and engineering thresholds
- endpointing / VAD trimming so fixed-duration recordings do not count leading or trailing silence as fluency, pause-ratio, or speech-rate evidence
- confidence-aware feedback: unreliable mora alignment or low F0 coverage is reported as a diagnostic limitation instead of being treated as a pronunciation error
- content-match gating: optional Whisper/faster-whisper ASR kana comparison when installed, with MFCC-DTW reference matching as a lightweight fallback
- recording-quality gate: estimated SNR, clipping, noise floor, and dynamic range reduce reliability instead of being treated as pronunciation errors
- structure-aware features for future training: speaker-normalized F0 movement, mora rate, special-mora density, and compression-risk flags are exported to JSON/CSV
- mora-level evidence gate: each mora now records boundary confidence, energy coverage, F0 coverage, duration plausibility, and whether strong mora-level judgement is available

Current scoring dimensions:

- pronunciation proxy: core pronunciation-related mora rhythm + special mora duration
- prosody: core pronunciation-related mora-level F0 / H-L pattern / sentence-final intonation
- fluency: delivery/style, based on endpointed speech rate + in-speech pauses
- tone / emotion proxy: expression/style, based on pitch range + energy + in-speech pause ratio

Important limitation: ASR is now an optional content gate, not a full phoneme-level pronunciation model. To judge fine substitutions such as `す` vs `ず` robustly, add CTC/GOP or a trained Japanese phoneme/kana aligner.

Endpointing note: `record_mic.py` still records a fixed-duration wav for CLI compatibility, but sentence-final scoring first detects `speech_start` and `speech_end`. The JSON keeps raw diagnostics under `endpointing` while duration-based scores use `speech_duration`.

Recording-quality note: v1.4 separates environment/device evidence from pronunciation evidence. A noisy microphone, clipped input, or aggressive browser noise suppression should lower `details.reliability.recording_quality` and add warnings, not create strong kana/pronunciation error feedback.

Mora-evidence note: v1.5 separates "this mora is wrong" from "this mora has enough acoustic evidence to judge." Low boundary confidence, low energy coverage, or low F0 coverage should suppress strong mora-level correction and lower `details.reliability.mora_evidence`.

---

## 1. Environment

Recommended: Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional ASR content gate:

```bash
python -m pip install faster-whisper
```

If no ASR backend is installed, the evaluator automatically falls back to MFCC-DTW content matching and reports this in `details.content_match.note`.

If you already installed the previous version, you can keep the same `.venv` and run:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

---

## 2. Test text frontend

```bash
python scripts/inspect_text.py --text "ラーメンをください"
```

Expected output includes:

```text
Kana: ラーメンヲクダサイ
Mora: ラ・ー・メ・ン・ヲ・ク・ダ・サ・イ
```

---

## 3. Prepare sentence cache

Do this once for each target/candidate sentence.

```bash
mkdir -p cache data outputs
python scripts/prepare_cache.py \
  --text "ラーメンをください" \
  --out cache/ramen_kudasai \
  --save-ref-wav
```

This creates:

```text
cache/ramen_kudasai.json
cache/ramen_kudasai.npz
cache/ramen_kudasai.ref.wav   # optional
```

The slow work is done here, not during realtime use.

---

## 4. Record a test wav

```bash
python scripts/record_mic.py --out data/ramen.wav --seconds 4
```

Say:

```text
ラーメンをください
```

---

## 5. Fast sentence-final evaluation

Recommended command:

```bash
python -m jp_speech_eval.cli \
  --cache cache/ramen_kudasai \
  --wav data/ramen.wav \
  --alignment cached_dtw \
  --json outputs/ramen_fast.json \
  --profile \
  --no-plot
```

Debug with plot:

```bash
python -m jp_speech_eval.cli \
  --cache cache/ramen_kudasai \
  --wav data/ramen.wav \
  --alignment cached_dtw \
  --json outputs/ramen_fast.json \
  --plot outputs/ramen_f0.png
```

Old style still works:

```bash
python -m jp_speech_eval.cli \
  --text "ラーメンをください" \
  --wav data/ramen.wav \
  --alignment equal
```

---

## 6. Realtime simulation

This simulates karaoke-like feedback by streaming a wav file chunk by chunk.

```bash
python scripts/realtime_sim.py \
  --cache cache/ramen_kudasai \
  --wav data/ramen.wav \
  --chunk-ms 20 \
  --csv outputs/ramen_realtime.csv
```

Example output:

```text
  0.18s  mora=01:ラ   T=H  f0= 145.5Hz  vol=ok        pitch=warming_up
  0.36s  mora=02:ー   T=L  f0= 160.0Hz  vol=ok        pitch=too_high
```

This realtime layer is intentionally coarse. Detailed mora alignment and scoring should still run at sentence end.

---

## 7. Benchmark speed

```bash
python scripts/benchmark_eval.py \
  --cache cache/ramen_kudasai \
  --wav data/ramen.wav \
  --alignment cached_dtw \
  --repeat 5
```

Use this to identify bottlenecks before moving to app/mobile.

---

## 8. Product-like debug UI

Run a local browser UI for recording/uploading audio and inspecting endpointing, realtime state, scores, feedback, and raw JSON:

```bash
../.venv/bin/python scripts/debug_ui.py \
  --host 127.0.0.1 \
  --port 8765 \
  --cache cache/ramen_kudasai \
  --wav data/ramen.wav
```

Then open:

```text
http://127.0.0.1:8765
```

The UI is for debugging product behavior. It keeps all scoring in the existing Python evaluator, records WAV in the browser, and stores uploaded recordings under `outputs/debug_ui/`.

The practice prompt includes separate audio players for the model reference and the saved user sample. The contour chart compares speaker-normalized relative log-F0 movement, not absolute Hz, so it is meant for pitch-accent movement/debugging rather than judging personal voice pitch or timbre.

The UI also shows reliability diagnostics:

- content-match status, ASR transcript/kana when available, DTW cost, and duration ratio
- endpointing confidence
- alignment confidence
- F0 coverage by mora
- whether the score should be read as a diagnostic result rather than a learner ability judgement

---

## 9. Project structure

```text
src/jp_speech_eval/
  text_frontend.py        # Japanese text -> kana/mora/pitch pattern
  sentence_cache.py       # precompute target sentence cache
  audio_features.py       # offline F0, energy, pause features
  vad.py                  # endpointing / VAD speech region detection
  alignment.py            # equal / legacy dtw / cached_dtw mora alignment
  streaming_features.py   # realtime RMS + autocorrelation F0
  realtime_evaluator.py   # karaoke-like coarse feedback
  scoring.py              # four scoring dimensions
  evaluator.py            # sentence-final detailed evaluation
  cli.py                  # command line entry

scripts/
  prepare_cache.py
  realtime_sim.py
  debug_ui.py
  benchmark_eval.py
  record_mic.py
  inspect_text.py

debug_ui/
  index.html              # local product-like debug page

configs/scoring_config.json
  Engineering thresholds. Calibrate before research claims.

docs/theory_basis.md
  Theory basis and limitations.

docs/non_reference_realtime_strategy.md
  Strategy for realtime free-speaking modes: acoustic-only, ASR pseudo-reference,
  transcript-assisted diagnosis, and future CTC/GOP evidence.

docs/evaluation_infrastructure.md
  Unified result schema, JSONL logging, feature-table export, and mode comparison.
```

## 11. Evaluation logs and feature tables

Append a unified JSONL log from the existing reference-based CLI:

```bash
python -m jp_speech_eval.cli \
  --cache cache/ramen_kudasai \
  --wav data/ramen.wav \
  --alignment cached_dtw \
  --log-jsonl outputs/eval_log.jsonl \
  --no-plot
```

Compare several evaluation modes on the same wav:

```bash
python scripts/compare_modes.py \
  --wav data/ramen.wav \
  --cache cache/ramen_kudasai \
  --modes reference,acoustic,transcript_assisted_light,asr_pseudo_reference \
  --jsonl outputs/compare_log.jsonl \
  --csv outputs/compare_features.csv
```

Convert JSONL logs to a training CSV:

```bash
python scripts/export_feature_table.py \
  --jsonl outputs/eval_log.jsonl outputs/compare_log.jsonl \
  --csv outputs/features.csv
```

The exported scores are still rule/proxy scores. Teacher/native/listener labels
should be added to the reserved annotation columns before training calibrated
models.

---

## 10. Recommended app architecture

Use two layers:

```text
During speech:
  streaming_features.py + realtime_evaluator.py
  -> low-latency endpoint state + volume/F0/pitch feedback

After sentence end:
  evaluator.py with cached_dtw
  -> detailed pronunciation/prosody/fluency/tone feedback using endpointed speech duration
```

Slow target-sentence work must be done before the user speaks:

```text
scenario/candidate sentence -> prepare_cache.py -> cache/*.json + cache/*.npz
```
