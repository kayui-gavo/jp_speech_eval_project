# Evaluation Infrastructure

This document records the lightweight infrastructure added before training
calibrated pronunciation models.

## 1. Output mismatch found in current modes

The existing evaluators were useful but not training-friendly:

- Reference-based evaluation returned an `EvaluationResult` dataclass.
- Reference-free acoustic evaluation returned a plain `dict`.
- Mode names were stored inconsistently: sometimes top-level, sometimes under
  `details.mode`, sometimes inferred from alignment/cache.
- Features were scattered:
  - endpointing under `endpointing` and `details.endpointing`
  - acoustic features under `details.acoustic_features`
  - prosody features under `prosody_metrics`, `details.prosody_metrics`, and
    `details.prosody`
  - content/ASR features under `details.content_match` or `details.asr`
- Warnings were split between `reliability.warnings`, feedback text, and content
  match notes.
- Latency was stored as seconds in `timing.total`, with no unified `latency_ms`.

The fix is an adapter layer, not a rewrite.

## 2. Unified schema

`src/jp_speech_eval/unified_result.py` provides `UnifiedEvaluationResult`:

- `mode`
- `input_info`
- `features`
- `scores`
- `reliability`
- `warnings`
- `feedback`
- `debug`
- `latency_ms`
- `raw_metrics`

This schema keeps old raw outputs in `raw_metrics` while exposing stable fields
for logging, comparison, and model training.

## 3. JSONL evaluation log

`src/jp_speech_eval/evaluation_log.py` appends one JSON object per sample.

Each record includes:

- audio path
- target text
- ASR transcript if available
- mode
- acoustic/prosody/alignment/ASR features
- rule scores
- reliability
- warnings
- feedback
- raw metrics

Reserved annotation columns:

- `teacher_pronunciation_score`
- `teacher_prosody_score`
- `teacher_fluency_score`
- `native_naturalness_score`
- `listener_intelligibility_score`
- `error_long_vowel`
- `error_sokuon`
- `error_nasal`
- `error_pitch_accent`

## 4. Feature table export

`scripts/export_feature_table.py` converts JSONL logs into CSV for training
Ridge, RandomForest, LightGBM, or a small MLP.

Targets should be trained per dimension first:

- pronunciation score
- prosody score
- fluency score
- naturalness / intelligibility
- specific error labels

Do not train only a single overall score at the beginning.

## 5. Mode comparison

`scripts/compare_modes.py` runs the same wav through multiple modes and logs the
results in a shared schema.

Supported modes:

- `reference`
- `acoustic`
- `transcript_assisted_light`
- `asr_pseudo_reference`

Important interpretation rules:

- `reference_free_acoustic` cannot output kana correctness.
- `transcript_assisted_light` uses transcript for mora count only; it does not
  do TTS reference or DTW.
- `asr_pseudo_reference` uses ASR transcript as pseudo-reference; ASR transcript
  is not ground truth.
- TTS-generated reference must not be called native reference.

## 6. Minimal transcript-assisted light mode

`src/jp_speech_eval/transcript_assisted.py` implements:

```text
user_wav -> VAD -> F0/energy/pause -> transcript or ASR transcript
         -> kana/mora count -> mora rate / acoustic proxy feedback
```

It does not generate TTS, does not run DTW, and does not output mora-level
correction.

This mode is useful as a future free-speaking bridge because it is lighter and
less reference-dependent than ASR pseudo-reference.

## 7. Structural feature additions

`src/jp_speech_eval/structure_features.py` adds lightweight features intended
for future regression/classification models:

- mora count, mora rate, average mora duration
- long vowel / sokuon / nasal counts
- special-mora density
- special-mora compression risk
- speaker-normalized F0 range, slope, direction-change rate, and final movement

These fields are exported through the unified schema and CSV table. They are
not hard correctness labels; they are acoustic/phonological proxies designed to
support later teacher-labeled training.

## 8. v1.4 recording-quality gate

`src/jp_speech_eval/recording_quality.py` estimates input/channel conditions:

- estimated SNR
- noise floor
- clipping ratio
- dynamic range
- recording-quality score

This evidence is attached to `details.recording_quality` and exported as CSV
features. It should lower reliability when the microphone/environment is bad,
but it must not be used as direct pronunciation correctness.

## 9. v1.5 mora-level evidence gate

`src/jp_speech_eval/mora_evidence.py` adds an evidence layer for fixed-sentence
evaluation. For every mora, it records:

- boundary confidence
- energy coverage
- F0 coverage
- duration expectedness
- special mora type
- whether strong mora-level judgement is available

This is the main guard against overclaiming. A future segmental/pronunciation
model should first check whether the mora has enough evidence, then decide
whether it is correct. Low evidence should suppress strong correction rather
than produce misleading feedback.
