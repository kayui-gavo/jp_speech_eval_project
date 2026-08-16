# Free-Speech Acceptance v8 — Local Runbook

## Purpose

This runbook executes the next evidence gate for the C-end free-speech path. It does not tune thresholds and does not promote the shadow partial-evidence aggregate.

The runtime contract is fixed:

- evaluator path: real `transcript_assisted_light` product path;
- `transcript=None` for every scoring call;
- no gold/manual response transcript;
- current ProductScore remains authoritative for product behavior;
- `partial_evidence_aggregate_shadow_v1` is telemetry only;
- held recordings must not be used to tune score thresholds or shrinkage.

## 1. Prepare a manifest

Use:

`data/human_eval/free_speech_sample_manifest_template_v1.csv`

The corresponding schema is:

`data/human_eval/free_speech_sample_manifest_schema_v1.json`

Minimum useful held acceptance should contain, where legally/operationally available:

- learner Japanese;
- native Japanese;
- short natural Japanese turns such as acknowledgements and brief answers;
- longer spontaneous Japanese answers;
- English speech negative controls;
- Mandarin Chinese speech negative controls;
- `expected_language=non_speech` controls for silence/noise/unusable input;
- same-source clean + device/noise/codec variants for channel sensitivity.

Do not relabel Japanese learner speech as a non-Japanese control.

Do not use separately re-spoken takes as a same-source channel pair.

## 2. Run the entire acceptance pipeline

From `ver1.3/`:

```bash
python scripts/run_free_speech_acceptance_v8.py \
  data/human_eval/your_manifest.csv \
  --audio-root /absolute/path/to/audio/root \
  --out-dir outputs/free_speech_acceptance_v8
```

Optional ASR selection:

```bash
python scripts/run_free_speech_acceptance_v8.py \
  data/human_eval/your_manifest.csv \
  --audio-root /absolute/path/to/audio/root \
  --out-dir outputs/free_speech_acceptance_v8 \
  --asr-model small \
  --asr-provider faster-whisper
```

The runner is resumable by default. Use `--no-resume` only when intentionally rebuilding the output from scratch.

## 3. Outputs

The runner writes:

- `free_speech_validation_v8.jsonl`
  - append/resume-safe per-sample product-condition results;
  - current user-score contract;
  - shadow partial-evidence candidate;
  - no gold transcript use.

- `partial_evidence_analysis_v8.json`
  - current ProductScore distribution;
  - shadow candidate distribution;
  - candidate-current delta;
  - effective evidence coverage;
  - neutral-prior count;
  - task/speaker/channel breakdowns.

- `acceptance_summary_v8.json`
  - manifest validation result;
  - batch execution summary;
  - routing/no-score summary;
  - embedded partial-evidence analysis;
  - explicit no-promotion decision.

## 4. Routing metrics must stay separated

The acceptance summary deliberately separates:

- `expected_japanese`
- `expected_non_japanese_speech`
- `expected_nonspeech_control`

Do not merge these into one failure rate.

Important counts include:

- valid-Japanese false no-score count;
- non-Japanese-speech normal-score count;
- nonspeech-control normal-score count.

A language-routing false accept and a silence/noise false accept are different product failures.

## 5. What to inspect before any score-contract proposal

### Valid Japanese availability

Ordinary analyzable Japanese should normally receive a score. Inspect false no-score cases manually before changing any gate.

Short Japanese must not fail merely because it is too short for pseudo-reference synthesis.

### Negative-control safety

English/Mandarin speech and nonspeech controls should not receive a normal Japanese practice score.

Do not tune against the held controls after viewing the results. Fix a clear implementation bug only when the failure mechanism is independently understood, then rerun on a new held set if a threshold/model decision changed.

### Score dispersion

Compare current vs partial-evidence shadow:

- range;
- IQR;
- ceiling concentration;
- task-mode behavior;
- speaker-group behavior.

More dispersion is not automatically better. A candidate that spreads scores by becoming microphone-sensitive or speaker-identity-sensitive must not be promoted.

### Channel sensitivity

Use only same-source derived channel pairs. Re-speaking is not a channel perturbation.

### Evidence coverage

Inspect how often the shadow aggregate has one, two, three, or four eligible dimensions. A visually attractive headline score with very low evidence coverage should remain strongly shrunk/low-confidence.

## 6. Human criterion remains mandatory for construct promotion

The descriptive acceptance can reject an unsafe candidate, but it cannot prove construct validity.

After routing/channel/dispersion safety, join the already-frozen listener protocol for:

- clarity/comprehensibility;
- fluency;
- rhythm naturalness;
- utterance-level intonation naturalness;
- contextual intonation appropriateness separately.

Do not promote ASR recoverability as clarity, word timing as rhythm, or global F0 movement as intonation solely because a score distribution looks useful.

## 7. Decision boundary

Possible conclusions from the acceptance run are:

- `unsafe` — valid Japanese false no-score, negative-control false score, or unacceptable channel sensitivity;
- `insufficient` — too little speaker/task/channel coverage;
- `promising_for_listener_validation` — routing and robustness are acceptable and the candidate has useful non-pathological dispersion;
- never `production_validated` from this run alone.

Any official change to neutral-prior aggregation or component promotion requires a new ScoreContract version only after held real-audio and construct-matched human validation.
