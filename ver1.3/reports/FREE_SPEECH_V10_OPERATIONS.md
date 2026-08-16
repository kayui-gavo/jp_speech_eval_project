# Free-Speech v10 Operations

This is the shortest practical path from an empty collection directory to a promotion-readiness report.

It does **not** change ProductScore. A machine candidate remains shadow until held human criteria exist and both the frozen v5 gates and additive v10 C-end gates pass.

## 1. Generate the recording assignments

From `ver1.3/`:

```bash
python scripts/prepare_free_speech_collection_v10.py \
  --assignments-out outputs/free_speech_v10_collection/assignments.csv \
  --speaker-metadata-out outputs/free_speech_v10_collection/speaker_metadata_private.csv \
  --report-out outputs/free_speech_v10_collection/assignment_report.json
```

Expected planning output from the frozen minimum collection blueprint:

- 80 clean Japanese recording assignments;
- 18 pseudonymous speakers;
- 6 development identities and 12 held identities;
- 44 short-target assignments and 36 long-target assignments;
- 44 spontaneous and 36 controlled-dialogue assignments.

Do not replace the generated pseudonymous speaker ids with names in the research manifest.

## 2. Fill only real participant metadata

Edit the private `speaker_metadata_private.csv` after recruitment/recording.

Useful fields include:

- `l1`;
- `japanese_proficiency`;
- `recording_device`;
- `consent_or_dataset_provenance`.

Do not infer or fabricate these fields when they are unknown.

## 3. Record WAV files at the generated relative paths

For example:

```text
held/learner/HL01/HL01_01.wav
held/learner/HL01/HL01_02.wav
...
held/native/HN01/HN01_01.wav
```

The prompt asks for a naturally short or naturally longer response. Do not force a participant to hit an exact stopwatch duration. The v10 short/long bucket is assigned later from the product endpointing result.

The target response transcript is not required for scoring and should not be placed in the listener pack.

## 4. Materialize the real manifest incrementally

```bash
python scripts/materialize_free_speech_manifest_v10.py \
  outputs/free_speech_v10_collection/assignments.csv \
  --audio-root /path/to/free_speech_audio \
  --speaker-metadata outputs/free_speech_v10_collection/speaker_metadata_private.csv \
  --out outputs/free_speech_v10_collection/free_speech_manifest.csv
```

This emits only recordings that actually exist and re-runs `free_speech_sample_manifest_v1` validation.

For the final frozen run, require every planned recording:

```bash
python scripts/materialize_free_speech_manifest_v10.py \
  outputs/free_speech_v10_collection/assignments.csv \
  --audio-root /path/to/free_speech_audio \
  --speaker-metadata outputs/free_speech_v10_collection/speaker_metadata_private.csv \
  --out outputs/free_speech_v10_collection/free_speech_manifest.csv \
  --require-all
```

## 5. Run machine acceptance and create the blinded listener pack

```bash
python scripts/run_free_speech_promotion_readiness_v10.py \
  outputs/free_speech_v10_collection/free_speech_manifest.csv \
  --audio-root /path/to/free_speech_audio \
  --out-dir outputs/free_speech_v10_readiness \
  --raters r01,r02,r03,r04,r05
```

The scoring path uses `transcript=None`.

Before human ratings exist, the expected terminal state is:

```text
awaiting_human_ratings
```

That is a successful boundary condition, not a failed experiment.

The private asset map contains source paths. The listener CSV does not expose speaker group, L1, split, channel label, target-response transcript, or machine scores.

## 6. Collect blinded human ratings

The isolated presentation covers the four public constructs:

- `clarity_comprehensibility`;
- `fluency`;
- `rhythm_naturalness`;
- `intonation_utterance_naturalness`.

Controlled-dialogue clips may additionally receive a separate contextual-intonation presentation. Contextual appropriateness is not substituted for utterance-level intonation.

The target is five ratings per held presentation; the frozen minimum for held inclusion remains three.

## 7. Run promotion readiness after ratings

```bash
python scripts/run_free_speech_promotion_readiness_v10.py \
  outputs/free_speech_v10_collection/free_speech_manifest.csv \
  --audio-root /path/to/free_speech_audio \
  --out-dir outputs/free_speech_v10_readiness \
  --ratings-csv /path/to/completed_listener_ratings.csv
```

The runner normalizes blinded ratings, runs the frozen v5 scientific gates, then applies the additive v10 consumer-discrimination gates.

Interpretation:

- `pass`: that individual candidate may enter a **separate** score-changing A/B branch with a new ScoreContract;
- `fail`: keep it shadow;
- `insufficient`: collect more criterion evidence; do not tune held thresholds to force a pass.

## 8. What historical data may still be reused

Use `audit_historical_free_speech_reuse_v10.py` before attempting to convert old assets into the new manifest.

Current project boundaries are intentionally conservative:

- JVS read speech: native fixed-reading/broad-mode regression only;
- JANON: real learner/native fixed-reading regression only;
- synthetic JVS edits: engineering robustness only;
- historical TTS/noise controls: routing controls if audio is still recoverable;
- none of those assets should be relabeled as held learner free-speech criterion data.

## 9. What not to do

Do not:

- use gold/manual response transcripts as ProductScore input;
- let development and held share speaker identities;
- count channel variants as independent human productions;
- call synthetic perturbations learner errors;
- interpret native control separation as the definition of pronunciation quality;
- promote a candidate because overall native+learner correlation looks good while learner scores remain compressed;
- change the four public score semantics inside this validation branch.
