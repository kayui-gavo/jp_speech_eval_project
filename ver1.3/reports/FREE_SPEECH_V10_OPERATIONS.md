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

## 3. Record the assigned WAV files

The recommended path is the local browser collector:

```bash
python scripts/free_speech_collection_server_v10.py \
  outputs/free_speech_v10_collection/assignments.csv \
  --audio-root /path/to/free_speech_audio
```

Then open:

```text
http://127.0.0.1:8765
```

The participant enters only the pseudonymous speaker id such as `HL01`. The page then walks through that speaker's pending prompts and writes mono PCM WAV files directly to the assignment paths.

The collector deliberately does not show:

- held/development split;
- learner/native analysis label;
- L1 or proficiency metadata;
- machine scores;
- filesystem paths;
- target-response transcripts.

The browser requests raw-ish mono capture with echo cancellation, noise suppression, and automatic gain control disabled when the browser/device honors those constraints. It encodes PCM WAV in the browser instead of relying on browser-specific MediaRecorder codecs.

The server binds to `127.0.0.1` by default, only accepts predeclared sample ids and `.wav` paths, rejects path traversal, and refuses silent overwrite of an existing recording. It is a local research utility and must not be deployed as the public Hugging Face Space.

If recording is performed with another tool, preserve the generated relative paths exactly, for example:

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

## 5. Run machine acceptance and actual-coverage preflight

```bash
python scripts/run_free_speech_promotion_readiness_v10.py \
  outputs/free_speech_v10_collection/free_speech_manifest.csv \
  --audio-root /path/to/free_speech_audio \
  --out-dir outputs/free_speech_v10_readiness
```

The scoring path uses `transcript=None`.

The runner first derives actual product-condition `speech_duration_sec` and checks held learner coverage against the frozen v10 structure gates.

Possible machine-only stages are:

```text
collection_coverage_insufficient
```

or:

```text
awaiting_human_ratings
```

`collection_coverage_insufficient` means the real endpointed learner set still lacks enough short/long or task coverage. Replace or add recordings before freezing the final listener pack. Native clips cannot rescue a missing learner slice.

`awaiting_human_ratings` means the collection structure is ready for the criterion stage. It is not a promotion result.

The detailed preflight is written to:

```text
outputs/free_speech_v10_readiness/machine_coverage_preflight_v10.json
```

## 6. Build the final blinded listener pack

Once machine coverage is ready:

```bash
python scripts/run_free_speech_promotion_readiness_v10.py \
  outputs/free_speech_v10_collection/free_speech_manifest.csv \
  --audio-root /path/to/free_speech_audio \
  --out-dir outputs/free_speech_v10_readiness \
  --raters r01,r02,r03,r04,r05
```

The private asset map contains source paths. The listener CSV does not expose speaker group, L1, split, channel label, target-response transcript, or machine scores.

If a pilot listener pack is generated while machine coverage is still insufficient, the runner marks it provisional rather than final-freeze-ready.

## 7. Collect blinded human ratings with the local listener UI

The recommended path is:

```bash
python scripts/free_speech_listener_server_v10.py serve \
  outputs/free_speech_v10_readiness/listener_pack_held_v10.csv \
  outputs/free_speech_v10_readiness/private_asset_map_held_v10.csv \
  --audio-root /path/to/free_speech_audio \
  --responses-dir outputs/free_speech_v10_readiness/listener_responses
```

Then open:

```text
http://127.0.0.1:8766
```

Each listener enters only an anonymized rater id such as `r01`. The server shows only that rater's assigned presentations.

The UI reads the construct instructions and 1-7 scale directly from `consumer_rating_schema_v3.json`; it does not maintain a second copy of the human criterion wording.

The isolated presentation covers:

- `clarity_comprehensibility`;
- `fluency`;
- `rhythm_naturalness`;
- `intonation_utterance_naturalness`.

Controlled-dialogue clips may additionally receive a separate contextual-intonation presentation. Contextual appropriateness is not substituted for utterance-level intonation.

The listener UI does not expose sample ids, source paths, speaker group, L1, split, channel labels, response transcripts, or machine scores. Source audio paths stay inside the private asset map on the server.

Each response is atomically stored as a separate JSON document keyed by presentation id. Closing/reopening the browser therefore does not erase previously completed presentations, and multiple local raters do not write a shared CSV row concurrently.

If a presentation is marked unanalyzable, the server rejects construct scores for it. If it is analyzable, exactly the assigned constructs must receive integer ratings from 1 to 7.

The target is five ratings per held presentation; the frozen minimum for held inclusion remains three.

## 8. Export listener responses back to the existing v3 schema

After rating:

```bash
python scripts/free_speech_listener_server_v10.py export \
  outputs/free_speech_v10_readiness/listener_pack_held_v10.csv \
  outputs/free_speech_v10_readiness/private_asset_map_held_v10.csv \
  --audio-root /path/to/free_speech_audio \
  --responses-dir outputs/free_speech_v10_readiness/listener_responses \
  --out outputs/free_speech_v10_readiness/completed_listener_ratings.csv \
  --require-complete
```

The exporter reconstructs the original v3 listener rows, fills only the assigned ratings, and runs the existing `validate_consumer_ratings_v3.py` contract before accepting the output.

Without `--require-complete`, a partial CSV may be exported for progress inspection, but missing presentations are reported and never fabricated.

## 9. Run promotion readiness after ratings

```bash
python scripts/run_free_speech_promotion_readiness_v10.py \
  outputs/free_speech_v10_collection/free_speech_manifest.csv \
  --audio-root /path/to/free_speech_audio \
  --out-dir outputs/free_speech_v10_readiness \
  --ratings-csv outputs/free_speech_v10_readiness/completed_listener_ratings.csv
```

The runner normalizes blinded ratings, runs the frozen v5 scientific gates, then applies the additive v10 consumer-discrimination gates.

Interpretation:

- `pass`: that individual candidate may enter a **separate** score-changing A/B branch with a new ScoreContract;
- `fail`: keep it shadow;
- `insufficient`: collect more criterion evidence; do not tune held thresholds to force a pass.

## 10. What historical and external data may still be reused

Use `audit_historical_free_speech_reuse_v10.py` before attempting to convert old project assets into the new manifest.

Current project boundaries are intentionally conservative:

- JVS read speech: native fixed-reading/broad-mode regression only;
- JANON: real learner/native fixed-reading regression only;
- synthetic JVS edits: engineering robustness only;
- historical TTS/noise controls: routing controls if audio is still recoverable;
- none of those assets should be relabeled as held learner free-speech criterion data.

For external corpora, see `reports/FREE_SPEECH_EXTERNAL_DATA_POLICY_V10.md`. In particular, a scientifically relevant research corpus is not automatically authorized to calibrate or validate a commercial C-end ProductScore.

## 11. What not to do

Do not:

- use gold/manual response transcripts as ProductScore input;
- let development and held share speaker identities;
- count channel variants as independent human productions;
- call synthetic perturbations learner errors;
- interpret native control separation as the definition of pronunciation quality;
- let native samples rescue a failing learner short/long or task slice;
- promote a candidate because overall native+learner correlation looks good while learner scores remain compressed;
- spend the full listener-rating budget before actual endpointed learner coverage is checked;
- change the four public score semantics inside this validation branch.
