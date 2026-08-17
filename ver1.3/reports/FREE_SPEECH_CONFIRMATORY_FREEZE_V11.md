# Free-speech confirmatory freeze v11

## Status

Engineering freeze layer for the v10 direct free-speech promotion-readiness workflow.

This branch does not change ProductScore, ScoreContract, candidate definitions, promotion thresholds, public UI, or deployment.

## Why this exists

Once held real audio and listener ratings exist, the main risk is no longer missing another heuristic. It is accidental confirmatory contamination:

- a learner/native identity appears in development and held data;
- the same underlying source recording appears in both splits under different rows;
- identical audio bytes appear in both splits under different filenames or ids;
- a held WAV is replaced after listener work starts;
- the promotion protocol is edited after seeing held results;
- a runner is invoked against a manifest or protocol different from the intended frozen set.

The existing manifest validator already catches the first class at metadata level. v11 adds byte-level and execution-time locking.

## Freeze artifact

`free_speech_confirmatory_freeze_v11.py freeze` records:

- raw manifest SHA256;
- canonical row SHA256;
- row count;
- SHA256 and byte size for every audio file;
- split, speaker id and source recording id attached to each audio digest;
- SHA256 and byte size for the frozen v5 and v10 promotion protocol files;
- a single freeze fingerprint covering all of the above.

A freeze is not valid when any audio is missing, one source recording crosses development/held, or identical audio bytes occur across development/held.

## Verification

`free_speech_confirmatory_freeze_v11.py verify` reconstructs the freeze from current files and compares the fingerprint. Any manifest, protocol, or audio-byte change causes `verification_ok=false`.

This is intentionally stricter than checking Git commit history because the collected audio is expected to live outside the repository.

## Confirmatory runner

`run_free_speech_confirmatory_v11.py` verifies the freeze before calling the existing v10 readiness runner.

It obtains the following only from the freeze artifact:

- manifest path;
- audio root;
- v5 promotion protocol;
- v10 consumer promotion protocol.

A verification failure aborts before v10 evaluation is called.

The run summary records the freeze fingerprint and repeats the scientific lock:

- held set is not for threshold tuning;
- protocol retuning is not allowed;
- ProductScore is unchanged;
- ScoreContract is unchanged.

## Intended operational sequence

1. Finish development work and collect/materialize the v10 development + held manifest.
2. Run v10 machine coverage preflight until the frozen structural requirements are met.
3. Freeze the final manifest, audio bytes and both promotion protocols.
4. Keep the freeze JSON with the experiment artifacts.
5. Build/collect blinded listener ratings.
6. Run the confirmatory workflow through `run_free_speech_confirmatory_v11.py`.
7. If an individual candidate passes, create a separate score-changing A/B branch with a new ScoreContract. Do not edit the v11 held gate to make it pass.

## Example commands

```bash
cd ver1.3

python scripts/free_speech_confirmatory_freeze_v11.py freeze \
  /path/to/final_manifest.csv \
  --audio-root /path/to/audio \
  --protocol data/research_eval/free_speech_v5_promotion_protocol.json \
  --protocol data/research_eval/free_speech_v10_consumer_promotion_protocol.json \
  --out /path/to/experiment/free_speech_confirmatory_freeze_v11.json

python scripts/free_speech_confirmatory_freeze_v11.py verify \
  /path/to/experiment/free_speech_confirmatory_freeze_v11.json

python scripts/run_free_speech_confirmatory_v11.py \
  /path/to/experiment/free_speech_confirmatory_freeze_v11.json \
  --out-dir /path/to/experiment/confirmatory_run \
  --ratings-csv /path/to/completed_blinded_ratings.csv
```

## Tests

Permanent tests cover:

- reproducible freeze + verification;
- WAV byte drift;
- promotion-protocol drift;
- identical audio bytes across development/held;
- source-recording id leakage across development/held;
- confirmatory runner aborting before v10 on failed verification;
- runner using only frozen manifest/audio/protocol paths.

## Scientific boundary

A clean v11 confirmatory run still does not automatically alter production scoring. It only makes the resulting per-candidate promotion decision auditable and reproducible. Any score-changing experiment remains a separate branch and requires an explicit ScoreContract version change.
