# Free-Speech Promotion Readiness v10

## Status

Branch: `free-speech-promotion-readiness-v10`

Base: `score-evidence-reliability-v9`

This round does not change ProductScore. It strengthens the evidence required before any direct free-speech component may replace a neutral prior or otherwise change the score contract.

## Why v5 was not enough for a consumer product

The frozen v5 promotion protocol already protects construct validity, held-set separation, channel robustness, negative-control routing, version consistency, F0 missingness, and speaker/task stability.

One C-end failure mode remained possible: a candidate could achieve a good overall native+learner correlation and a wide overall score range mainly by separating native and learner groups, while giving learner users a narrow cluster of nearly identical scores.

That would be scientifically descriptive but commercially poor: the score would have little discrimination where the product needs it most.

v10 therefore layers consumer-discrimination gates on top of v5 rather than changing the frozen v5 protocol.

## Frozen additive v10 gates

Protocol: `data/research_eval/free_speech_v10_consumer_promotion_protocol.json`

For each direct ProductScore candidate, v10 additionally requires:

- at least 30 held learner construct-matched pairs;
- held learner-only Spearman rho >= 0.25;
- held learner candidate IQR >= 6 score points;
- at least 8 short-utterance criterion pairs;
- at least 8 long-utterance criterion pairs;
- positive association direction in both short and long groups once coverage is sufficient;
- native-vs-learner candidate direction consistent with the actual human criterion direction when the human group gap is large enough to be interpretable.

These thresholds are frozen before v10 criterion results. They must not be tuned on held data to force promotion.

## Short and long utterances

The v10 evidence exporter adds `speech_duration_sec` from the actual product-condition endpointing result.

Current frozen operational buckets are:

- short: speech duration <= 3.0 s;
- long: speech duration >= 8.0 s.

The duration bucket uses no gold/manual transcript. It is an operational C-end robustness slice, not a linguistic definition of sentence length.

## Native controls are not a native-likeness target

v10 deliberately does not hard-code that every native sample must outrank every learner sample.

For each construct, the analyzer first measures the native-minus-learner direction in the human criterion. Only when that human gap is at least 0.25 points on the 1-7 scale does the group-direction gate become interpretable. The machine candidate must then have the same direction.

This keeps native speech as a false-alarm/safety control without silently redefining clarity, fluency, rhythm, or intonation as native-likeness.

## Evidence export

`export_free_speech_v10_evidence.py` reuses the v5 candidate definitions and v5 availability semantics unchanged. It only adds product-condition speech duration.

Therefore:

- neutral placeholders still do not count as available evidence;
- missing F0 is still not a low intonation observation;
- no gold transcript is used for duration slicing;
- v10 does not create a new candidate score formula.

## Historical-data reuse audit

`audit_historical_free_speech_reuse_v10.py` makes the old-data boundary explicit instead of relying on project memory.

The current repository contains useful historical assets, but they must not be silently relabeled as free-speech criterion data:

- `JVS_parallel100` contains scripted native read speech. Two historical samples were also executed through `transcript_assisted_light`, but changing the evaluator mode does not change the original elicitation construct. These remain native fixed-reading/broad-mode regression assets.
- `JANON` contains real native and learner productions, but the inventory entries used here are isolated fixed-reading words. They remain valuable for pronunciation/timing regression, not spontaneous fluency or long-utterance intonation validation.
- `JVS_controlled_edit` rows are synthetic edits of real recordings. They can test engineering robustness but must not be called learner errors or independent human productions.
- the historical demo recording lacks criterion-grade speaker/task/split provenance and remains a smoke-test asset.
- historical English/Mandarin/noise controls are useful for routing logic, but the saved inventory points to temporary paths; they must be recovered or regenerated before a fresh acceptance run.

The legacy `data/human_eval/learner_recording_needed.csv` also remains a fixed-word acquisition plan. It should not be reused as the main v10 collection plan because it does not cover spontaneous fluency or long-utterance rhythm/intonation.

This audit is intentionally conservative: a historical sample is not promoted to free-speech criterion eligibility merely because it was once evaluated in a broad mode.

## Minimum practical new collection

Operational blueprint: `data/human_eval/free_speech_v10_minimum_collection_plan.json`

The plan is derived from the already-frozen v5/v10 engineering gates with buffer; it is not presented as a formal psychometric power calculation.

### Held Japanese core

- 8 learner speakers x 5 clean free-speech clips = 40 learner clips;
- 4 native speakers x 4 clean free-speech clips = 16 native clips;
- total = 56 clean held Japanese clips.

The learner mix intentionally includes short spontaneous, short controlled-dialogue, long spontaneous, and long controlled-dialogue speech. The 40 learner clips provide buffer above the v10 minimum of 30 held learner pairs.

### Development core

- 4 learner speakers x 4 clips = 16;
- 2 native speakers x 4 clips = 8;
- total = 24 clean development clips.

Development identities never appear in held.

### Duration collection policy

Participants are given natural response instructions rather than asked to hit an exact stopwatch duration. The actual product endpointing result determines the v10 bucket after recording:

- <= 3 s: short slice;
- >= 8 s: long slice;
- 3-8 s: still valid for overall criterion analysis, but not counted toward either duration-specific gate.

The plan deliberately oversamples short and long prompts so ordinary variation in response length does not leave the held set below coverage requirements.

### Channel and negative controls

- derive channel variants from the exact same clean source recording and preserve `source_recording_id`;
- do not create a channel pair by asking a person to re-speak the utterance;
- include real English and real Mandarin speech in the final negative-control set rather than relying only on TTS;
- non-speech/environmental controls remain routing tests, not four-dimension listener-rating material.

## End-to-end runner

`run_free_speech_promotion_readiness_v10.py` provides one execution path:

1. validate the private sample manifest;
2. run the real product-condition free-speech acceptance path with `transcript=None`;
3. export v10 machine evidence;
4. optionally create a held-only blinded listener pack;
5. when completed listener ratings are supplied, normalize ratings and run frozen v5 plus v10 promotion gates.

Example before ratings:

```bash
python scripts/run_free_speech_promotion_readiness_v10.py \
  data/private/free_speech_manifest.csv \
  --audio-root /path/to/audio \
  --out-dir outputs/free_speech_v10 \
  --raters r01,r02,r03,r04,r05
```

This validly ends in:

`awaiting_human_ratings`

After blinded ratings are collected:

```bash
python scripts/run_free_speech_promotion_readiness_v10.py \
  data/private/free_speech_manifest.csv \
  --audio-root /path/to/audio \
  --out-dir outputs/free_speech_v10 \
  --ratings-csv outputs/free_speech_v10/completed_listener_ratings.csv
```

## Promotion semantics

A v10 `pass` does not directly mutate production scoring. It means the candidate may enter a separate score-changing A/B branch, where a new ScoreContract version is required.

A `fail` remains shadow.

An `insufficient` result means more held criterion data are required. It is not permission to tune thresholds on held data.

## Explicit non-changes

v10 does not change:

- `consumer_four_score_v2`;
- current ProductScore weights;
- display transform;
- fixed-reference scoring;
- free-speech neutral priors;
- v5 construct definitions;
- human rating scale;
- public four-dimension semantics;
- history comparability.

The next actual score-changing branch should only be created for an individual candidate after it passes both v5 and v10.
