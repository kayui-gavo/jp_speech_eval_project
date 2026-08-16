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
