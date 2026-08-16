# Free-Speech Promotion Readiness v10

## Status

Branch: `free-speech-promotion-readiness-v10`

Base: `score-evidence-reliability-v9`

Draft PR: #10

Latest validated head: `550f04eae975124044c3f8a70d25f95103235b83`

Permanent PR-context `baseline-evolution-light-tests` run `31968169566` completed successfully on that head.

This round does not change ProductScore. It strengthens the evidence required before any direct free-speech component may replace a neutral prior or otherwise change the score contract, and it provides the practical collection/validation path needed to obtain that evidence.

## Why v5 was not enough for a consumer product

The frozen v5 promotion protocol already protects construct validity, held-set separation, channel robustness, negative-control routing, version consistency, F0 missingness, and speaker/task stability.

A C-end failure mode nevertheless remained possible: a candidate could achieve a good overall native+learner correlation and a wide overall score range mainly by separating native and learner groups, while giving learner users a narrow cluster of nearly identical scores.

A second failure mode also remained possible: native clips could rescue a weak learner-only short/long or task slice. That is unacceptable for a learner-facing product.

v10 therefore layers consumer-discrimination gates on top of v5 rather than changing the frozen v5 protocol.

## Frozen additive v10 protocol v2

Protocol: `data/research_eval/free_speech_v10_consumer_promotion_protocol.json`

Schema: `free_speech_v10_consumer_promotion_protocol_v2`

The protocol was tightened before any v10 human criterion results existed. It must not be retuned on held ratings to force a pass.

For each direct ProductScore candidate, v10 additionally requires:

- at least 30 held learner construct-matched pairs;
- held learner-only Spearman rho >= 0.25;
- held learner candidate IQR >= 6 score points;
- at least 8 held learner short-utterance pairs;
- at least 8 held learner long-utterance pairs;
- positive learner association direction in both short and long slices once coverage is sufficient;
- at least 10 held learner spontaneous pairs;
- at least 10 held learner controlled-dialogue pairs;
- positive learner association direction in both task slices once coverage is sufficient;
- native-vs-learner behavior that does not contradict the human construct criterion.

Current operational duration slices are:

- short: actual product-condition speech duration <= 3.0 s;
- long: actual product-condition speech duration >= 8.0 s.

The duration bucket uses no gold/manual transcript. It is an operational C-end robustness slice, not a linguistic definition of sentence length.

## Native controls are a construct-contamination check

v10 does not hard-code that native samples must outrank learner samples.

For each construct:

1. if the human native-minus-learner gap is at least 0.25 points on the 1-7 criterion scale, the machine candidate must follow the same direction;
2. if the human group gap is smaller than 0.25, the machine must not invent a large native/learner separation. The current frozen maximum machine gap in that regime is 6 score points.

This catches a particularly dangerous failure mode: a system that claims to score clarity, rhythm, fluency, or intonation but is actually functioning as a hidden native-likeness detector.

## Learner-only length and task robustness

Short/long and spontaneous/controlled-dialogue gates are calculated inside the held learner population.

Native samples cannot rescue:

- missing learner long-speech coverage;
- a negative learner long-speech association;
- missing learner controlled-dialogue coverage;
- a task-specific learner failure.

This is deliberately stricter than an overall corpus correlation because the actual user population is L2 Japanese learners.

## Evidence export

`export_free_speech_v10_evidence.py` reuses the v5 candidate definitions and v5 availability semantics unchanged. It only adds product-condition `speech_duration_sec`.

Therefore:

- neutral placeholders still do not count as available evidence;
- missing F0 is still not a low intonation observation;
- no gold transcript is used for duration slicing;
- v10 does not create a new candidate score formula.

## Machine coverage preflight before human-rating spend

`assess_free_speech_machine_coverage_v10.py` runs after real product-condition acceptance and before the final listener-rating freeze.

It checks actual held learner coverage rather than trusting the planned prompt bucket:

- total held learner samples;
- actual endpointed short clips;
- actual endpointed long clips;
- spontaneous clips;
- controlled-dialogue clips;
- per-candidate evidence availability in those slices.

It uses no human ratings and makes no validity claim.

The main runner now has three meaningful terminal stages:

- `collection_coverage_insufficient`: actual learner coverage is not yet sufficient; collect/replace recordings before spending full rating effort;
- `awaiting_human_ratings`: machine-side collection structure is ready and blinded held ratings are the next boundary;
- `promotion_readiness_evaluated`: completed ratings were normalized and frozen v5 + v10 gates were executed.

This prevents an avoidable failure where five raters score a large pack only to discover afterwards that too few learner long responses actually exceeded 8 seconds.

## Historical-data reuse audit

`audit_historical_free_speech_reuse_v10.py` makes the old-data boundary explicit instead of relying on project memory.

The current repository contains useful historical assets, but they must not be silently relabeled as free-speech criterion data:

- `JVS_parallel100` contains scripted native read speech. Historical execution through `transcript_assisted_light` does not change the elicitation construct. These remain native fixed-reading/broad-mode regression assets.
- `JANON` contains real native and learner productions, but the inventory entries used here are isolated fixed-reading words. They remain useful for pronunciation/timing regression, not spontaneous fluency or long-utterance intonation validation.
- `JVS_controlled_edit` rows are synthetic edits of real recordings. They can test engineering robustness but must not be called learner errors or independent human productions.
- the historical demo recording lacks criterion-grade speaker/task/split provenance and remains a smoke-test asset.
- historical English/Mandarin/noise controls are routing assets only and must be recovered/regenerated when their old temporary files no longer exist.

The legacy `data/human_eval/learner_recording_needed.csv` remains a fixed-word acquisition plan. It should not be reused as the primary v10 collection because it does not cover spontaneous fluency or long-utterance rhythm/intonation.

Permanent tests enforce that broad-mode execution does not magically turn JVS read speech into free speech and that JANON fixed words remain fixed-reading regression data.

## Minimum practical new collection

Operational blueprint: `data/human_eval/free_speech_v10_minimum_collection_plan.json`

This is an engineering acquisition plan with buffer above the frozen gates, not a formal psychometric power calculation.

### Held Japanese core

- 8 learner speakers x 5 clean free-speech clips = 40 learner clips;
- 4 native speakers x 4 clean free-speech clips = 16 native clips;
- total = 56 clean held Japanese clips.

The held learner mix includes short spontaneous, short controlled-dialogue, long spontaneous, and long controlled-dialogue speech. Forty learner clips provide buffer above the frozen minimum of 30 construct-matched learner pairs.

### Development Japanese core

- 4 learner speakers x 4 clips = 16;
- 2 native speakers x 4 clips = 8;
- total = 24 clean development clips.

Development identities never appear in held.

### Collection assignment generator

`prepare_free_speech_collection_v10.py` deterministically creates:

- 80 recording assignments;
- 18 pseudonymous speaker ids;
- 6 development identities;
- 12 held identities;
- 44 planned short responses;
- 36 planned long responses;
- 44 spontaneous responses;
- 36 controlled-dialogue responses.

It does not invent L1, proficiency, device, or consent metadata and never asks for a target response transcript.

### Local browser recorder

`free_speech_collection_server_v10.py` removes manual filename handling during recording.

It is local-only by default (`127.0.0.1`) and:

- asks the participant only for a pseudonymous speaker id;
- walks through that speaker's pending prompts;
- does not expose held/development, learner/native, L1, machine scores, or filesystem paths;
- requests mono capture with echo cancellation, noise suppression, and AGC disabled when the browser honors those constraints;
- encodes mono PCM WAV in-browser;
- accepts writes only for predeclared sample ids and `.wav` paths;
- rejects path traversal;
- refuses silent overwrite;
- performs atomic writes.

It is a research collection utility and is not part of the public Hugging Face Space.

### Manifest materializer

`materialize_free_speech_manifest_v10.py` converts actual collected files into the existing `free_speech_sample_manifest_v1` contract.

By default it emits only WAV files that actually exist. `--require-all` provides a final fail-closed collection freeze.

It joins only real private speaker metadata, adds no gold transcript, preserves speaker split, and reruns the existing manifest validator.

## Channel and negative controls

The practical plan keeps channel and language-routing evidence separate from four-dimension criterion validity:

- channel variants must be derived from the exact same clean source recording and preserve `source_recording_id`;
- re-speaking an utterance is not a valid channel pair;
- final negative controls should include real English and real Mandarin speech rather than rely only on TTS;
- non-speech/environmental controls remain routing tests, not four-dimension listener-rating material.

## Blinded listener ratings

The isolated held presentation covers the four public constructs:

- `clarity_comprehensibility`;
- `fluency`;
- `rhythm_naturalness`;
- `intonation_utterance_naturalness`.

Controlled-dialogue clips may additionally receive a separate contextual-intonation presentation. Contextual appropriateness is not substituted for utterance-level intonation.

The target is five ratings per held presentation; the frozen minimum for held inclusion remains three.

The listener pack does not expose speaker group, L1, split, channel label, target-response transcript, or machine scores.

## End-to-end runner

`run_free_speech_promotion_readiness_v10.py` now provides one path from a validated real-audio manifest to the promotion decision boundary:

1. validate manifest;
2. run real product-condition acceptance with `transcript=None`;
3. export v10 machine evidence;
4. preflight actual learner short/long/task coverage;
5. optionally build the held-only blinded listener pack;
6. normalize completed ratings when supplied;
7. run frozen v5 scientific gates;
8. run frozen v10 consumer-discrimination gates.

Operational commands are documented in `reports/FREE_SPEECH_V10_OPERATIONS.md`.

## Promotion semantics

A v10 `pass` does not mutate production scoring. It means that individual candidate may enter a separate score-changing A/B branch, where a new ScoreContract version is required.

A `fail` remains shadow.

An `insufficient` result means more held criterion evidence is required. It is not permission to tune held thresholds.

A candidate may therefore progress independently. For example, clarity may eventually pass while intonation remains shadow if long-utterance F0 evidence is still insufficient.

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
- history comparability;
- the public HF Space scoring behavior.

No Codex was required for this round. Nothing in v10 has been merged or deployed.

The next actual score-changing branch should only be created for an individual candidate after it passes both v5 and v10 on held real audio plus construct-matched human criteria.
