# Scoring Integrity v8

## Status

Branch: `scoring-integrity-v8`

Base: `hf-space-karaoke-integration-v7`

This round is a scoring-integrity repair after re-auditing the complete C-end path from input eligibility through ASR, fixed-reference alignment, four public dimensions, reliability, karaoke visualization, shadow research evidence, and CI.

The central finding is that the project’s evidence/reliability safeguards are now more mature than several underlying score measurements. v8 therefore fixes places where evidence provenance or routing could produce a false claim, while deliberately **not** pretending that the free-speech four-dimensional score is already solved.

The public score contract remains:

- `score_contract_version = consumer_four_score_v2`
- clarity .30 / rhythm .25 / fluency .25 / intonation .20
- the same display transform
- the same official ProductScore values for otherwise equivalent scoring evidence

The evidence schema is bumped to:

- `evidence_schema_version = consumer_evidence_v4`

because reference-boundary provenance now changes whether target-local evidence is eligible for public interpretation.

---

## 1. P0 repair: direct free speech no longer inherits pseudo-reference length limits

### Previous problem

`transcript_assisted_light` reused `check_asr_transcript_sanity()`. That gate was originally designed to decide whether an ASR transcript was suitable for generating a pseudo-reference, and therefore contained limits such as a minimum of three content characters and a maximum of eighty.

Those are not valid eligibility rules for direct conversation.

A natural Japanese turn such as:

- `はい`
- `え？`

could be rejected for being too short, while a natural longer answer could be rejected merely because it exceeded a realtime pseudo-reference synthesis limit.

### New contract

Direct free speech now uses `check_free_speech_transcript_sanity()`.

It asks only whether the transcript is usable evidence for direct Japanese speech assessment:

- non-empty content;
- predominantly Japanese script after the independent language gate;
- no obvious repetition/noise-like ASR hallucination pattern.

It deliberately has no pseudo-reference 3–80 character rule.

Utterance length can still make a particular dimension weak or unavailable. That is handled later as evidence sufficiency/reliability, not by falsely declaring the Japanese turn ineligible.

Pseudo-reference-specific constraints remain separate from direct free-speech scoreability.

The intended separation is now:

```text
language eligibility
    != direct-free-speech scoreability
    != pseudo-reference eligibility
```

---

## 2. P0 repair: reference-boundary provenance is now consumed, not merely recorded

### Previous problem

Earlier work correctly added explicit fixed-reference boundary provenance:

- `ref_boundary_method`
- `ref_boundary_confidence`
- `ref_boundary_tier`
- `ref_boundary_source`

For example, an equal-mora reference cache can correctly record a low-confidence `equal_fallback` boundary prior.

However, downstream cached DTW could map those approximate reference boundaries into learner time, and later consumers could reason mainly from the DTW path/evidence health. This created a scientific loophole: a good acoustic path could make target-local evidence look more precise than the boundary source it inherited from.

### New contract

`details.alignment` now carries the reference-boundary provenance all the way to product consumers.

Target-local precision is limited when either:

- the learner alignment itself fell back; or
- the reference boundary tier is low precision (`equal_fallback` / `unknown_alignment`); or
- reference-boundary confidence is below the conservative local-evidence threshold.

This affects **local evidence eligibility**, not learner ability.

When reference timing is approximate:

- the broad/global score remains available;
- exact target-local timing evidence is downgraded;
- strict pitch feedback is blocked;
- special-mora learner feedback is blocked;
- local pronunciation detail is blocked;
- karaoke mora synchronization is labelled approximate.

This implements the standing product rule:

```text
measurement uncertainty -> lower confidence/detail
measurement uncertainty != learner performance penalty
```

---

## 3. Karaoke synchronization now uses effective local precision

Previously the karaoke layer could expose a mora alignment based mainly on the learner-side alignment confidence.

v8 defines effective local confidence conservatively from both sides of the mapping. When reference-boundary confidence is available, the local confidence cannot exceed it.

Mora timeline rows now preserve separate provenance fields including:

- effective `alignment_confidence`
- `path_alignment_confidence`
- `reference_boundary_confidence`
- `reference_boundary_tier`
- `approximate`

A low-precision reference therefore cannot produce visually overconfident mora-level synchronization merely because the DTW path itself is healthy.

This remains a playback-alignment visualization, not phone correctness.

---

## 4. Special-mora learner feedback is now opt-in rather than default-on

The limited special-mora candidate has useful engineering safeguards, but its current calibration still has an important scientific limitation:

- native false-alarm calibration does not prove learner error detection or learner benefit;
- synthetic feature shortening is not a substitute for real learner-labelled validation.

Accordingly, `render_user_facing_result()` now defaults:

```text
enable_user_facing_calibrated_special_mora = false
```

Runtime/shadow evidence remains available for analysis.

Tests and explicit experimental scenarios may opt in to the limited candidate deliberately. Ordinary product rendering no longer opts users into that candidate implicitly.

---

## 5. Optional SSL rhythm evidence no longer invalidates SSL pronunciation evidence

A v7 full-suite control exposed a pre-existing shadow-path coupling issue.

A short SSL feature sequence could produce a valid pronunciation/reference cosine-DTW distance while the derivative `rhythm_dtw_v1` metric failed because its smoothing window required more frames. The exception could make the whole SSL shadow look unavailable.

v8 isolates the constructs:

- valid SSL pronunciation distance is retained;
- insufficient rhythm frames make only `rhythm_dtw_v1` unavailable;
- reason: `insufficient_dtw_frames_for_rhythm_metric`.

One optional sub-evidence failure therefore no longer contaminates another successful research measurement.

---

## 6. Public demo no longer computes hidden realtime debug rows

The established Hugging Face Space remains based on `debug_ui/index.html`.

Public mode already hides research/debug panels, but fixed-reference evaluation still computed realtime replay rows that the public UI did not show.

v8 skips that hidden computation in public-demo mode while preserving it for local/debug use.

This is a latency/CPU cleanup only; it does not change scoring semantics.

---

## 7. The major unresolved score problem remains: free speech still contains neutral 70 priors

The re-audit confirmed that the official free-speech ProductScore still behaves approximately as:

- clarity: neutral prior 70 when no promoted evidence exists;
- rhythm: neutral prior 70 when no promoted evidence exists;
- intonation: neutral prior 70 when no promoted evidence exists;
- fluency: the primary moving product component.

This is scientifically honest at the per-dimension evidence layer because `neutral_prior` is explicitly labelled as such, but it still compresses headline-score dispersion because the neutral placeholders retain their normal ProductScore weights.

v8 deliberately **does not solve this by promoting unvalidated v4 shadows**.

ASR recoverability is not automatically human clarity/comprehensibility. ASR word timing is not automatically rhythm naturalness. Global F0 movement is not automatically context-appropriate intonation.

The official `/100` formula is therefore unchanged in this branch.

---

## 8. New shadow candidate: neutral priors do not count as measured aggregate evidence

New module:

`src/jp_speech_eval/partial_evidence_aggregate.py`

Schema:

`partial_evidence_aggregate_shadow_v1`

Policy:

`neutral_prior_excluded_coverage_shrunk_v1`

This candidate tests an aggregation question only. It does **not** promote any new acoustic model.

It consumes the current product component values together with their existing evidence states.

Experimental evidence-state strengths are:

| evidence state | candidate strength |
|---|---:|
| `measured_proxy` | 1.00 |
| `broad_proxy` | 0.75 |
| `neutral_prior` | 0.00 |
| `unavailable` | 0.00 |

Procedure:

1. start from the frozen four ProductScore weights;
2. multiply each weight by the evidence-state strength;
3. exclude neutral/unavailable placeholders from the evidence mean;
4. compute effective evidence coverage;
5. shrink the evidence-only mean toward the neutral anchor 70 by `sqrt(coverage)`;
6. apply the existing display transform only for apples-to-apples telemetry.

If no measured/broad component evidence exists, the candidate is unavailable rather than inventing a measured 70.

Every output explicitly remains:

- `score_mapped = false`
- `product_calibrated = false`
- `user_facing = false`
- `product_score_changed = false`

### Why this is useful

For the common current free-speech case where only fluency is a `broad_proxy`, effective coverage is:

```text
0.25 * 0.75 = 0.1875
```

The candidate therefore allows real fluency evidence to move the headline more than the current three equally weighted neutral anchors do, while still shrinking strongly toward 70 because only a small fraction of the intended four-dimensional construct has actual evidence.

This is a hypothesis about safer score aggregation and useful C-end dispersion. It is not a validated new score contract.

---

## 9. Real-validation tooling now records and analyzes the candidate

`run_free_speech_validation_batch.py` now writes:

```text
score_candidates.partial_evidence_aggregate
```

while preserving the critical validation rule:

```text
transcript = None
scoring_used_gold_transcript = false
```

The batch schema is bumped to `free_speech_validation_batch_v2`.

New analyzer:

`scripts/analyze_partial_evidence_aggregate.py`

It:

- keeps only the latest attempt per `sample_id`;
- can recompute the candidate from stored product component/evidence metadata;
- compares current ProductScore and candidate dispersion;
- reports candidate-current deltas;
- reports effective evidence coverage;
- reports neutral-prior and available-component counts;
- breaks results down by task mode, speaker group, and channel condition.

It does **not** fit a formula, tune shrinkage, or declare promotion.

A useful score distribution without construct-matched human validity is not enough for rollout.

---

## 10. Evidence schema v4, ScoreContract v2

v8 intentionally bumps only the evidence schema:

```text
score_contract_version = consumer_four_score_v2
evidence_schema_version = consumer_evidence_v4
```

Reason:

- public numeric component weights and display transform are unchanged;
- reference-boundary provenance now changes whether local evidence is eligible;
- new shadow aggregate telemetry changes evidence metadata, not ProductScore.

This preserves legitimate numeric history continuity while still making evidence-semantic evolution explicit.

---

## 11. Baseline-control findings and regression cleanup

For the same lightweight Python environment, the unmodified v7 branch produced:

- 498 passed;
- 3 failed;
- 6 warnings;
- 13 subtests passed.

The three failures were not caused by the v8 scoring repairs:

1. a stale ASR test double did not accept the already-established `word_timestamps` keyword;
2. a stale UI contract test still expected the old public-mode list and excluded direct `transcript_assisted_light`;
3. the SSL shadow coupled a valid pronunciation distance to an unavailable short-sequence rhythm derivative.

v8 updates the first two tests to the already-established runtime contract and repairs the third implementation issue as described above.

This prevents the project from treating a historically red full suite as an acceptable baseline.

---

## 12. Final regression

Latest-source full lightweight regression:

```text
514 passed
0 failed
6 warnings
13 subtests passed
```

The warnings are dependency/runtime deprecations or fallback warnings, including legacy Python audio modules, `cgi`, and librosa/audioread compatibility paths. No scoring-integrity test failed.

The permanent `baseline-evolution-light-tests` workflow also passes and now includes the v8 scoring-integrity, partial-evidence aggregate, and analysis guards.

---

## 13. What v8 intentionally does not change

v8 does not:

- change the four public dimensions;
- change ProductScore weights;
- change the display transform;
- replace neutral priors in the official score yet;
- promote ASR recoverability to clarity;
- promote ASR word timing to rhythm;
- promote target-independent F0 movement to intonation;
- promote WavLM/CTC/GOP/rhythm-DTW into `/100`;
- claim strict lexical pitch-accent correctness from unverified targets;
- treat recording quality as learner pronunciation ability;
- add history/progress deltas before score semantics are ready.

---

## 14. Remaining release gates

### Gate A — fresh real-audio free-speech acceptance

Run the real `transcript=None` product path on a speaker-diverse set including, where available:

- native Japanese;
- learner Japanese;
- short natural Japanese turns;
- longer spontaneous Japanese answers;
- English/Mandarin negative controls;
- silence/noise/unusable recordings;
- same-source channel/device/noise variants.

Compare current ProductScore and the partial-evidence shadow without tuning on the held set.

At minimum inspect:

- score availability;
- false no-score on valid Japanese;
- false acceptance on non-Japanese/unusable input;
- IQR/range/ceiling concentration;
- task-mode behavior;
- same-source channel drift;
- speaker dependence;
- candidate evidence coverage.

### Gate B — execute the frozen human criterion protocol

The existing v5 infrastructure should collect construct-matched listener criteria for:

- clarity/comprehensibility;
- fluency;
- rhythm naturalness;
- utterance-level intonation naturalness;
- contextual intonation separately.

Do not use a gold transcript in the product-condition run.

### Gate C — decide component promotion independently

Each free-speech dimension may fail independently.

A failed clarity candidate should not block a successful fluency model. A failed rhythm derivative should not invalidate pronunciation evidence. Missing F0 must not become low intonation.

### Gate D — only then consider ScoreContract v3

A future score-contract change may consider:

- replacing one or more neutral priors with validated component evidence;
- changing headline aggregation so neutral priors do not behave like measurements;
- recalibrating score spread/ceiling behavior using development data only.

Any such change requires a new score-contract version and held acceptance before user rollout.

---

## 15. Product interpretation after v8

The project is now safer in an important way:

- ordinary short/long Japanese is less likely to be rejected for an irrelevant pseudo-reference constraint;
- approximate reference timing can no longer silently become precise-looking local evidence;
- experimental special-mora feedback is no longer implicitly enabled;
- optional research sub-evidence fails independently;
- the known free-speech neutral-prior problem is now measurable through a shadow candidate without changing the user score.

The next information gain should come from real recordings and listener criteria, not another round of arbitrary threshold tuning.
