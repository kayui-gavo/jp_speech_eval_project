# Scoring Integrity v8

## Status

Branch: `scoring-integrity-v8`

Base: `hf-space-karaoke-integration-v7`

Draft PR: #8

This round follows a fresh end-to-end audit of the C-end path: recording eligibility, language routing, ASR, fixed-reference alignment, four public dimensions, reliability, feedback, karaoke visualization, research shadows, validation tooling, and CI.

The main conclusion is unchanged: the project’s evidence/reliability safeguards are currently more mature than several underlying score measurements. v8 therefore fixes places where routing or provenance could create a false product claim, while deliberately **not** pretending that free-speech four-dimensional scoring is already solved.

Official product contract remains:

- `score_contract_version = consumer_four_score_v2`
- clarity .30 / rhythm .25 / fluency .25 / intonation .20
- same display transform
- no official ProductScore promotion in this branch

Evidence semantics advance to:

- `evidence_schema_version = consumer_evidence_v4`

because reference-boundary provenance now changes whether target-local evidence is eligible for public interpretation.

---

## 1. Direct free speech no longer inherits pseudo-reference length limits

### Previous problem

`transcript_assisted_light` reused `check_asr_transcript_sanity()`. That gate was designed for deciding whether an ASR transcript was suitable for pseudo-reference generation and therefore contained limits such as a minimum of three content characters and a maximum of eighty.

Those are not valid eligibility rules for direct conversation. Natural Japanese turns such as `はい` or `え？`, and natural longer answers, could be rejected for reasons unrelated to whether they were valid Japanese speech.

### v8 behavior

Direct free speech now uses `check_free_speech_transcript_sanity()`.

It checks only direct-speech usability after the independent language gate:

- non-empty content;
- predominantly Japanese script;
- no obvious repetition/noise-like ASR hallucination pattern.

It deliberately has no pseudo-reference 3–80 character rule.

The product contract is now explicit:

```text
language eligibility
    != direct-free-speech scoreability
    != pseudo-reference eligibility
```

A short utterance may still have insufficient evidence for a particular dimension. That becomes evidence/reliability metadata, not a false claim that the Japanese turn itself is invalid.

---

## 2. Reference-boundary provenance is now consumed, not merely recorded

Earlier work correctly added:

- `ref_boundary_method`
- `ref_boundary_confidence`
- `ref_boundary_tier`
- `ref_boundary_source`

However, a low-precision equal-mora reference could still be mapped through a healthy DTW path and later appear more precise than its source timing actually justified.

v8 carries reference-boundary provenance into `details.alignment` and all downstream consumers.

Target-local precision is limited when:

- learner alignment itself falls back; or
- reference boundary tier is `equal_fallback` / `unknown_alignment`; or
- reference-boundary confidence is below the local-evidence floor.

Consequences are deliberately local:

- broad/global scoring remains available;
- exact target-local timing evidence is downgraded;
- strict pitch feedback is blocked;
- special-mora learner feedback is blocked;
- local pronunciation detail is blocked;
- karaoke mora synchronization becomes approximate.

Most importantly, **approximate reference timing alone does not change the learner’s `practice_check_result` to `needs_attention`**. It is a system/reference limitation, not evidence that the learner performed poorly.

The intended rule is:

```text
measurement uncertainty -> lower confidence / less local detail
measurement uncertainty != learner performance penalty
```

---

## 3. Karaoke synchronization now respects both sides of the alignment

The mora replay layer previously depended mainly on learner-side alignment confidence.

v8 defines effective local confidence conservatively. When reference-boundary confidence exists, local confidence cannot exceed it.

Mora timeline rows preserve:

- effective `alignment_confidence`
- `path_alignment_confidence`
- `reference_boundary_confidence`
- `reference_boundary_tier`
- `approximate`

A healthy DTW path therefore cannot visually upgrade low-precision reference timing into precise-looking mora synchronization.

This remains playback alignment, not phone correctness.

---

## 4. Special-mora learner feedback is explicit opt-in

The limited special-mora candidate has useful false-alarm safeguards, but current evidence still does not prove learner-error detection or learner benefit. Native false-alarm calibration and synthetic feature shortening are not substitutes for learner-labelled validation.

Therefore `render_user_facing_result()` now defaults:

```text
enable_user_facing_calibrated_special_mora = false
```

Runtime/shadow evidence remains available. Explicit experiments may opt in deliberately.

---

## 5. Optional SSL rhythm evidence no longer invalidates pronunciation evidence

A v7 full-suite control exposed a pre-existing research-path coupling issue: a short SSL feature sequence could produce a valid pronunciation/reference cosine-DTW distance while `rhythm_dtw_v1` failed because its smoothing window required more frames.

v8 isolates these constructs:

- valid SSL pronunciation evidence remains available;
- only `rhythm_dtw_v1` becomes unavailable;
- reason: `insufficient_dtw_frames_for_rhythm_metric`.

One optional sub-evidence failure no longer contaminates another successful measurement.

---

## 6. Public demo avoids hidden realtime-debug work

The formal Hugging Face Space UI remains `debug_ui/index.html`.

Public mode already hid research/debug panels, but fixed-reference evaluation still computed realtime replay rows that users never saw. v8 skips that hidden computation in public-demo mode while preserving it for local/debug use.

This is a latency/CPU cleanup only.

---

## 7. The major unresolved score problem remains explicit

Official free-speech ProductScore still commonly behaves approximately as:

- clarity: neutral prior 70 when no promoted evidence exists;
- rhythm: neutral prior 70 when no promoted evidence exists;
- intonation: neutral prior 70 when no promoted evidence exists;
- fluency: the primary moving product component.

The per-dimension evidence layer is honest because these values are labelled `neutral_prior`, but the headline score is still compressed because neutral placeholders retain their normal ProductScore weights.

v8 does **not** solve this by promoting unvalidated free-speech shadows.

ASR recoverability is not automatically human clarity/comprehensibility. ASR word timing is not automatically rhythm naturalness. Global F0 movement is not automatically context-appropriate intonation.

The official `/100` formula is unchanged.

---

## 8. Shadow partial-evidence aggregate

New module:

`src/jp_speech_eval/partial_evidence_aggregate.py`

Schema:

`partial_evidence_aggregate_shadow_v1`

Policy:

`neutral_prior_excluded_coverage_shrunk_v1`

This candidate tests aggregation semantics only. It does not promote any new acoustic model.

Experimental evidence strengths are:

| evidence state | candidate strength |
|---|---:|
| `measured_proxy` | 1.00 |
| `broad_proxy` | 0.75 |
| `neutral_prior` | 0.00 |
| `unavailable` | 0.00 |

Procedure:

1. start from frozen ProductScore component weights;
2. multiply each by its evidence-state strength;
3. exclude neutral/unavailable placeholders from the evidence mean;
4. compute effective evidence coverage;
5. shrink the evidence-only mean toward 70 by `sqrt(coverage)`;
6. apply the existing display transform for apples-to-apples telemetry only.

If no measured/broad evidence exists, the candidate is unavailable instead of inventing a measured 70.

Every output remains:

- `score_mapped = false`
- `product_calibrated = false`
- `user_facing = false`
- `product_score_changed = false`

For the common current case where only fluency is `broad_proxy`, effective coverage is:

```text
0.25 * 0.75 = 0.1875
```

The candidate therefore tests whether real fluency evidence can create more useful headline dispersion without pretending that the other three dimensions were measured. It remains a hypothesis, not a new score contract.

---

## 9. Real-validation tooling

`run_free_speech_validation_batch.py` now stores:

```text
score_candidates.partial_evidence_aggregate
```

while preserving:

```text
transcript = None
scoring_used_gold_transcript = false
```

The batch schema is `free_speech_validation_batch_v2`.

`analyze_partial_evidence_aggregate.py` reports:

- current ProductScore distribution;
- shadow candidate distribution;
- candidate-current deltas;
- effective evidence coverage;
- neutral-prior count;
- available-component count;
- task-mode breakdown;
- speaker-group breakdown;
- channel-condition breakdown.

It does not fit or tune a formula.

### One-command acceptance runner

New:

`scripts/run_free_speech_acceptance_v8.py`

Runbook:

`reports/FREE_SPEECH_ACCEPTANCE_V8_RUNBOOK.md`

The runner performs in one execution:

1. manifest validation;
2. real `transcript_assisted_light` product evaluation with `transcript=None`;
3. current ProductScore + partial-evidence shadow collection;
4. descriptive aggregate analysis;
5. routing/availability summary.

Routing is kept separate for:

- `expected_japanese`
- `expected_non_japanese_speech`
- `expected_nonspeech_control`

This prevents language-routing false accepts from being mixed with silence/noise failures.

For valid Japanese, a runtime/evaluation error counts as failure to return a normal score. The availability denominator therefore cannot be improved by silently dropping crashes.

No result from this runner alone is sufficient for production promotion.

---

## 10. Evidence schema v4, ScoreContract v2

v8 intentionally changes only evidence semantics:

```text
score_contract_version = consumer_four_score_v2
evidence_schema_version = consumer_evidence_v4
```

Reason:

- public numeric component weights are unchanged;
- display transform is unchanged;
- reference provenance now changes local evidence eligibility;
- new aggregate telemetry remains shadow-only.

This preserves legitimate numeric-history continuity while making evidence-semantic evolution explicit.

---

## 11. Baseline-control findings

Using the same lightweight Python environment, unmodified v7 produced:

- 498 passed;
- 3 failed;
- 6 warnings;
- 13 subtests passed.

The three failures were pre-existing:

1. stale ASR test double did not accept the established `word_timestamps` keyword;
2. stale UI contract still expected the old public-mode list;
3. SSL pronunciation evidence was coupled to a short-sequence rhythm derivative failure.

v8 updates the first two tests to the established runtime contract and repairs the third implementation issue rather than accepting a permanently red full-suite baseline.

---

## 12. Final regression

Fresh full lightweight regression after the final C-end semantic review and acceptance-runner addition:

```text
517 passed
0 failed
6 warnings
13 subtests passed
```

The warnings are dependency/runtime deprecations or fallback warnings, including legacy Python audio modules, `cgi`, and librosa/audioread compatibility paths. No scoring-integrity or acceptance-runner test failed.

Permanent `baseline-evolution-light-tests` also passes and covers:

- v8 scoring-integrity guards;
- partial-evidence aggregate;
- partial-evidence analysis;
- one-command free-speech acceptance runner;
- existing reliability-cap audit.

Temporary full-regression workflows used during development were removed after successful runs.

---

## 13. What v8 intentionally does not change

v8 does not:

- change the four public dimensions;
- change ProductScore weights;
- change the display transform;
- replace neutral priors in the official score;
- promote ASR recoverability to clarity;
- promote ASR word timing to rhythm;
- promote target-independent F0 movement to intonation;
- promote WavLM/CTC/GOP/rhythm-DTW into `/100`;
- claim lexical pitch-accent correctness from unverified targets;
- treat recording/reference quality as learner ability;
- add history/progress deltas before score semantics are ready.

---

## 14. Remaining release gates

### Gate A — real speaker-diverse free-speech acceptance

Use `run_free_speech_acceptance_v8.py` with locally available real audio containing, where possible:

- learner Japanese;
- native Japanese;
- short natural Japanese turns;
- longer spontaneous Japanese answers;
- English negative controls;
- Mandarin Chinese negative controls;
- silence/noise/unusable controls;
- same-source device/noise/codec variants.

Do not tune thresholds on the held acceptance.

Inspect at minimum:

- valid-Japanese normal-score availability, including runtime errors;
- non-Japanese-speech false normal scores;
- nonspeech false normal scores;
- current vs shadow score range/IQR/ceiling concentration;
- task-mode behavior;
- speaker dependence;
- same-source channel drift;
- candidate evidence coverage.

### Gate B — construct-matched listener criteria

Use the frozen listener protocol for:

- clarity/comprehensibility;
- fluency;
- rhythm naturalness;
- utterance-level intonation naturalness;
- contextual intonation appropriateness separately.

### Gate C — promote dimensions independently

A failed clarity candidate must not block a valid fluency model. A failed rhythm derivative must not invalidate pronunciation evidence. Missing F0 must not become low intonation.

### Gate D — only then consider ScoreContract v3

A future contract may consider:

- replacing validated neutral priors;
- changing headline aggregation so neutral priors do not behave like measurements;
- recalibrating score spread/ceiling behavior using development data only.

Any official score change requires a new ScoreContract version and a fresh held acceptance.

---

## 15. Product interpretation after v8

The system is materially safer after this round:

- short/long natural Japanese is less likely to be rejected by an irrelevant pseudo-reference rule;
- low-precision reference timing can no longer silently become precise-looking local evidence;
- a system/reference limitation no longer becomes a learner `needs_attention` state by itself;
- experimental special-mora feedback is not implicitly enabled;
- optional research sub-evidence fails independently;
- neutral-prior headline behavior is now measurable through a shadow candidate without changing user scores;
- the next real-audio validation can be executed reproducibly in one command.

The next information gain should come from real recordings and listener criteria, not another arbitrary threshold round.
