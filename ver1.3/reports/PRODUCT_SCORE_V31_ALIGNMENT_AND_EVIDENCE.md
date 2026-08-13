# Product Score v3.1 — Alignment Integrity and Evidence Completeness

## Executive verdict

**Do not promote ProductScore v3 to a broad A/B candidate yet.** The P0 semantic defect is fixed: an equal-time fallback is explicit and cannot be consumed as local mora timing. None of the four non-neural DTW variants recovered the codec condition, so local-timing coverage is not yet robust enough. ProductScore v2 remains the only user-facing score; v3 remains default-off/candidate-only.

## P0: explicit alignment provenance

`AlignmentResult` is now the result object consumed by the evaluator. It carries `method`, `available`, `confidence`, `used_equal_fallback`, `failure_reason`, normalized DTW cost/path fields, feature kind and band radius. Every failure route returns equal boundaries only as an explicitly unavailable compatibility result.

Recorded failure reasons include `user_audio_too_short`, feature/frame failure, DTW exception, boundary-mapping failure and `boundary_health_unstable`. A mapped DTW path that fails health now fails inside alignment, permitting the documented wider-band retry.

The prior list-only result did not persist internal fallback provenance, so an exact historical count of *silent internal* fallbacks is not reconstructable. The earlier 37-item artifact recorded 8 outer `cached_dtw_fallback_equal` cases but cannot establish whether any of the 29 remaining `cached_dtw` items had internally fallen back. The v3.1 rerun is unambiguous: **8/37** explicit equal fallbacks and **29/37** available local alignments. This historical ambiguity is why the new fields are necessary.

## Codec robustness comparison

The exact endpointed waveform sent by product VAD was tested on clean, codec, band-limit, gain, mild RIR, 15 dB noise and a wrong-target control. Raw rows: [ALIGNMENT_V31_BENCH.csv](ALIGNMENT_V31_BENCH.csv).

| Variant | Normal channel conditions with local alignment | Codec | Wrong target | Decision |
|---|---:|---:|---:|---|
| A: MFCC DTW | 5/6 | unavailable | unavailable | Keep as candidate baseline |
| B: MFCC + delta | 3/6 | unavailable | unavailable | Rejects more normal audio |
| C: log-mel | 3/6 | unavailable | unavailable | Rejects more normal audio |
| D: MFCC + wider-band retry | 5/6 | unavailable after retry | unavailable | No recovery; adds latency |

All codec failures are `boundary_health_unstable`; none become apparently precise boundaries. A wider band is attempted only after an explicit first-pass failure and did not rescue this codec sample. No alignment feature default changes in this task.

## Evidence scope and global/local timing

The v3 timing record now separates `global_timing_available` from `local_timing_available`. Global duration/reference-duration rate can remain available without mora boundaries and influences fluency. Equal boundaries disable local duration CV, local warp, special-mora timing and target-local rhythm. They do not yield a perfect rate or rhythm result.

The aggregate records `dimensions_available`, `evidence_coverage`, `score_scope` and `ab_candidate_eligible`. These are diagnostic/A-B selection fields only: they neither cap v2 nor create a user no-score.

| 37-item fixed-reference panel measure | Result |
|---|---:|
| Full local alignment / `partial` scope | 29 |
| Equal fallback / `continuity_only` scope | 8 |
| A/B-eligible candidate items | 29 |
| Candidate mean / SD | 79.839 / 10.474 |
| Candidate min / median / max | 49.347 / 82.573 / 98.513 |
| Exact candidate 100s | 0 |

The reproducible, ignored item output is `outputs/product_score_v31/v31_panel.csv`.

## Channel retest

[PRODUCT_SCORE_V31_CHANNEL_RETEST.csv](PRODUCT_SCORE_V31_CHANNEL_RETEST.csv) compares v2, v3 candidate, scope and synthetic-evidence use. No condition used fake local timing. Codec is `continuity_only` (coverage 0.20, not A/B eligible), rather than receiving spurious local rhythm. Its v2 delta is -1 point and v3 candidate delta is -8.891 points. Gain is near-invariant (+0.098 v3); band-limit is -3.475; mild RIR -0.774; 15 dB noise +2.170. These are evidence/scope observations, not calibration targets.

## Sentence-disjoint held-out content cascade

The 80 hard content items are split by target sentence ID: 1–10 development (41: 20 correct/21 wrong) and 11–20 held-out (39: 10 correct/29 wrong). Target IDs are disjoint by a tested helper. Speakers overlap (`jvs001`, `jvs002`, `jvs003`), so this is not speaker-disjoint generalization.

The deterministic development choice is rescue floor 0.40: among zero-false-verification settings, it has maximal fixed retention. Held-out results:

| Metric | Result |
|---|---:|
| Wrong-target false verified | 0/29 (0%) |
| Correct fixed-detail retention | 10/10 (100%) |
| Small-model invocation | 2/39 (5.13%) |
| Broad fallback rate | 29/39 (74.36%) |
| Recorded warm p90 | 0.701 s |

The complete grid is reproducibly generated at `outputs/product_score_v31/content_cascade_heldout.csv`; no content policy changed.

## Native reference bank and WavLM eligibility

[native_reference_overlap.csv](../data/audit/native_reference_overlap.csv) audits the seven local JANON isolated-word targets and exact JVS `parallel100` transcript matches. JVS has no exact match for these isolated targets; each has **four** JANON human-native references (`jpf1`, `jpf2`, `jpm1`, `jpm2`). [native_reference_bank.csv](../data/audit/native_reference_bank.csv) contains 28 entries (7 × 4) and excludes banks outside the requested 3–5 count range.

WavLM ran successfully from the local checkpoint only on sufficient target `うっとうしい`: native leave-one-out used three references and each learner used four. Layer-12 native medians were 0.168–0.183 versus learner 0.208/0.223; layer-24 native medians were 0.168–0.229 versus learner 0.235/0.252. This is one-target feasibility evidence only, without a score mapping or product field. Raw output: `outputs/product_score_v31/wavlm_reference_bank_subset.csv`.

## Tests and promotion decision

The focused v3.1 tests cover provenance/failure, retry, global/local rate, scope/eligibility and split integrity. Full repository regression passed: **170 passed, 6 warnings**.

**Promotion decision: not ready for broad A/B.** At most, a future internal diagnostic can use `ab_candidate_eligible=true` items. No v2 replacement, score mapping adjustment, special-mora work or WavLM score promotion is justified by this evidence.
