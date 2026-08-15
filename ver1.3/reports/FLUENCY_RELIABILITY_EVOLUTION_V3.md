# Fluency / Reliability Evolution v3

## Status

Branch: `fluency-reliability-evolution-v3`

Base: `score-contract-evolution-v2`

This branch continues the C-end baseline without changing the valid-Japanese ProductScore weights or remapping the four user-facing dimensions. Its two goals are:

1. make free-speaking fluency evidence more construct-valid while remaining usable in a conversation product;
2. make reliability-driven legacy score caps auditable without confusing measurement uncertainty with learner ability.

A third P0 issue was discovered while implementing free-speaking fluency: the instant-conversation ASR path could force Japanese decoding before language eligibility was established. That routing problem is fixed in this branch.

## 1. Free speech no longer forces Japanese before language gating

`transcript_assisted_light` previously called the Japanese-constrained ASR helper. That helper is appropriate for fixed-target content verification, but it is unsafe as the first step of free-speaking assessment because non-Japanese speech can be rendered as plausible-looking Japanese text.

The new free-speaking path is:

1. unforced language-aware ASR;
2. Japanese-language eligibility gate;
3. transcript sanity gate;
4. only then Japanese frontend / mora-rate / F0 analysis.

A confident non-Japanese result is now a routing outcome, not a low Japanese score:

- raw score fields are `None`, not zero;
- `details.score_eligible = false`;
- `details.transcript_sanity.ok = false` so the existing product no-score/retry policy applies;
- mora/F0 scoring is not run after rejection.

Deep Review and Instant Conversation now share the same reusable free-speech language eligibility policy.

An externally supplied transcript is also checked before it can drive Japanese mora-rate scoring.

## 2. Spontaneous fluency v2 is a three-channel shadow construct

New module:

`src/jp_speech_eval/spontaneous_fluency.py`

Schema:

`spontaneous_fluency_evidence_v1`

The evidence is explicitly divided into:

- speed;
- breakdown;
- repair.

This follows the established L2 fluency decomposition, but the implementation is adapted conservatively to Japanese and to the evidence currently available in the product.

Everything in this module remains:

- `score_mapped = false`;
- `product_calibrated = false`;
- `user_facing = false`.

The current learner-facing fluency `/100` is not changed by these features.

## 3. Speed evidence

The shadow stores both:

- speech rate in mora / second;
- articulation rate in mora / second, excluding detected long silent pauses.

It also stores:

- mora count;
- utterance speech duration;
- phonation time;
- phonation-time ratio;
- mean mora duration during phonation.

The use of mora is a Japanese engineering adaptation. Numeric thresholds from English syllable-rate studies must not be copied directly into this product.

## 4. Breakdown evidence

Long silent pauses currently use the existing acoustic pause detector with a nominal 0.30 s threshold. The shadow reports:

- pause count;
- total pause duration;
- pause ratio;
- mean / median / maximum pause duration;
- pauses per minute;
- pauses per 100 mora;
- mean run duration between long silent pauses.

These are descriptive evidence, not learner-error counts.

## 5. Weak pause-location evidence from ASR timestamps

The language-aware faster-whisper path can now request word timestamps. When present, a detected acoustic pause can be anchored between ASR words.

The only location labels exposed are low-confidence candidates:

- `leading_pause_candidate`;
- `trailing_pause_candidate`;
- `after_asr_punctuation_candidate`;
- `within_asr_phrase_candidate`;
- `unanchored_pause_candidate`.

ASR punctuation is not treated as a syntactic clause annotation. Therefore this branch does **not** claim to measure strict mid-clause versus clause-final pause frequency.

If word timestamps are unavailable, pause location remains unavailable instead of being guessed.

## 6. Repair evidence is deliberately conservative

Transcript-side repair evidence currently records:

- relatively lexicalized filler candidates such as `えーと`, `えっと`, `うーん`, `んー`;
- ambiguous discourse-marker counts such as `あの`, `その`, `まあ`, `なんか` in a separate field;
- ambiguous repair markers such as `いや`, `というか`, `じゃなくて` in a separate field;
- exact adjacent one- or two-token repetition candidates.

These features have low confidence because general ASR can omit or normalize fillers, false starts, repetitions and cut-offs. They are not mapped to a penalty.

## 7. Reliability caps are still unchanged in runtime

The fixed-reference legacy evaluator still contains the existing post-score caps:

- equal-alignment fallback: pronunciation at most 80;
- insufficient mora evidence: pronunciation at most 60;
- F0 coverage below 0.50: prosody at most 55;
- overall reliability below 0.75: aggregate total at most 82.

This branch does **not** remove or retune any of them.

The reason is methodological as well as product-driven: before changing them we need to know how much they actually move scores, on which recordings, and whether removing them improves agreement with human criteria without creating misleadingly confident scores.

## 8. Historical cap A/B exposed scorer/config drift

New audit code:

- `src/jp_speech_eval/reliability_counterfactual.py`
- `scripts/run_reliability_cap_ab.py`

The existing `c_end_v2_acceptance/new_results.jsonl` contains enough stored mora/F0/pause/reliability evidence to replay several scorers, even though the original WAV paths are local and unavailable in CI.

However, a historical replay is not automatically a cap counterfactual. The current scorer is first passed through the historical component-cap equations and the final aggregate cap, and that full path must reproduce the stored post-cap result.

Frozen historical audit result:

`reports/data/reliability_cap_historical_acceptance_v3.json`

Strict replay summary:

- 40 acceptance rows total;
- 37 applicable fixed-evaluator rows;
- 37 current-scorer formula replays available;
- only 6 reproduce the complete historical post-cap scoring path;
- 31 show scorer/config drift under current replay;
- 4 of the compatible rows are censored at a cap ceiling;
- only 2 rows are historically compatible and uncensored enough for a trustworthy historical counterfactual;
- both of those two show zero cap effect.

The raw current-scorer candidate deltas must therefore **not** be interpreted as the effect of removing caps. In particular, prosody replay ranges from -78 to +52 relative to the stored historical score, directly demonstrating why historical scorer drift must be separated from cap mechanics.

Decision from this historical acceptance: **none**. It does not justify removing the caps.

## 9. Old telemetry was insufficient for exact historical cap recovery

The old evaluator stored only scores after the component caps and final aggregate cap. If a stored score is exactly on a cap ceiling, the original pre-cap value has been destroyed:

- stored pronunciation 60 could have been 60, 75, 90, or 100 before the 60 cap;
- stored prosody 55 could have been any higher value before the F0 cap.

No later replay can recover that lost magnitude reliably when scorers/configs have evolved.

This is now fixed for all future fixed-reference results.

## 10. Evaluator-native exact pre/post-cap telemetry

The evaluator now records:

`details.reliability_cap_audit`

schema:

`reliability_cap_audit_v1`

For the *same evaluation run*, it stores:

- exact scorer outputs before reliability caps;
- exact component scores after reliability caps;
- all cap-trigger booleans;
- cap values;
- aggregate weights;
- aggregate score using uncapped components;
- aggregate score after component caps but before the overall reliability cap;
- final stored aggregate score after the overall cap.

This telemetry is audit-only:

- `user_facing = false`;
- `product_score_changed = false`.

The existing numeric scoring path is preserved.

New parser and audit utility:

- `src/jp_speech_eval/native_cap_telemetry.py`
- `scripts/audit_native_reliability_caps.py`

They can recover and summarize an exact same-run cap counterfactual without rerunning ASR, alignment, F0 extraction, the acoustic model, or the scorer.

This is the preferred evidence for the next fresh acceptance study.

## 11. What changed for users in this branch

For valid Japanese speech, the current numeric ProductScore contract is intentionally unchanged.

One user-visible behavioral correction is intentional:

- confidently non-Japanese free speech now receives no score / retry instead of being coerced through Japanese ASR and potentially receiving a misleading Japanese score.

That behavior matches the product policy that plausible Japanese should receive scores, while non-Japanese input is an eligibility/routing case rather than a pronunciation failure.

## 12. What is intentionally not promoted

This branch does **not**:

- change the ProductScore 30 / 25 / 25 / 20 weights;
- change the display transform;
- change valid-Japanese fluency thresholds;
- map repair candidates to penalties;
- call ASR punctuation a true clause boundary;
- map spontaneous-fluency v2 to `/100`;
- remove reliability caps;
- infer a cap effect from historical scorer drift;
- promote WavLM / CTC / GOP / rhythm-DTW to ProductScore.

## 13. Next experimental gate

The next cap decision must use a fresh held acceptance generated by code that already emits native cap telemetry.

The acceptance should preserve the existing C-end philosophy and include, where legally/operationally available:

- native Japanese;
- real learner Japanese;
- same-duration wrong Japanese target/content cases;
- English and Mandarin human negative controls;
- silence and noise;
- low-level, clipping and channel variants.

For every fixed-reference sample, one evaluation run will now provide both:

- the current capped legacy evaluator path;
- the exact uncapped component/aggregate counterfactual.

No cap/threshold tuning should use the held acceptance itself.

Only after combining this fresh A/B with the already defined human criteria should the product decide whether reliability should continue to alter raw scores, or instead primarily alter confidence, detail and abstention.

## 14. Codex / local execution boundary

The architecture, audit equations, telemetry, fluency evidence, tests and statistics do not require Codex.

Local/Codex execution becomes useful only when the needed assets are local or computationally heavy, for example:

- rerunning the physical acceptance audio that is not stored in GitHub;
- batch faster-whisper / WavLM / CTC inference over local licensed corpora;
- using external human recordings or UME-JRF files held locally;
- collecting and processing real listener ratings.

When used, Codex should execute this frozen protocol rather than redesigning the scoring logic.
