# Free-Speech Promotion Gate v5

Status: engineering-complete validation infrastructure; no ProductScore promotion

Branch: `free-speech-promotion-gate-v5`

Base: `free-speech-four-dimension-evidence-v4`

## 1. Why this round exists

v4 made free-speech evidence observable without silently replacing neutral product priors.  A second audit found that the v4 promotion protocol promised more than the analysis pipeline could actually execute.

The v4 evidence CSV did not preserve all metadata required for speaker-disjoint, task-mode, or channel-paired analysis.  The human intonation rubric also mixed two different questions: whether the pitch movement of an isolated utterance sounds natural, and whether it is appropriate in the actual dialogue context.

For a C-end product these are not minor research bookkeeping issues.  They can create false confidence that a candidate is robust enough for users when it has only been checked on pooled clean recordings or against a construct the model did not observe.

v5 therefore changes the validation infrastructure, not the user-facing score.

## 2. Product behavior deliberately unchanged

This round does not:

- change `consumer_four_score_v2`;
- change the 30/25/25/20 component weights;
- change the display transform;
- promote any v4 free-speech shadow candidate;
- add another online ASR pass;
- add WavLM, CTC/GOP, TTS, or reference DTW to free conversation;
- turn recording quality into clarity or pronunciation ability;
- turn raw ASR probability into human comprehensibility;
- turn word timestamps into phone/mora correctness;
- turn isolated F0 range into contextual intonation appropriateness.

The public dimensions remain exactly:

1. 明瞭さ
2. 流暢さ
3. リズム
4. 抑揚

The extra intonation distinction introduced in v5 is a validation sub-criterion, not a fifth product dimension.

## 3. A real sample metadata spine

Added:

- `data/human_eval/free_speech_sample_manifest_template_v1.csv`
- `data/human_eval/free_speech_sample_manifest_schema_v1.json`
- `scripts/validate_free_speech_sample_manifest.py`

The manifest now preserves the analysis variables required by the frozen gates:

- `speaker_id`
- `speaker_group`
- `l1`
- `task_mode`
- `prompt_id`
- `split`
- `expected_language`
- `channel_condition`
- `channel_pair_id`
- `source_recording_id`
- context provenance

Development and held Japanese learner/native speakers must be disjoint.

### Same-source channel controls

A channel pair is not merely two recordings from the same speaker reading/responding to the same prompt.  Re-speaking introduces real production variation and cannot identify microphone/noise/codec sensitivity.

v5 therefore requires every channel-paired variant to share one non-empty `source_recording_id`.  Clean, low-level, noisy, or codec/device variants are expected to be derived from the same underlying source utterance.  A clean anchor is mandatory.

This makes the C-end channel-drift gate interpretable as recording/channel sensitivity rather than learner-performance variation.

## 4. Context-safe human criterion v3

Added:

- `data/human_eval/consumer_rating_schema_v3.json`
- `scripts/build_free_speech_listener_pack_v3.py`
- `scripts/validate_consumer_ratings_v3.py`
- `scripts/normalize_consumer_ratings_v3.py`

The public intonation dimension remains one product dimension, but human validation is separated into:

- `intonation_utterance_naturalness`: isolated target utterance;
- `intonation_contextual_appropriateness`: target utterance judged with its real prompt or preceding-turn context.

Current target-independent F0 evidence is eligible only for the first construct.  No current v5 candidate is declared eligible for contextual intonation.

### Two presentation variants

`isolated` presentations collect:

- clarity/comprehensibility
- fluency
- rhythm naturalness
- utterance-level intonation naturalness

`contextual` presentations collect only:

- contextual intonation appropriateness

This prevents the context needed for an intonation judgement from priming the listener's clarity judgement.  The target response transcript is not displayed.

### Listener blinding

Listener-facing CSV rows do not expose:

- learner/native group
- L1
- split
- channel condition
- channel pair id
- source recording id
- source provenance
- machine scores
- raw source path

A separate private asset map resolves randomized audio asset ids to source files.

Analysis-only metadata is joined back after rating collection.

## 5. Practical human-rating floor

This is a C-end validation pipeline rather than formal high-stakes educational measurement, so v5 does not add IRT or many-facet Rasch machinery.

It does, however, enforce a practical criterion reliability floor:

- held promotion analysis includes a sample × criterion aggregate only when it has at least 3 valid listener ratings;
- target collection remains 5 ratings per analyzable sample × criterion.

Insufficient listener coverage is missing criterion evidence, not a bad learner score.

## 6. Product-realistic batch evaluation

Added:

- `scripts/run_free_speech_validation_batch.py`

The runner executes the actual `evaluate_transcript_assisted_light` path and then the current `apply_user_score_policy`.

The most important rule is hard-coded:

`transcript=None`

The validation run never substitutes a manually corrected or gold transcript for the product ASR transcript.  `scoring_used_gold_transcript=false` is persisted and later checked by the promotion analyzer.

The runner is resumable.  A per-sample failure is recorded rather than aborting the full batch.  Previously successful samples are skipped; failed samples can be retried.

## 7. Retry-safe and provenance-complete evidence export

Added:

- `scripts/export_free_speech_v5_evidence.py`

A resumable JSONL may contain an old error attempt followed by a successful retry.  The raw attempt history remains append-only, but the analysis export uses only the latest attempt per `sample_id` and reports how many attempts were superseded.

The export retains:

- speaker/task/split/channel/source-recording metadata;
- language and product eligibility outcomes;
- score-contract version;
- evidence-schema version;
- candidate-surface policy id;
- correct ASR model provenance for ASR evidence;
- correct candidate-policy provenance for shadow score surfaces.

## 8. Neutral placeholder is not measured evidence

This round found an especially important semantic edge case.

The v4 shadow surface intentionally returns a neutral numeric value such as 70 when source evidence is unavailable.  This is useful for safe product fallback, but the same number must not enter criterion correlation as if the model measured a 70.

v5 therefore separates:

- `evidence_value`: populated only when the underlying candidate evidence is actually available;
- `fallback_numeric_value`: preserves the neutral numeric placeholder for safety/UX auditing.

Example:

- F0 extraction unavailable;
- intonation shadow surface keeps neutral `70`;
- exported `available=false`;
- exported `evidence_value` is blank;
- exported `fallback_numeric_value=70`;
- failure reason is `neutral_placeholder_without_source_evidence`.

The candidate loses availability coverage instead of receiving a fake measured observation.  Separately, the F0-missingness safety check verifies that the neutral fallback did not turn model failure into a low intonation penalty.

## 9. Raw diagnostic features and 0–100 score surfaces are not the same object

v4's protocol used gates such as score IQR in points and channel drift in points.  Applying those thresholds to a 0–1 ASR probability or a log-duration MAD would be dimensionally invalid.

v5 classifies candidates as:

- `diagnostic_feature`
- `diagnostic_feature_nonmonotonic`
- `score_surface`

Raw diagnostics may be inspected for construct association and model design, but they are not directly eligible for ProductScore promotion.

Only these current score surfaces are predeclared for direct promotion testing:

- `clarity.shadow_candidate_score` → `clarity_comprehensibility`
- `rhythm.shadow_candidate_score` → `rhythm_naturalness`
- `fluency.current_product_proxy` → `fluency`
- `intonation.shadow_candidate_score` → `intonation_utterance_naturalness`

No current candidate is eligible for `intonation_contextual_appropriateness`.

## 10. Executable frozen promotion gate

Added:

- `data/research_eval/free_speech_v5_promotion_protocol.json`
- `scripts/analyze_free_speech_promotion_v5.py`

The analyzer does not fit thresholds, tune shrinkage, remap scores, or promote candidates.  It executes the predeclared gates and returns one of:

- `pass`
- `fail`
- `insufficient`
- `diagnostic_only`

A score surface must satisfy the frozen C-end checks including:

- enough held construct-matched aggregates;
- candidate availability;
- construct-matched association;
- useful 0–100 score dispersion;
- both free-speech task modes represented;
- positive task-wise direction rather than a pooled effect driven by one task;
- leave-one-speaker-out direction stability;
- learner/native speaker diversity;
- same-source channel-paired drift;
- negative-control no-score/retry behavior;
- zero oracle-transcript use;
- development/held speaker disjointness;
- F0-missingness safety for intonation;
- exact score-contract/evidence-schema/candidate-policy version consistency.

Missing robustness data produce `insufficient`, not a false pass.

Mixed score-contract/evidence/candidate-policy versions fail closed instead of being silently pooled.

## 11. C-end interpretation

The purpose of these gates is not to make a research table look stricter.  They target concrete product failure modes:

- a score that changes because the microphone changes;
- a score that looks valid only because native and learner labels are separable;
- a pooled correlation driven by one speaker or one task type;
- an ASR confidence score being mistaken for human clarity;
- an F0 extraction failure being shown as poor intonation;
- a neutral fallback being counted as measured evidence;
- an oracle transcript making offline validation unrealistically easy;
- history from two scoring versions being mixed and interpreted as one model.

For this product, these errors are more damaging than failing to squeeze a few extra points of correlation from a small clean benchmark.

## 12. Validation

The final v5 branch light CI at head `566f81cfeadba2ebec261f6420964c47004c5861` completed successfully:

- 128 tests passed;
- 3 dependency deprecation warnings;
- legacy reliability-cap audit completed successfully and retained its previous historical-data limitations.

The v5 regression set includes explicit guards for:

- sample split leakage;
- same-source channel-pair provenance;
- mandatory clean channel anchor;
- isolated/contextual listener separation;
- listener metadata blinding;
- context payload requirements;
- post-rating metadata rejoin;
- no oracle transcript in batch evaluation;
- resumable batch processing;
- latest-attempt evidence export;
- shadow policy/version provenance;
- neutral-placeholder availability semantics;
- full-gate pass behavior;
- missing-channel-control `insufficient` behavior;
- mixed score-contract fail-closed behavior.

## 13. Next real information gain

The next high-value step is no longer another heuristic formula.

The infrastructure is now ready for a real speaker-diverse Japanese validation collection under the frozen v5 protocol.  Once that evidence exists, each dimension can fail independently:

- if clarity fails, investigate stronger target-independent human-intelligibility/comprehensibility evidence;
- if rhythm fails, replace weak ASR-word timing evidence rather than adjusting arbitrary score thresholds;
- if fluency fails, inspect speed/breakdown/repair behavior by task and speaker;
- if utterance-level intonation fails, improve acoustic/prosodic representation;
- contextual intonation remains unclaimed until a future evaluator consumes equivalent dialogue context.

Only a score surface that passes the frozen held gate should enter a new product A/B branch, and that promotion must use a new score-contract version.
