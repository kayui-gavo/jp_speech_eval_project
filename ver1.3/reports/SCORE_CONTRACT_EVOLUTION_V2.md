# Score Contract Evolution v2

## Status

Branch: `score-contract-evolution-v2`

Base: `baseline-evolution-v1`

This branch is a product-semantics and research-baseline evolution. It deliberately avoids retuning the existing C-end score distribution while fixing places where old/raw scores could be misinterpreted as comparable product progress.

## 1. ScoreContract is now executable and versioned

New module:

`src/jp_speech_eval/score_contract.py`

It is the single source of truth for:

- four product constructs;
- component weights;
- display transform;
- score/evidence schema versions;
- mode families;
- fixed-reference identity;
- history comparability.

Current contract:

- `score_contract_version = consumer_four_score_v2`
- `evidence_schema_version = consumer_evidence_v2`
- weights: clarity .30 / rhythm .25 / fluency .25 / intonation .20
- display transform: `70 + 1.08*(raw - 70)` clipped to `[0,100]`

The numeric formula is intentionally unchanged from the immediately preceding executable product policy. The change is that code, API telemetry, history, and documentation now share one versioned definition.

Fixed-reference comparison prefers a stable explicit `reference_id`, then a path-independent generated-reference config hash. Cache paths are retained only as a legacy fallback identity.

## 2. Fake progress protection

New progress records store the four consumer scores plus score context. Legacy evaluator scores remain under `legacy_scores` for audit.

A headline score delta is suppressed when:

- the previous record has no score-contract version;
- score-contract versions differ;
- evaluation mode families differ;
- fixed targets differ;
- the fixed-reference identity changed or is missing.

Stable observables such as speech rate may still be compared when appropriate.

Cross-target voice calibration continues to model personal acoustics, but it no longer silently treats its mean `/100` score as a same-item correctness baseline.

## 3. Reference-dependency gap correction

The previous app-core prototype subtracted Step 3 free-production total from Step 1 shadowing total and interpreted the difference as reference dependence.

That inference is unsafe when the two steps use different score/evidence families. The new behavior:

- computes a headline score gap only if score contexts are explicitly comparable;
- otherwise returns `gap=None`;
- retains descriptive cross-step rate and pause changes.

This prevents a scoring-mode change from being presented as learner regression.

## 4. WavLM-DTW rhythm shadow

New module:

`src/jp_speech_eval/rhythm_dtw.py`

The SSL DTW implementation now exposes an internal optimal warping path while preserving the existing path-free `cosine_dtw_distance()` API.

`rhythm_dtw_v1` derives tempo irregularity from the path:

1. map every reference frame to the mean aligned learner-frame position;
2. smooth the trace with a five-frame moving average;
3. estimate local warp slopes/angles;
4. measure mean absolute deviation from the utterance's average warp angle.

The implementation uses only valid moving-average centers so edge padding does not create artificial irregularity in a perfectly linear warp path. This boundary artifact was caught by CI and fixed in the algorithm rather than hidden by loosening the test.

A global frame-duration ratio is reported separately. A uniformly slower/faster utterance can therefore have low local tempo irregularity instead of being misclassified as locally unstable rhythm.

Everything remains:

- shadow only;
- `score_mapped=false`;
- `product_calibrated=false`;
- not user-facing.

## 5. Multi-native SSL reference panels

New module:

`src/jp_speech_eval/ssl_reference_panel.py`

An explicit research panel can now provide several references for the same target with:

- stable `reference_id`;
- target text;
- audio path;
- speaker ID;
- reference kind;
- provenance.

When a panel is supplied, WavLM pronunciation distance and DTW-rhythm evidence are computed independently against each matching human reference and then aggregated using the preselected strategy, currently median by default.

TTS fallback entries are not mixed into a human panel when human references are available. If an explicitly supplied panel does not contain the requested target, the shadow fails closed instead of silently reverting to another reference generation.

Template:

`data/research_eval/ssl_reference_panel_template.json`

## 6. Interval distortion boundary

`interval_distortion_from_dtw_path()` is implemented only as a pure metric over externally supplied frame labels:

- vowel;
- consonant;
- silence.

No Japanese interval classifier is claimed. Promotion is blocked until such a classifier is validated independently.

This avoids importing the paper's TIMIT-trained auxiliary classifiers as if they were automatically valid for Japanese L2 speech.

## 7. Conversation product routing

New module:

`src/jp_speech_eval/conversation_mode_policy.py`

The product now has an explicit routing contract instead of exposing internal evaluator names as interchangeable product modes:

### `instant_conversation`

- routes to broad `transcript_assisted_light` evaluation;
- evaluates immediately;
- does not generate a hidden TTS scoring reference;
- does not claim target-relative kana/phone errors;
- does not enable strict lexical pitch-accent correction.

### `deep_review`

- without confirmed text: requests transcript confirmation and does not create a scoring reference;
- after confirmation: may use the existing `asr_confirmed_weak_reference` route;
- the resulting TTS reference remains weak practice evidence, not pronunciation ground truth.

### `fixed_practice`

- requires a fixed target;
- routes to the fixed-reference evaluator;
- target-local detail still depends on verified reference/alignment/construct-specific reliability gates.

## 8. Human criterion split

New protocol:

`reports/CONSUMER_CRITERION_PROTOCOL_V2.md`

The existing frozen pronunciation-accuracy protocol is preserved. The new protocol separately defines human criteria for:

- clarity/comprehensibility;
- fluency;
- rhythm naturalness;
- phrase/sentence intonation naturalness;
- independent analyzability.

Pronunciation accuracy is not renamed clarity.

New data contract/tooling:

- `data/human_eval/consumer_rating_schema_v2.json`
- `data/human_eval/consumer_rating_template_v2.csv`
- `scripts/validate_consumer_ratings.py`

The validator enforces:

- unanalyzable audio is not converted into low construct ratings;
- assigned constructs require a 1–7 rating when analyzable;
- unassigned constructs remain null;
- intonation ratings without pragmatic context are retained but flagged for separate analysis;
- listener/presentation duplicates are rejected.

## 9. External research benchmark boundary

New plan:

`reports/UME_JRF_BENCHMARK_PLAN_V1.md`

New normalized research manifest support:

- `data/research_eval/ume_jrf_manifest_template.csv`
- `scripts/validate_research_manifest.py`

The repository does not bundle UME-JRF. Local licensed data is mapped into a normalized manifest, then frozen benchmark code consumes that manifest. This minimizes corpus-specific code and keeps licensing/access separate from product runtime.

## 10. Frozen research split mechanics

New script:

`scripts/build_research_splits.py`

It creates deterministic whole-group folds before benchmark outcomes are seen:

- a speaker-disjoint evaluation view;
- a target-disjoint evaluation view.

These are explicitly two separate views, not falsely described as one simultaneously speaker-and-target-disjoint split. Leakage guards verify that one speaker/target is never split across folds within its corresponding view.

## 11. What is intentionally unchanged

This branch does **not**:

- change the 30/25/25/20 product weights;
- change the current display transform numerically;
- promote WavLM to a clarity `/100` score;
- promote CTC/GOP to a clarity `/100` score;
- tune ASR content thresholds;
- tune special-mora thresholds;
- tune F0 thresholds;
- delete legacy evaluator reliability caps yet;
- create a Japanese vowel/consonant interval classifier;
- treat UME-JRF as commercially redistributable product data;
- collect or fabricate human ratings.

## 12. Work split: repository vs local/Codex/human execution

### Repository work that does not require Codex

- score/product semantics and versioning;
- history/personalization safety;
- DTW-rhythm implementation;
- multi-reference aggregation/provenance;
- conversation product routing;
- research manifests and split/leakage guards;
- rating schemas and validators;
- benchmark/statistics code and CI.

### External/local work that cannot be completed from repository access alone

- accepting external dataset access/license terms and obtaining UME-JRF;
- recording/obtaining several consented native references per production target;
- producing trusted/manual or locally forced-aligned reference boundaries when the exact audio is not present in the repo;
- recruiting listeners and collecting real criterion ratings;
- running large WavLM/CTC/MFA batches when licensed audio/model checkpoints exist only on a local machine.

If Codex is used for those tasks, it should execute a frozen repository protocol rather than redesign the scoring model.

## 13. Next gates

### Gate A — light CI

All score-contract, history, conversation-mode, rhythm, multi-reference, reference, content, manifest, split, and human-rating validation tests must pass.

### Gate B — reference bank

Finish true boundary provenance migration for production/demo fixed targets and assign stable reference IDs.

### Gate C — external research execution

After UME-JRF is legally obtained locally, map it into the normalized manifest and run the frozen evidence export/benchmark. Heavy model execution may be delegated to a local worker/Codex only as an execution task.

### Gate D — human criteria

Run pronunciation-accuracy and consumer-criterion studies with held-out speaker/target evaluation.

### Gate E — product calibration

Only after criterion evidence should score mappings, weights, confidence shrinkage, or shadow promotion be changed.
