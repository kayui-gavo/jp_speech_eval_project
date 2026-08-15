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

The implementation uses only valid moving-average centers so edge padding does not create artificial irregularity in a perfectly linear warp path.

A global frame-duration ratio is reported separately. A uniformly slower/faster utterance can therefore have low local tempo irregularity instead of being misclassified as locally unstable rhythm.

Everything remains:

- shadow only;
- `score_mapped=false`;
- `product_calibrated=false`;
- not user-facing.

## 5. Interval distortion boundary

`interval_distortion_from_dtw_path()` is implemented only as a pure metric over externally supplied frame labels:

- vowel;
- consonant;
- silence.

No Japanese interval classifier is claimed. Promotion is blocked until such a classifier is validated independently.

This avoids importing the paper's TIMIT-trained auxiliary classifiers as if they were automatically valid for Japanese L2 speech.

## 6. Human criterion split

New protocol:

`reports/CONSUMER_CRITERION_PROTOCOL_V2.md`

The existing frozen pronunciation-accuracy protocol is preserved. The new protocol separately defines human criteria for:

- clarity/comprehensibility;
- fluency;
- rhythm naturalness;
- phrase/sentence intonation naturalness;
- independent analyzability.

Pronunciation accuracy is not renamed clarity.

## 7. External research benchmark boundary

New plan:

`reports/UME_JRF_BENCHMARK_PLAN_V1.md`

New normalized research manifest support:

- `data/research_eval/ume_jrf_manifest_template.csv`
- `scripts/validate_research_manifest.py`

The repository does not bundle UME-JRF. Local licensed data is mapped into a normalized manifest, then frozen benchmark code consumes that manifest. This minimizes corpus-specific code and keeps licensing/access separate from product runtime.

## 8. What is intentionally unchanged

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
- treat UME-JRF as commercially redistributable product data.

## 9. Next gates

### Gate A — light CI

All score-contract, history, rhythm, reference, content, and manifest tests must pass.

### Gate B — reference bank

Finish true boundary provenance migration for production/demo fixed targets.

### Gate C — external research execution

After UME-JRF is legally obtained locally, map it into the normalized manifest and run the frozen evidence export/benchmark. Heavy model execution may be delegated to a local worker/Codex only as an execution task.

### Gate D — human criteria

Run pronunciation-accuracy and consumer-criterion studies with held-out speaker/target evaluation.

### Gate E — product calibration

Only after criterion evidence should score mappings, weights, confidence shrinkage, or shadow promotion be changed.
