# Japanese phone-GOP Stage-0 status v1

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Human recording gate: **BLOCKED**

## Why this status file exists

Human recording should not be used to discover basic model/API/metric bugs. Stage 0 therefore uses synthetic tests, bundled reference audio, pinned models, target-frontend snapshots, and automatic perturbations before any new human recording is requested.

## Real pinned-model smoke test already completed

An earlier GitHub Actions model-preflight run on commit `2c51cd8e0a290d8c67fdd3bc01fd9f5cb718e2cd` successfully loaded the pinned backend:

- model: `prj-beatrice/japanese-hubert-base-phoneme-ctc-v4`
- revision: `f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99`
- target frontend: `pyopenjtalk-plus 0.4.1.post8`
- bundled audio: `assets/reference_cache/ramen_kudasai_aivis.ref.wav`
- correct target phones: `r a a m e N o k u d a s a i`
- wrong target: `コーヒーをください。`

The old report incorrectly allowed the human gate to open when the backend smoke test passed. Current preflight code has corrected that semantic bug: backend-preflight PASS is now necessary but **not sufficient** for human recording.

## Useful positive findings

The model/backend plumbing is real, not only a synthetic-logit prototype:

- all 14 canonical phones received finite evidence;
- pilot target inventory was covered by the backend vocabulary;
- no `pau`/`sil` token leaked into segmental competitors after Japanese-specific filtering;
- correct-target sequence log posterior per frame: `-0.3392543268`;
- deliberately wrong-target sequence log posterior per frame: `-0.8654156102`;
- correct-minus-wrong gap: `+0.5261612834`;
- 0.8x amplitude gain changed the sequence evidence by about `0.00193356`;
- 1.2x amplitude gain changed it by about `0.00001364`.

Thus the correct-vs-wrong target gap was orders of magnitude larger than the mild gain perturbation on this one bundled reference.

These observations are engineering sanity evidence only. They are not L2 pronunciation validity and do not justify a `/100` mapping.

## Critical finding: forced CTC phone frames are too peaky to be the main clarity criterion

On the bundled correct target, `single_frame_support_ratio = 0.8571428571`.

This means most canonical phones received only one Viterbi CTC support frame. The unconstrained greedy phone sequence was only:

`k u d a s a i`

rather than the full canonical `r a a m e N o k u d a s a i`.

Therefore the current frame-local forced-Viterbi GOP/logit features are retained as diagnostics, but they should **not** be promoted as the primary Japanese clarity backbone for this model. CTC peak timing is not a physical phone segmentation.

## Critical finding: naive leave-one-phone-out deletion interpretation is insufficient

On the same clean reference, the exploratory canonical-minus-deleted sequence log-posterior-per-frame values for the first phrase were negative, including approximately:

- `r`: `-0.08045`
- first `a`: `-0.06573`
- second `a`: `-0.06573`
- `m`: `-0.03766`
- `e`: `-0.03144`
- `N`: `-0.03845`
- `o`: `-0.02266`

Later phones in `ください` became positive.

A single deletion ratio by itself is therefore not a sufficient deletion detector. It also cannot distinguish which member of identical repeated phones was deleted, because deleting either member yields the same alternate transcription.

The new implementation direction follows segmentation-free GOP feature work instead: retain the canonical sequence log posterior (LPP) together with a full vector of log-posterior ratios (LPR) for all one-phone substitutions plus deletion at each target position. This allows the evidence to be interpreted jointly rather than inventing a threshold on one leave-one-out number.

## Current alignment-free implementation

`src/jp_speech_eval/segmentation_free_gop.py` now implements an enumerated, transparent SD feature extractor:

- exact CTC posterior for the canonical sequence;
- exact CTC posterior for every one-phone substitution at position `i`;
- exact CTC posterior for deletion at position `i`;
- canonical/alternative log-posterior ratios;
- best noncanonical alternative and its type;
- scalar canonical-vs-summed-SD-alternative log ratio;
- no forced phone boundaries;
- no insertion in the fixed feature vector;
- no `Occ(i)` normalization yet;
- no `/100` mapping.

The implementation is explicitly named `enumerated_fgop_ctc_sf_sd_features_v1`; it does not claim to be the optimized graph implementation or the normalized method from the paper.

## Remaining Stage-0 gates before human recording

1. Latest full repository tests must pass in the fresh Python 3.11 CI environment.
2. The pyopenjtalk-plus target-phone snapshot for every planned pilot phrase must be generated, inspected, frozen, and regression-tested.
3. The new alignment-free SD features must pass a fresh real-model bundled-audio preflight.
4. Frame-local mean/max competitor provenance must remain explicit; CTC support duration must never be interpreted as a physical phone duration.
5. Existing-data automatic benchmarks should be run wherever referenced audio is locally available; missing external corpus files must skip rather than trigger new recording requests.
6. A batch analysis/report path must exist before a human records anything.
7. Human recording remains explicitly blocked until a Stage-0 readiness report promotes the gate.

## Product policy

No code in this Stage-0 line may directly alter the C-end clarity `/100` score. The current work is evidence validation. Only after Japanese L2 behavior is demonstrated should phone evidence be fused with WavLM/reference evidence and ASR intelligibility for the C-end `明瞭さ` dimension.
