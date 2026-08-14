# Japanese phone-GOP Stage-0 status v1

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Human recording gate: **BLOCKED**

## Why this status file exists

Human recording must not be used to discover basic model/API/metric bugs. Stage 0 therefore uses synthetic tests, bundled/reference audio, pinned models, target-frontend snapshots, automatic perturbations and automatic batch analysis before any new human recording is requested.

## Fresh CI status

The current Stage-0 branch has passed a clean Python 3.11 CI run with **228 tests passed and 6 deprecation/audio-backend warnings**. The same heavy run successfully loaded the pinned Japanese phone-CTC backend and completed both the conventional backend smoke checks and the enumerated segmentation-free SD feature preflight.

This is code/engineering evidence only. It does not open the human recording gate.

## Real pinned-model smoke test

GitHub Actions successfully loaded:

- model: `prj-beatrice/japanese-hubert-base-phoneme-ctc-v4`
- revision: `f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99`
- target frontend: `pyopenjtalk-plus 0.4.1.post8`
- bundled audio: `assets/reference_cache/ramen_kudasai_aivis.ref.wav`
- correct target phones: `r a a m e N o k u d a s a i`
- wrong target: `コーヒーをください。`

The old preflight semantics incorrectly allowed the human gate to open when this backend smoke test passed. Current code fixes that: backend-preflight PASS is necessary but **not sufficient** for human recording.

## Useful positive findings

The model/backend plumbing is real, not only a synthetic-logit prototype:

- all 14 canonical phones received finite evidence;
- pilot target inventory was covered by the backend vocabulary;
- no `pau`/`sil` leaked into segmental competitors after Japanese-specific filtering;
- correct-target sequence log posterior/frame: approximately `-0.3392540`;
- wrong-target sequence log posterior/frame: approximately `-0.8654185`;
- correct-minus-wrong gap: approximately `+0.5261644`;
- 0.8x amplitude gain changed sequence evidence by only about `0.00194`;
- 1.2x amplitude gain changed sequence evidence by only about `0.000012`.

Thus target mismatch had far greater effect than mild gain change on this bundled reference. This is engineering sanity evidence, not L2 pronunciation validity or `/100` calibration.

## Critical finding: forced CTC support frames are too peaky to be the main clarity criterion

On the bundled correct target, `single_frame_support_ratio = 0.8571428571`. The unconstrained greedy phone sequence was only `k u d a s a i`, rather than the full canonical `r a a m e N o k u d a s a i`.

Therefore frame-local forced-Viterbi GOP/logit features remain diagnostics. CTC support frames are not physical phone boundaries, and the generic extractor records mean-vs-max competitor provenance separately and labels support duration explicitly.

## Critical finding: naive leave-one-phone-out deletion interpretation is insufficient

On the clean reference, exploratory canonical-minus-deleted sequence evidence was negative for several phones in the first phrase. Identical repeated phones are also inherently ambiguous under one-position deletion because deleting either identical member can create the same transcript.

A standalone leave-one-out deletion threshold is therefore not accepted as a detector.

## Fresh segmentation-free SD preflight result

`src/jp_speech_eval/segmentation_free_gop.py` now evaluates exact CTC sequence evidence without forcing phone boundaries. On the same bundled Aivis reference, the correct target remained globally much better than the deliberately wrong target, but local SD alternatives exposed an important limitation.

For the correct target:

- canonical sequence log posterior: approximately `-18.659`;
- 14 target phones;
- 40 segmental alternative-phone classes;
- **7 of 14 positions had a noncanonical SD alternative with higher posterior than the canonical sequence**;
- the strongest noncanonical preference was about `-4.425` log-posterior-ratio units relative to the canonical sequence.

The problematic positions were concentrated in the first phrase `ラーメンを`: `r`, the two `a` positions, `m`, `e`, `N`, and `o` tended to prefer a deletion alternative. The later `ください` phones behaved much more plausibly.

For the deliberately wrong target, noncanonical alternatives were still more strongly preferred overall, so the model retains useful whole-target discrimination. However, **clean-reference local phone evidence is not yet trustworthy enough for learner-facing phone feedback**.

This pattern is not explained by VAD trimming: the preflight speech region covered the full bundled utterance. A plausible unresolved explanation is backend/reference-domain interaction (for example, this Aivis TTS voice versus the phone model's training distribution), but that is an inference to test, not a conclusion.

Therefore the Beatrice backend remains a valid research candidate, but its local SD/GOP evidence is **not promoted** to the C-end `明瞭さ` score.

## Current alignment-free implementation

`src/jp_speech_eval/segmentation_free_gop.py` implements an enumerated SD feature extractor inspired by segmentation-free GOP feature work:

- exact CTC posterior of the canonical sequence (LPP);
- exact CTC posterior for every one-phone substitution at each position;
- exact deletion posterior;
- canonical/alternative log-posterior ratios (LPR);
- best noncanonical alternative/type;
- canonical-vs-summed-SD-alternative log ratio;
- no forced phone boundaries;
- no insertion in the fixed feature vector;
- no `Occ(i)` activation normalization yet;
- no optimized alternative graph yet;
- no `/100` mapping.

The method name is deliberately `enumerated_fgop_ctc_sf_sd_features_v1`; it does not claim full normalized/optimized paper equivalence.

## Target frontend is frozen before recording

A clean GitHub Actions environment generated the current pilot target set with `pyopenjtalk-plus 0.4.1.post8`: 21 unique target texts, no frontend ambiguity and no frontend warnings. The exact text/kana/phone/mora snapshot is frozen in:

`data/audit/phone_target_snapshot_v1.csv`

A regression test regenerates those targets and requires exact equality. This prevents a dictionary/frontend update from silently changing canonical phones after recording starts.

## Human-time protection automation

The repository contains:

- `scripts/generate_phone_target_snapshot.py` — target/frontend check without loading an acoustic model;
- `scripts/run_phone_gop_preflight.py` — pinned backend, correct/wrong target and gain checks on bundled audio;
- `scripts/run_segmentation_free_gop_preflight.py` — alignment-free SD features on bundled audio;
- `scripts/benchmark_phone_gop_existing_data.py` — runs only audio that already exists in the current research workspace and skips unavailable external corpus paths;
- `src/jp_speech_eval/phone_gop_batch_analysis.py` and `scripts/analyze_phone_gop_manual_batch.py` — future grouped N1/N2/E analysis, clean-repeat variance, intended-error delta, competitor match and neighbor leakage without manually opening dozens of JSON files.

Missing future clip files are reported/skipped; missing files never trigger an automatic request for a replacement human recording.

## Current Stage-0 gates before human recording

1. **PASS** — latest full repository tests in a fresh Python 3.11 CI environment.
2. **PASS** — frozen `pyopenjtalk-plus 0.4.1.post8` target snapshot regression.
3. **PASS** — pinned Beatrice backend loads and separates correct/wrong target much more than mild gain perturbations.
4. **NOT PASS** — clean bundled TTS local phone evidence is still pathological for part of the utterance; local clarity use is blocked.
5. **PENDING** — run the same automatic evidence on existing *human native* audio (preferably JVS/other already available research audio) before spending user recording time.
6. **PENDING** — compare at least one second Japanese phone-CTC backend if the human-native check shows the same pathology, so a Beatrice-specific failure is not mistaken for a general GOP failure.
7. **PASS (infrastructure)** — grouped automatic batch analysis exists before any manual battery starts.
8. **BLOCKED by design** — a separate Stage-0 readiness decision must explicitly promote the human gate. Backend smoke PASS alone can never do so.

## Next automatic work

The next useful experiment is not a user recording. It is:

1. run `benchmark_phone_gop_existing_data.py` wherever the already-referenced JVS/JANON files are present;
2. prioritize native human JVS clips to distinguish TTS-domain mismatch from backend-localization failure;
3. if local phone alternatives remain pathological, benchmark a second compact Japanese phone-CTC model before reconsidering the metric;
4. only after that decide whether the minimum human pilot is justified.

## Product policy

No Stage-0 phone-GOP code directly changes the C-end clarity `/100`. Phone evidence remains shadow-only. Only after Japanese L2 behavior is demonstrated should it be fused with WavLM/reference evidence and ASR intelligibility for `明瞭さ`.
