# Japanese phone-GOP Stage-0 status v1

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Human recording gate: **BLOCKED**

## Why this status file exists

Human recording must not be used to discover basic model/API/metric bugs. Stage 0 therefore uses synthetic tests, bundled/reference audio, pinned models, target-frontend snapshots, automatic perturbations and automatic batch analysis before any new human recording is requested.

The most important Stage-0 rule is now explicit: **raw segmentation-free CTC feature signs are diagnostics/features, not a pronunciation-error decision rule.** A clean phone is not required to beat every substitution/deletion alternative in isolation.

## Fresh CI status

The current Stage-0 branch has passed clean Python 3.11 CI after the phone-candidate adapter additions. Ordinary tests have reached **233 passed with 6 existing deprecation/audio-backend warnings**. Heavy model preflight has successfully run the pinned Beatrice backend and a separately pinned DistilHuBERT dual-CTC phone candidate on bundled audio.

This is code/engineering evidence only. It does not open the human recording gate.

## Literature correction: FGOP-SF features are not per-alternative thresholds

The implementation is aligned conceptually with:

- Xinwei Cao, Zijian Fan, Torbjørn Svendsen, Giampiero Salvi, **“Segmentation-Free Goodness of Pronunciation”**, arXiv:2507.16838, accepted/published in IEEE TASLP 2026, DOI 10.1109/TASLPRO.2026.3654852.

That work distinguishes a scalar segmentation-free GOP from **feature vectors derived from segmentation-free CTC sequence evidence**. The feature approach retains canonical log posterior (LPP) and log-posterior ratios (LPRs) against substitution/deletion alternatives, then uses those jointly in a downstream pronunciation-assessment/MDD model. Its normalized feature form also includes an activation/occupancy statistic `Occ(i)` computed from the alternative-graph forward variables.

Therefore our earlier provisional interpretation—treating `best_noncanonical_log_posterior_ratio < 0` on clean speech as a Stage-0 failure by itself—was too strong. The code may still report how many positions have a higher-posterior noncanonical alternative, but this is **descriptive evidence only**. It must not be interpreted as “that phone is wrong”, used as a direct clean-speech pass/fail criterion, or mapped to `/100`.

Our current enumerated extractor remains an intentionally transparent approximation of the paper's feature family. It does **not** yet implement the paper's joint alternative graph or its `Occ(i)` normalization, and it does not claim normalized-method equivalence.

## Real pinned Beatrice smoke test

GitHub Actions successfully loaded:

- model: `prj-beatrice/japanese-hubert-base-phoneme-ctc-v4`
- revision: `f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99`
- target frontend: `pyopenjtalk-plus 0.4.1.post8`
- bundled audio: `assets/reference_cache/ramen_kudasai_aivis.ref.wav`
- correct target phones: `r a a m e N o k u d a s a i`
- wrong target: `コーヒーをください。`

Useful engineering findings:

- all 14 canonical phones received finite evidence;
- pilot target inventory was covered by the backend vocabulary;
- no `pau`/`sil` leaked into segmental competitors after Japanese-specific filtering;
- correct-target sequence log posterior/frame: approximately `-0.3392540`;
- wrong-target sequence log posterior/frame: approximately `-0.8654185`;
- correct-minus-wrong gap: approximately `+0.5261644`;
- mild 0.8x/1.2x amplitude gain perturbations were tiny relative to the correct-vs-wrong target gap.

Thus target mismatch had much greater effect than mild gain change on this bundled reference. This is engineering sanity evidence, not L2 pronunciation validity or `/100` calibration.

## Forced CTC support remains unsuitable as physical phone segmentation

On the bundled correct target, `single_frame_support_ratio = 0.8571428571`. The unconstrained greedy phone sequence was only `k u d a s a i`, rather than the full canonical `r a a m e N o k u d a s a i`.

Therefore frame-local forced-Viterbi GOP/logit features remain diagnostics. CTC support frames are not physical phone boundaries or physical phone durations. The generic extractor records mean-vs-max competitor provenance separately and labels support duration explicitly.

## Enumerated segmentation-free SD preflight: what it does and does not mean

`src/jp_speech_eval/segmentation_free_gop.py` evaluates exact CTC posterior of the canonical sequence and enumerated one-phone substitution/deletion alternatives without forced phone boundaries.

On the Aivis `ラーメンをください` reference, Beatrice produced 7/14 positions where one noncanonical SD alternative had higher sequence posterior than the canonical sequence, concentrated in `ラーメンを`. The largest negative best-alternative LPR was approximately `-4.425`.

After literature alignment, the interpretation is now:

- this pattern **does not prove seven pronunciation errors**;
- it **does not by itself invalidate** the feature family;
- it demonstrates why raw LPR sign must not be promoted directly to local learner feedback;
- the entire `{LPP, LPR(·)}` vector should be treated as a research feature pending labeled criterion validation;
- identical repeated phones remain intrinsically ambiguous for one-position deletion alternatives when the resulting sequence is identical.

Whole-target discrimination remains useful: correct target is much better explained than the deliberately wrong target. Local phone correctness, however, remains **unmapped and unvalidated**.

## Independent compact candidate: DistilHuBERT dual CTC

A second Japanese phone-CTC backbone has now been integrated as a pinned shadow adapter:

- model: `TylorShine/distilhubert-hiragana-ctc`
- immutable revision: `01ffc3e5b0e49ba34180d50c48ea4111aa041cfd`
- product role: **research/shadow only**
- score mapping: **disabled**

The candidate repository currently omits `preprocessor_config.json`. To avoid mutable/arbitrary defaults, the adapter explicitly reproduces the pinned base `ntu-spml/distilhubert` waveform-preprocessing contract: 16 kHz, one waveform feature, zero padding, no waveform normalization, no attention mask. The provenance is recorded in every result.

On the same bundled Aivis reference:

- correct canonical CTC log posterior: approximately `-32.2533`;
- deliberately wrong target canonical log posterior: approximately `-44.9798`;
- correct-minus-wrong sequence log-posterior gap: approximately `+12.7264`;
- mild 0.8x/1.2x gain changes altered the canonical value only slightly;
- 7 target positions again had a higher-posterior noncanonical alternative.

The last item is **not a direct failure criterion**. Its main value is that two independently trained Japanese phone-CTC backbones reproduce similar raw LPR structure on the same TTS sample, reinforcing the decision not to interpret an individual LPR sign as phone correctness.

A second WavLM dual-CTC backbone is also being benchmarked as a research-only comparison. Because its licensing is less attractive for a C-end default, it is not a preferred product dependency even if its research behavior is useful.

## Current alignment-free implementation

`src/jp_speech_eval/segmentation_free_gop.py` implements `enumerated_fgop_ctc_sf_sd_features_v1`:

- exact CTC posterior of the canonical sequence (LPP);
- exact CTC posterior for every one-phone substitution at each position;
- exact deletion posterior;
- canonical/alternative log-posterior ratios (LPR);
- best noncanonical alternative/type as a diagnostic;
- canonical-vs-summed-SD-alternative log ratio;
- no forced phone boundaries;
- no insertion in the fixed feature vector;
- no `Occ(i)` activation/occupancy normalization yet;
- no optimized/joint alternative graph yet;
- no downstream learned MDD classifier/regressor yet;
- no `/100` mapping.

The method name deliberately does not claim full normalized/optimized paper equivalence.

## Target frontend is frozen before recording

A clean GitHub Actions environment generated the current pilot target set with `pyopenjtalk-plus 0.4.1.post8`: 21 unique target texts with exact text/kana/phone/mora snapshot frozen in:

`data/audit/phone_target_snapshot_v1.csv`

A regression test regenerates those targets and requires exact equality. This prevents dictionary/frontend updates from silently changing canonical phones after recording starts.

## Human-time protection automation

The repository contains:

- `scripts/generate_phone_target_snapshot.py` — target/frontend regression candidate generation;
- `scripts/run_phone_gop_preflight.py` — pinned Beatrice correct/wrong target and gain controls;
- `scripts/run_segmentation_free_gop_preflight.py` — alignment-free enumerated SD features;
- `scripts/resolve_phone_ctc_candidate_revision.py` — resolves immutable Hugging Face revisions before remote custom code is trusted;
- `scripts/run_dual_ctc_candidate_preflight.py` — pinned alternative phone-CTC correct/wrong/gain shadow preflight;
- `scripts/benchmark_phone_gop_existing_data.py` — runs only audio already present in the research workspace and skips unavailable corpus paths;
- `src/jp_speech_eval/phone_gop_batch_analysis.py` and `scripts/analyze_phone_gop_manual_batch.py` — grouped N1/N2/E analysis infrastructure for later labeled evaluation.

Missing external corpus files are skipped. No script automatically asks for replacement/new human recordings.

## Current Stage-0 gates before human recording

1. **PASS** — full repository tests in fresh Python 3.11 CI.
2. **PASS** — frozen `pyopenjtalk-plus 0.4.1.post8` target snapshot regression.
3. **PASS** — pinned Beatrice backend loads, yields finite evidence, separates correct/wrong target, and is much less sensitive to mild gain than target mismatch.
4. **PASS (infrastructure)** — segmentation-free `{LPP,LPR}` feature extraction exists without forced phone boundaries.
5. **PASS (scientific correction)** — negative individual LPR is no longer treated as a direct pronunciation-error or clean-speech failure criterion.
6. **PASS** — a second pinned compact Japanese phone-CTC backbone (DistilHuBERT dual CTC) can run through the same shadow preflight.
7. **PENDING** — implement/verify the paper-faithful alternative-graph/`Occ(i)` normalization before claiming FGOP-SF-Norm equivalence.
8. **PENDING** — run the feature family on already-available native/learner human audio wherever corpus files exist; missing paths skip.
9. **PENDING** — obtain a genuinely labeled criterion for local pronunciation correctness before learning a phone error classifier/regressor or interpreting features as correctness.
10. **BLOCKED by design** — a separate readiness decision must explicitly promote the human recording gate. Backend/model smoke PASS alone can never do so.

## Next automatic work

The machine-only priority is now:

1. finish the Beatrice / DistilHuBERT / WavLM dual-CTC comparison on bundled audio;
2. inspect/port the published segmentation-free GOP reference implementation if accessible, especially alternative-graph forward computation and `Occ(i)` normalization;
3. regression-test any graph implementation against exact enumerated sequence probabilities on tiny synthetic cases;
4. run those features over existing human native/learner data wherever already available;
5. only after the metric semantics and labeled criterion are clear decide whether any new human recording is worth the user's time.

## Product policy

No Stage-0 phone-GOP code directly changes the C-end clarity `/100`. Phone evidence remains shadow-only. It may later be fused with WavLM/reference evidence and ASR intelligibility for `明瞭さ` only after Japanese L2 criterion validity is demonstrated.
