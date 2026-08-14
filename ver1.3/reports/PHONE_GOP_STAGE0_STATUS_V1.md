# Japanese phone-GOP Stage-0 status v1

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Human recording gate: **BLOCKED**
Product score impact: **NONE — PHONE EVIDENCE REMAINS SHADOW/RESEARCH ONLY**

## Why this status file exists

Human recording must not be used to discover basic model/API/metric bugs. Stage 0 therefore exhausts synthetic tests, bundled/reference audio, pinned models, frozen target-frontends, automatic perturbations, published-algorithm regression tests and already-public native audio before asking anyone to record new material.

The central scientific rule is explicit: **raw segmentation-free CTC features are features, not pronunciation-error decisions.** An individual negative LPR, a low `Occ(i)`, or a noncanonical sequence having higher posterior must never be translated directly into “this phone is wrong” or a `/100` score.

## Current CI baseline

The last completed normalized-feature CI run passed:

- Python 3.11 clean environment;
- **238 tests passed**;
- 6 existing deprecation/audio-backend warnings;
- pinned Beatrice, DistilHuBERT and WavLM model preflights all succeeded;
- paper-aligned SD alternative-graph/`Occ(i)` features were finite on correct and deliberately wrong targets for all three backbones.

Later JVS native-anchor helper tests are being added on top of this baseline; they do not change product scoring.

## Literature-aligned feature semantics

Primary reference:

- Xinwei Cao, Zijian Fan, Torbjørn Svendsen, Giampiero Salvi, **“Segmentation-Free Goodness of Pronunciation”**, arXiv:2507.16838 / IEEE TASLP 2026, DOI `10.1109/TASLPRO.2026.3654852`.
- Authors' public implementation: `frank613/CTC-based-GOP/taslpro26`.

The paper/code distinguishes several related objects:

1. canonical CTC sequence log posterior (`LPP`);
2. log-posterior ratios against deletion/substitution alternatives (`LPR` feature vector);
3. an SD alternative-graph denominator for segmentation-free GOP;
4. normalized forward occupancy/activation `Occ(i)` from the wildcard state.

The project now keeps those semantics separate. `Occ(i)` is graph occupancy, **not a physical phone duration**. The `{LPP,LPR,Occ}` family requires a labeled downstream criterion before it can be interpreted as learner phone correctness.

## Two alignment-free implementations now coexist deliberately

### Transparent enumerated feature extractor

`src/jp_speech_eval/segmentation_free_gop.py`

Method: `enumerated_fgop_ctc_sf_sd_features_v1`

It computes:

- exact canonical CTC posterior;
- every one-phone substitution posterior;
- one-phone deletion posterior;
- LPRs;
- best noncanonical alternative as a diagnostic;
- canonical-vs-summed-enumerated-SD ratio.

It requires no forced phone boundaries and has no product mapping.

### Published SD normalized-forward implementation

`src/jp_speech_eval/segmentation_free_gop_norm.py`

Method: `paper_sd_norm_forward_v1`

It independently re-implements the normalized arbitrary-token forward recursion from the authors' public TASLP-2026 code and returns:

- SD alternative-graph log posterior;
- `GOP-SF-SD`-style log ratio;
- `Occ(i)`.

Regression tests freeze the public-algorithm outputs on deterministic synthetic probability grids, including repeated-phone context. This closes the earlier “`Occ(i)` not implemented” Stage-0 gap.

## Forced CTC frames remain diagnostic only

On the bundled Aivis `ラーメンをください` reference, Beatrice has `single_frame_support_ratio ≈ 0.857`, and its unconstrained greedy phone sequence covers only part of the canonical sequence.

Therefore forced-Viterbi support frames are **not** accepted as physical phone segmentation or duration. They remain secondary diagnostics only.

## Three-backbone bundled-audio result

Bundled audio:

`assets/reference_cache/ramen_kudasai_aivis.ref.wav`

Correct target:

`ラーメンをください。`

Deliberately wrong target:

`コーヒーをください。`

Pinned backbones:

- Beatrice: `prj-beatrice/japanese-hubert-base-phoneme-ctc-v4@f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99`
- DistilHuBERT dual CTC: `TylorShine/distilhubert-hiragana-ctc@01ffc3e5b0e49ba34180d50c48ea4111aa041cfd`
- WavLM dual CTC: `TylorShine/wavlm-base-plus-hiragana-ctc@47fa985035342365bcec4948bd821aaf58dd778a`

Within each backbone, the correct phone sequence is substantially better supported than the unrelated target. Raw sequence log-posterior gaps are approximately:

- Beatrice: `+28.939`;
- DistilHuBERT: `+12.726`;
- WavLM: `+8.668`.

These raw values are **not cross-model comparable scales**. Their useful role is within-model target discrimination and perturbation comparison.

Mild `0.8x` / `1.2x` amplitude gain changes are far smaller than the correct-vs-wrong target effect on this sample.

## Published normalized-forward result on the three backbones

The paper-aligned SD graph/`Occ(i)` path now runs successfully on all three models.

For the correct vs wrong target respectively, mean normalized SD-GOP values are approximately:

- Beatrice: `-1.484` vs `-4.222`;
- DistilHuBERT: `-2.425` vs `-4.254`;
- WavLM: `-2.735` vs `-3.801`.

Again, absolute values are not compared between models. Within every backbone, the known correct target has the less-negative aggregate normalized evidence.

All `Occ(i)` values were finite; correct-target ranges were approximately:

- Beatrice: `1.35e-6` to `2.385`;
- DistilHuBERT: `4.08e-11` to `2.529`;
- WavLM: `4.58e-11` to `2.752`.

Near-zero occupancy is **not** called a phone error and is not called a duration failure.

A useful structural sanity check also appears automatically: the correct and wrong targets share the suffix `ください`, and normalized evidence for the shared suffix is nearly unchanged while the differing prefix carries almost all of the target-discrimination effect. This supports the feature implementation's target-local behavior, but still does not establish learner phone correctness.

## Independent compact candidate

DistilHuBERT dual CTC is retained as the preferred compact second engineering backbone. Its repository lacks `preprocessor_config.json`, so the adapter explicitly reproduces the pinned base `ntu-spml/distilhubert` waveform preprocessing contract rather than accepting mutable/default preprocessing. Provenance is stored in the research result.

WavLM remains a useful research comparison but is not presumed to be a product dependency.

## Frozen Japanese target frontend

The planned Stage-0 target set is frozen with `pyopenjtalk-plus 0.4.1.post8` in:

`data/audit/phone_target_snapshot_v1.csv`

The snapshot fixes exact text/kana/phone/mora sequences for 21 unique targets and is regression-tested so later dictionary/frontend changes cannot silently change canonical labels.

## Official JVS native-anchor preflight now replaces a new-user-recording request

The next automatic test uses the three small human-native sample clips linked directly by the official JVS corpus project page (`jvs001`, `jvs002`, `jvs003`, same `VOICEACTRESS100_001` text).

New scripts:

- `scripts/download_official_jvs_samples.py`
- `scripts/run_official_jvs_phone_ctc_anchor_preflight.py`

Policy:

- JVS audio is downloaded ephemerally into `outputs/`;
- downloaded WAV files are deleted before artifact collection;
- audio is never committed to this repository and never uploaded as a workflow artifact;
- only hashes/metadata and derived JSON diagnostics are retained;
- no new user recording is requested.

The native-anchor criterion is intentionally modest: for each of the three human native speakers and each of the three phone-CTC backbones, compare the known target sequence with a deterministic **same-phone-count rotated phone sequence**. This removes the trivial phone-count/length confound of comparing targets with different numbers of phones. JVS001 also receives `0.8x`/`1.2x` gain controls.

This is still not a local phone-error benchmark because these samples do not provide phone-correctness labels.

## Human-time protection automation

Current machine-only infrastructure includes:

- frozen target frontend generation/regression;
- pinned Beatrice backend preflight;
- pinned alternative-model revision resolution;
- DistilHuBERT/WavLM dual-CTC adapters;
- enumerated alignment-free LPP/LPR extraction;
- published SD normalized-forward + `Occ(i)` extraction;
- automatic grouped phone-GOP batch analysis;
- existing-data-only benchmark runner with skip-on-missing behavior;
- official-JVS ephemeral downloader and native-anchor benchmark.

No missing corpus file automatically turns into a request for a new human recording.

## Current Stage-0 gates before human recording

1. **PASS** — full repository tests in fresh Python 3.11 CI.
2. **PASS** — frozen `pyopenjtalk-plus 0.4.1.post8` target snapshot regression.
3. **PASS** — pinned Beatrice backend loads and separates correct/wrong target on bundled audio.
4. **PASS** — alignment-free enumerated `{LPP,LPR}` feature extraction.
5. **PASS** — individual LPR sign is explicitly prohibited as a direct error rule.
6. **PASS** — pinned DistilHuBERT and WavLM independent phone-CTC comparisons run successfully.
7. **PASS** — published SD normalized-forward/`Occ(i)` algorithm implemented, regression-tested, and finite on all three real model backbones.
8. **RUNNING/NEXT** — official JVS three-speaker native-anchor test across the three backbones.
9. **PENDING** — run the feature family over already-available learner/native corpora wherever the files are actually present.
10. **PENDING** — obtain a genuinely labeled criterion before learning a local phone-correctness classifier/regressor or mapping any phone feature to a user score.
11. **BLOCKED by design** — human recording requires a separate explicit readiness promotion; model smoke tests alone can never open the gate.

## Product policy

No Stage-0 phone-GOP code directly changes C-end `明瞭さ`, overall score, or any `/100` mapping. Phone evidence remains shadow-only. Product promotion requires Japanese L2 criterion validity, not merely model availability or plausible-looking acoustic features.
