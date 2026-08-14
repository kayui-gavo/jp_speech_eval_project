# Japanese phone-GOP Stage-0 status v1

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Human recording gate: **BLOCKED**
Product score impact: **NONE — PHONE EVIDENCE REMAINS SHADOW/RESEARCH ONLY**

## Stage-0 principle

Human recording must not be used to discover basic model/API/metric bugs. Stage 0 first exhausts synthetic tests, bundled/reference audio, pinned models, frozen target frontends, published-algorithm regression tests, public human-native anchors and already-existing learner/native corpora.

The core scientific rule is now enforced in code and reports:

> segmentation-free CTC outputs are **features**, not pronunciation-error decisions.

An individual negative LPR, a low `Occ(i)`, a forced CTC support frame, or a noncanonical alternative with higher posterior must never be translated directly into “this phone is wrong” or a `/100` score.

## Current code baseline

The latest completed ordinary CI before the JVS-byte freeze passed:

- Python 3.11 clean environment;
- **246 tests passed**;
- 6 existing deprecation/audio-backend warnings;
- no product-score changes.

Three additional no-network tests now cover frozen JVS source metadata/drift rejection. The heavy run triggered by this report verifies them together with the current criterion-ready feature bundles and the frozen JVS downloads.

## Literature-aligned feature stack

Primary methodological reference:

- Xinwei Cao, Zijian Fan, Torbjørn Svendsen, Giampiero Salvi, **“Segmentation-Free Goodness of Pronunciation”**, arXiv:2507.16838 / IEEE TASLP 2026, DOI `10.1109/TASLPRO.2026.3654852`.
- authors' public implementation: `frank613/CTC-based-GOP/taslpro26`.

The project now keeps the relevant concepts separate:

1. canonical CTC sequence log posterior (`LPP`);
2. substitution/deletion log-posterior ratios (`LPR` feature vector);
3. SD alternative-graph denominator;
4. graph-derived normalized `GOP-SF-SD` feature;
5. normalized wildcard-state occupancy/activation `Occ(i)`.

`Occ(i)` is **not physical phone duration**.

### Transparent enumerated extractor

`src/jp_speech_eval/segmentation_free_gop.py`

Method: `enumerated_fgop_ctc_sf_sd_features_v1`

It computes exact canonical, all one-phone substitution, and deletion sequence posteriors without forced phone boundaries.

### Published normalized-forward extractor

`src/jp_speech_eval/segmentation_free_gop_norm.py`

Method: `paper_sd_norm_forward_v1`

It independently re-implements the normalized arbitrary-token SD forward recursion from the authors' public TASLP-2026 code and returns graph denominator, normalized graph GOP and `Occ(i)`. Deterministic regression tests freeze reference outputs including repeated-phone context.

### Criterion-ready joined feature bundle

`src/jp_speech_eval/phone_criterion_features.py`

Schema: `phone_criterion_feature_bundle_v1`

For each canonical phone position it joins, under strict model/revision/phone-sequence provenance:

- canonical LPP and LPP/frame;
- deletion LPR;
- full substitution-LPR vector;
- enumerated SD-GOP;
- published normalized graph GOP;
- `Occ(i)`;
- best-alternative diagnostics.

The bundle explicitly declares:

- `cross_model_raw_averaging_allowed = false`;
- `individual_feature_is_pronunciation_decision = false`;
- `requires_labeled_phone_or_human_criterion = true`;
- `score_mapped = false`;
- `product_calibrated = false`.

It also contains a shared-suffix locality diagnostic so target pairs such as `...をください` can be checked for unnecessary global feature drift.

## Forced CTC support remains diagnostic only

On the bundled Aivis `ラーメンをください` reference, Beatrice has `single_frame_support_ratio ≈ 0.857`, and its unconstrained greedy sequence covers only part of the canonical phone sequence.

Therefore forced-Viterbi CTC support is not accepted as physical phone segmentation/duration and is not the main clarity criterion.

## Three-backbone bundled-audio result

Pinned backbones:

- Beatrice: `prj-beatrice/japanese-hubert-base-phoneme-ctc-v4@f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99`
- DistilHuBERT dual CTC: `TylorShine/distilhubert-hiragana-ctc@01ffc3e5b0e49ba34180d50c48ea4111aa041cfd`
- WavLM dual CTC: `TylorShine/wavlm-base-plus-hiragana-ctc@47fa985035342365bcec4948bd821aaf58dd778a`

On bundled Aivis audio, within every model the correct `ラーメンをください` sequence is substantially better supported than deliberately wrong `コーヒーをください`:

- Beatrice raw sequence gap: `+28.939`;
- DistilHuBERT: `+12.726`;
- WavLM: `+8.668`.

Raw values are not compared across models as a common scale.

Published normalized graph feature means, correct vs wrong target:

- Beatrice: `-1.484` vs `-4.222`;
- DistilHuBERT: `-2.425` vs `-4.254`;
- WavLM: `-2.735` vs `-3.801`.

All `Occ(i)` values are finite. Near-zero occupancy is not interpreted as a phone error.

The correct/wrong target pair shares `ください`; the shared suffix remains locally similar while most discrimination appears in the differing prefix. This is useful implementation-locality evidence, not learner correctness validity.

## Official JVS human-native anchor: PASS

To eliminate the concern that the positive bundled result is only a TTS-domain artifact, Stage 0 now uses the three small `VOICEACTRESS100_001` human-native samples linked directly from the official JVS corpus project page:

- jvs001;
- jvs002;
- jvs003.

All have the known target text:

`また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。`

The control is a deterministic **same-phone-count rotated canonical sequence**, avoiding the trivial phone-count confound of a differently sized wrong target.

### Result

All three models preferred the known native sequence over the same-length rotated phone sequence for **all 9 speaker × model cases**.

Per-frame canonical-minus-rotated margins:

- Beatrice: `0.865371`, `0.898176`, `0.900156`;
- DistilHuBERT: `0.552010`, `0.645254`, `0.691336`;
- WavLM: `0.870016`, `0.903181`, `0.946380`.

Minimum target-order margin versus maximum jvs001 mild-gain delta was approximately:

- Beatrice: `17083×`;
- DistilHuBERT: `2883×`;
- WavLM: `15427×`.

These ratios are engineering sanity diagnostics, not effect sizes or model-quality scores.

This resolves an important Stage-0 question: the phone-CTC direction is not merely producing plausible target evidence on Aivis TTS; the same target-conditioned behavior is present on three independent real native JVS speakers.

It still does **not** establish learner phone-error validity.

Detailed report:

`reports/JVS_NATIVE_PHONE_CTC_ANCHOR_V1.md`

## JVS source bytes are now frozen

The official sample WAVs remain ephemeral and are deleted before artifact upload. The exact successful source bytes are now frozen in `download_official_jvs_samples.py`:

- jvs001: 778284 bytes, SHA-256 `dc9fd6e4caefc6e1781ad225f0b41ca13153da4afe2fb92f39f175fa3d9d85a7`;
- jvs002: 642764 bytes, SHA-256 `d91e5199508d89b45d68f18473c013f90bbfd68c1940ad039f5ea0b183f61ae2`;
- jvs003: 661004 bytes, SHA-256 `7b164601457b27c8a89c6aaef971967e5a6c7cf9bf2d46c303e21d1e8e41ed15`.

If Google Drive later serves different bytes, CI fails closed and requires explicit inspection rather than silently changing the benchmark.

## Frozen Japanese target frontend

`pyopenjtalk-plus 0.4.1.post8` exact text/kana/phone/mora outputs for the 21 planned Stage-0 targets are frozen in:

`data/audit/phone_target_snapshot_v1.csv`

Frontend/dictionary drift therefore cannot silently change canonical labels after validation begins.

## Current Stage-0 gates

1. **PASS** — ordinary Python 3.11 repository tests.
2. **PASS** — frozen pyopenjtalk-plus target regression.
3. **PASS** — pinned Beatrice correct/wrong target engineering preflight.
4. **PASS** — transparent alignment-free `{LPP,LPR}` extraction.
5. **PASS** — individual LPR sign prohibited as a direct error rule.
6. **PASS** — independent DistilHuBERT and WavLM phone-CTC comparisons.
7. **PASS** — published normalized SD forward/`Occ(i)` implemented and regression-tested.
8. **PASS** — normalized features finite on all three real backbones.
9. **PASS** — official JVS 3-speaker native-human target-consistency anchor on all three backbones.
10. **PASS** — official JVS source bytes frozen with fail-closed drift detection.
11. **PASS (infrastructure)** — strict criterion-ready `{LPP,LPR,graph GOP,Occ}` feature bundle and target-locality diagnostic.
12. **NEXT** — run the criterion-ready bundle over already-existing learner/native corpus files wherever those files are actually present; missing paths continue to skip.
13. **PENDING** — establish a genuinely labeled local-pronunciation criterion before fitting an MDD/classifier/regressor or mapping any phone evidence to a user score.
14. **BLOCKED by design** — new human recording requires a separate explicit readiness promotion after existing data are exhausted.

## Product policy

No Stage-0 phone-GOP code changes C-end `明瞭さ`, overall score, or any `/100` mapping. Phone evidence remains shadow-only. Product promotion requires Japanese L2 criterion validity, not merely model availability, native target discrimination, or plausible-looking acoustic features.
