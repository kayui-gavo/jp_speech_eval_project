# Japanese phone-GOP Stage-0 status v1

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Human recording gate: **BLOCKED**
Product score impact: **NONE — PHONE EVIDENCE REMAINS SHADOW/RESEARCH ONLY**

## Stage-0 principle

Human recording must not be used to discover basic model/API/metric bugs. Stage 0 first exhausts synthetic tests, bundled/reference audio, pinned models, frozen target frontends, published-algorithm regression tests, public human-native anchors and already-existing expert-labeled learner corpora.

The core scientific rule is enforced in code and reports:

> segmentation-free CTC outputs are **features**, not pronunciation-error decisions.

An individual negative LPR, a low `Occ(i)`, a forced CTC support frame, or a noncanonical alternative with higher posterior must never be translated directly into “this phone is wrong” or a `/100` score.

## Current code baseline

The latest fully completed ordinary CI before the newest UME-JRF adapter commits passed:

- Python 3.11 clean environment;
- **246 tests passed**;
- 6 existing deprecation/audio-backend warnings;
- no product-score changes.

Additional no-network tests have since been added for JVS source semantics, existing-JANON discovery, UME-JRF research-only licensing/construct separation and fail-closed layout probing. The heavy run triggered by this report validates the current tree rather than relying on an older count.

## Literature-aligned feature stack

Primary methodological reference:

- Xinwei Cao, Zijian Fan, Torbjørn Svendsen, Giampiero Salvi, **“Segmentation-Free Goodness of Pronunciation”**, arXiv:2507.16838 / IEEE TASLP 2026, DOI `10.1109/TASLPRO.2026.3654852`.
- authors' public implementation: `frank613/CTC-based-GOP/taslpro26`.

The project keeps the relevant concepts separate:

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

Stage 0 uses the three small `VOICEACTRESS100_001` human-native samples linked directly from the official JVS corpus project page: jvs001, jvs002 and jvs003.

Known target text:

`また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。`

The control is a deterministic **same-phone-count rotated canonical sequence**, avoiding the trivial phone-count confound of a differently sized wrong target.

All three models preferred the known native sequence over the same-length rotated phone sequence for **all 9 speaker × model cases**.

Per-frame canonical-minus-rotated margins:

- Beatrice: `0.865371`, `0.898176`, `0.900156`;
- DistilHuBERT: `0.552010`, `0.645254`, `0.691336`;
- WavLM: `0.870016`, `0.903181`, `0.946380`.

Minimum target-order margin versus maximum jvs001 mild-gain delta was approximately:

- Beatrice: `17083×`;
- DistilHuBERT: `2883×`;
- WavLM: `15427×`.

These ratios are engineering sanity diagnostics, not effect sizes or model-quality scores. This confirms that the target-conditioned phone-CTC direction also behaves sensibly on real native human speech, not only Aivis TTS. It still does **not** establish learner phone-error validity.

Detailed report:

`reports/JVS_NATIVE_PHONE_CTC_ANCHOR_V1.md`

## Correction: JVS raw HTTP bytes are not an acoustic identity

A later heavy run exposed a useful reproducibility mistake in our own infrastructure. The official jvs001 Google Drive response changed from the first observed ~778 kB WAV representation to ~414 kB while preserving the expected ~8.621 s utterance. The smaller size is consistent with a different WAV sample-rate/container representation; treating the raw transport SHA-256 as immutable corpus identity was therefore too brittle.

The previous hard raw-byte gate is removed.

`download_official_jvs_samples.py` now validates:

- the reviewed official Google Drive file ID;
- readable uncompressed mono PCM WAV semantics;
- expected utterance duration within a tight tolerance;
- positive sample rate/frame count and plausible PCM width.

It records sample rate, sample width, raw bytes and raw SHA-256 as **provenance**, but explicitly marks raw transport hash as not equal to acoustic identity. The downstream native-anchor model test remains the actual acoustic sanity gate. This prevents harmless container changes from blocking CI while still failing on duration/content-level source drift.

## Frozen Japanese target frontend

`pyopenjtalk-plus 0.4.1.post8` exact text/kana/phone/mora outputs for the 21 planned Stage-0 targets are frozen in:

`data/audit/phone_target_snapshot_v1.csv`

Frontend/dictionary drift therefore cannot silently change canonical labels after validation begins.

## Existing JANON benchmark path is now criterion-bundle aware

`benchmark_phone_gop_existing_data.py` now:

- stays local-data-only for corpora;
- discovers matching JANON isolated-word files when a JANON checkout already exists near the research workspace;
- never calls non-native JANON speech “bad” or “error” without labels;
- groups it as `janon_non_native_isolated_unlabeled_quality`;
- emits the strict `{LPP,LPR,graph GOP,Occ}` criterion bundle for short targets;
- skips missing corpora instead of turning a missing path into a human-recording request;
- preserves native vs learner provenance without inferring quality.

This is useful descriptive evidence, but JANON's existing project use here still does not provide the expert local correctness criterion needed for a phone MDD mapping.

## Major criterion upgrade: UME-JRF

The official NII UME-JRF documentation resolves the biggest remaining Stage-0 design question more cleanly than asking for a new convenience recording set.

UME-JRF was explicitly designed for Japanese pronunciation-learning/education research and contains learner speech, matched native speech and expert pronunciation ratings. Official documentation states:

- 141 international students across 26 L1 backgrounds, intermediate through advanced Japanese;
- 41 Tokyo/Kanto Japanese native speakers for matched native readings;
- 16 kHz, 16-bit, mono WAV;
- four Japanese-language-education experts with substantial pronunciation-teaching experience;
- A: broad pronunciation 1–5 absolute ratings on selected phoneme-balanced sentences;
- B: item-specific difficult-sound/minimal-pair correctness, binary correct/incorrect, on 28–29 selected sentences;
- C: item-specific prosody ratings, 1–5, on 12 selected prosody sentences;
- D: item-specific target-phone correctness, 1–5, for ten rated difficult-sound words including `酸っぱい`, where the official example criterion is whether the geminate/促音 is produced.

This is much closer to the criterion required for local Japanese pronunciation validity than native false-alarm testing alone.

However, the official license is **research only / non-commercial**. Therefore UME-JRF is a criterion/algorithm research source, not a C-end production dataset. It must not be silently copied into runtime assets or used to train a commercial production model without separate permission/licensing review.

New research infrastructure:

- `reports/UME_JRF_CRITERION_PLAN_V1.md`
- `src/jp_speech_eval/ume_jrf_research.py`
- `scripts/probe_ume_jrf_layout.py`
- `tests/test_ume_jrf_research.py`

The adapter defines set-specific raw-label schemas and explicit license guards, but **does not invent an on-disk label parser**. The public introduction points to corpus-internal `Vol1/doc/FJlabel/description.txt`; until the actual corpus documentation is present and inspected, automatic label parsing remains blocked.

Priority after acquisition:

1. D-rated ten words for isolated local phone/special-mora criterion;
2. B-rated difficult-sound sentences for local correctness in sentence context;
3. A-rated sentences for broad pronunciation evidence;
4. C-rated prosody kept strictly separate from segmental clarity.

## Current Stage-0 gates

1. **PASS** — ordinary Python 3.11 repository tests on the last completed baseline; current tree revalidation is running.
2. **PASS** — frozen pyopenjtalk-plus target regression.
3. **PASS** — pinned Beatrice correct/wrong target engineering preflight.
4. **PASS** — transparent alignment-free `{LPP,LPR}` extraction.
5. **PASS** — individual LPR sign prohibited as a direct error rule.
6. **PASS** — independent DistilHuBERT and WavLM phone-CTC comparisons.
7. **PASS** — published normalized SD forward/`Occ(i)` implemented and regression-tested.
8. **PASS** — normalized features finite on all three real backbones.
9. **PASS** — official JVS 3-speaker native-human target-consistency anchor on all three backbones.
10. **CORRECTED** — JVS reproducibility now validates reviewed source ID + audio semantics; mutable raw HTTP container bytes are provenance only, not an acoustic-identity hard gate.
11. **PASS (infrastructure)** — strict criterion-ready `{LPP,LPR,graph GOP,Occ}` feature bundle and target-locality diagnostic.
12. **PASS (infrastructure)** — existing JANON local discovery + criterion-bundle benchmark path without invented learner quality labels.
13. **PASS (design)** — UME-JRF expert-label construct/license plan and fail-closed local layout probe implemented.
14. **NEXT** — validate the current tree/heavy JVS semantics; then, if UME-JRF is locally obtained, inspect `FJlabel` documentation and implement the real label/audio parser without guessing its format.
15. **PENDING** — fit/evaluate any local phone-correctness classifier only after genuine B/D expert labels are available; use speaker/item-held-out validation and report expert-correct learner false alarms.
16. **BLOCKED by design** — new human recording remains unnecessary until public/existing labeled data paths are exhausted.

## Product policy

No Stage-0 phone-GOP or UME-JRF research code changes C-end `明瞭さ`, overall score, or any `/100` mapping. UME-JRF itself is research-only/non-commercial. Phone evidence remains shadow-only until Japanese L2 criterion validity is demonstrated and any production training/calibration data have product-compatible rights.
