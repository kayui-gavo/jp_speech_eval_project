# Japanese phone-CTC backbone comparison v1

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Product score impact: **NONE — SHADOW / RESEARCH ONLY**
Human recording requirement: **NONE**

## Purpose

Compare independently trained Japanese phone-CTC backbones on the exact same bundled audio before spending human recording time. The comparison asks only engineering questions:

- can the model load reproducibly at an immutable revision?
- can it explain the correct target better than a deliberately wrong Japanese target?
- is that difference much larger than mild amplitude-gain perturbation?
- can it expose the same alignment-free LPP/LPR feature family?

This is **not** a phone-error benchmark because the bundled Aivis reference has no phone-error labels. Raw LPR signs are not treated as phone-correctness decisions.

## Models

### Beatrice Japanese HuBERT phone CTC

- `prj-beatrice/japanese-hubert-base-phoneme-ctc-v4`
- revision `f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99`
- existing project research backend

### DistilHuBERT Japanese dual CTC

- `TylorShine/distilhubert-hiragana-ctc`
- revision `01ffc3e5b0e49ba34180d50c48ea4111aa041cfd`
- compact independent candidate
- candidate repo lacks a preprocessor config, so the adapter explicitly reproduces the pinned base DistilHuBERT waveform preprocessing contract and records that provenance

### WavLM Japanese dual CTC

- `TylorShine/wavlm-base-plus-hiragana-ctc`
- revision `47fa985035342365bcec4948bd821aaf58dd778a`
- research comparison candidate
- licensing is less attractive for a future C-end default, so this model is not assumed to be a production dependency

## Bundled-audio protocol

Audio:

`assets/reference_cache/ramen_kudasai_aivis.ref.wav`

Correct target:

`ラーメンをください。`

Deliberately wrong target:

`コーヒーをください。`

Controls:

- original waveform
- 0.8x amplitude gain
- 1.2x amplitude gain

All candidate results remain `score_mapped = false` and `product_calibrated = false`.

## Results

| Backbone | correct canonical log posterior | wrong canonical log posterior | correct - wrong | noncanonical-win diagnostic, correct | mild-gain behavior |
|---|---:|---:|---:|---:|---|
| Beatrice | -18.6590 | -47.5980 | +28.9390 | 7 / 14 | tiny relative to target mismatch |
| DistilHuBERT dual CTC | -32.2534 | -44.9798 | +12.7264 | 7 / 14 | extremely small |
| WavLM dual CTC | -35.6813 | -44.3496 | +8.6682 | 6 / 14 | larger than Distil but still much smaller than target mismatch |

**Do not compare the absolute log-posterior values or gaps as if they shared a common scale.** The models have different posterior calibration/peakiness and possibly different frame behavior. The useful comparison is within each backbone: correct target versus wrong target and target mismatch versus channel/gain perturbation.

DistilHuBERT gain controls changed the correct canonical log posterior from about `-32.2534` to `-32.2578` (0.8x) and `-32.2518` (1.2x).

WavLM gain controls changed it to about `-35.9621` (0.8x) and `-35.5371` (1.2x), still well inside its correct-vs-wrong target gap on this single sample.

## Interpretation after literature alignment

All three models can provide useful **target-conditioned acoustic evidence** on this bundled utterance: the correct transcription is substantially better supported than an unrelated Japanese target within each model.

However, the count of positions where one substitution/deletion alternative has greater posterior than the canonical sequence is **not an error count**. The 2026 segmentation-free GOP work uses LPP/LPR values as a joint feature vector for downstream pronunciation assessment, and its normalized method adds the graph-derived `Occ(i)` term. Therefore:

- Beatrice `7/14` does not mean seven errors;
- DistilHuBERT `7/14` does not mean seven errors;
- WavLM `6/14` does not make WavLM automatically a better pronunciation judge;
- direct thresholding of an individual LPR is rejected;
- cross-model raw-score averaging is rejected.

The three-backbone result is valuable because it demonstrates that **whole-target sequence evidence is reproducible across independently trained Japanese phone models**, while local pronunciation interpretation still requires the paper-aligned feature semantics and labeled criterion data.

## Paper-aligned normalized forward implementation

The authors' public `frank613/CTC-based-GOP/taslpro26` implementation is now directly used as the algorithmic reference. The project contains an independent NumPy reimplementation in:

`src/jp_speech_eval/segmentation_free_gop_norm.py`

It reproduces the normalized SD alternative-graph forward recursion and returns:

- graph denominator log posterior;
- normalized `GOP-SF-SD`-style log ratio;
- `Occ(i)` from normalized wildcard-state forward mass.

`Occ(i)` is explicitly marked **not a physical phone duration**. Five regression tests freeze reference outputs on deterministic synthetic posterior grids, including repeated-phone context. The clean Python 3.11 suite now passes **238 tests with 6 existing warnings**.

The current heavy preflight is being advanced so Beatrice, DistilHuBERT and WavLM will all emit this graph-derived normalization alongside the existing enumerated LPP/LPR features. This still does not create a pronunciation decision or `/100` mapping.

## Engineering decision

1. Keep Beatrice, DistilHuBERT and WavLM phone evidence in shadow/research mode.
2. Prefer DistilHuBERT as the compact second backbone for engineering experiments; it is independent, small and robust to the mild gain controls used here.
3. Keep WavLM as a research comparison rather than a presumed product dependency.
4. The published SD alternative-graph normalized forward recursion and `Occ(i)` feature are now implemented and regression-tested; next verify them on real model outputs across all three backbones.
5. Validate the resulting feature vectors on already-existing human data wherever available before asking the user to record anything.
6. Do not alter C-end `明瞭さ` or overall `/100` from this experiment.
