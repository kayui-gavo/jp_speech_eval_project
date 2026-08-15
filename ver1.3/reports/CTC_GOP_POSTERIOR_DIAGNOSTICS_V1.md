# CTC-GOP posterior diagnostics — Stage-0 v1

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Product impact: **NONE**
Human-recording gate: **BLOCKED**

## Motivation

CTC removes the need for an externally forced phone boundary in several GOP
formulations, but standard CTC posteriors are characteristically sparse/peaky.
That property matters for pronunciation assessment because a posterior-derived
phone feature can become dominated by a small number of frames rather than
stable evidence over a phonetic event.

Li et al. (BEA 2026) explicitly study this problem for pronunciation
assessment. Their context-aware CTC variants combine output context dependency,
label priors and maximum-conditional-entropy regularization; on
SpeechOcean762, their best setup improves phoneme-level GOPT PCC from 0.612 to
0.641 and widens the correct-vs-mispronounced margin from 0.708 to 0.816.
This is evidence that CTC peakiness/context-independence is not merely a cosmetic
model property for GOP.

Cao et al. (Interspeech 2024) nevertheless show that CTC-based, alignment-free
GOP is useful and can handle substitution/deletion/insertion without relying on
phone-speech alignment; their best method reports a 29.02% relative improvement
over baseline GOP methods. Therefore the engineering conclusion is **not** to
abandon CTC-GOP, but to measure the posterior regime of every candidate
backbone and keep uncertainty/peakiness visible during criterion validation.

Parikh et al. (Interspeech 2025) further report that logit-based GOP can
outperform probability-based variants for MDD and that hybrid features,
uncertainty modeling and phone-specific weighting are promising. Consequently,
this project keeps raw logits/LPR-style evidence and posterior diagnostics as
separate research features rather than pretending a single softmax GOP is a
calibrated truth signal.

Primary references:

- Li, J.-T. et al. (2026), *Investigating Context-aware CTC for Pronunciation
  Assessment: Mitigating Peaky Behavior and Context Independency Assumption*,
  BEA 2026, DOI 10.18653/v1/2026.bea-1.3.
- Cao, X. et al. (2024), *A Framework for Phoneme-Level Pronunciation
  Assessment Using CTC*, Interspeech 2024.
- Parikh, A. K. et al. (2025), *Evaluating Logit-Based GOP Scores for
  Mispronunciation Detection*, Interspeech 2025, DOI
  10.21437/Interspeech.2025-1012.

## Implementation

`src/jp_speech_eval/ctc_posterior_diagnostics.py`

Schema:

`ctc_posterior_diagnostics_v1`

The module records, without score mapping:

- mean / p95 / maximum frame top-1 posterior;
- fraction of frames whose top-1 token is CTC blank;
- mean / p95 blank posterior;
- total phone-token probability mass (mean and p05);
- strongest phone posterior and best-vs-second phone margin;
- normalized full-vocabulary entropy;
- normalized phone-conditional entropy.

The phone inventory is supplied explicitly, so tokenizer/control/pause symbols
can be excluded from the phone-mass and phone-conditional diagnostics.

No universal threshold is defined. A highly peaked model is not automatically
"bad" and a diffuse model is not automatically "good"; these statistics are
model diagnostics to be related later to criterion MDD/regression performance,
native false alarms and robustness.

## Backbones

The diagnostics are now emitted in the bundled-audio preflight for:

1. pinned Beatrice Japanese HuBERT phone CTC;
2. pinned DistilHuBERT dual-CTC candidate;
3. pinned WavLM dual-CTC research comparator.

Raw GOP values from different backbones remain non-comparable and must not be
averaged. Posterior peakiness statistics may be compared descriptively because
they are normalized probability/entropy summaries, but model selection still
requires downstream criterion evidence.

## Decision rule for the project

Before a phone backbone can become primary evidence for C-end `明瞭さ`, require:

- Japanese learner criterion discrimination, preferably phone-level expert
  labels;
- acceptable expert-correct/native false alarms;
- speaker-held-out validation;
- local controlled-error directionality;
- channel/speed robustness;
- uncertainty/peakiness analysis showing the selected feature does not rely on
  unstable isolated CTC spikes;
- explicit score mapping/calibration after the above, not before.

Until then:

- `score_mapped = false`;
- `product_calibrated = false`;
- C-end four-score fallback policy is unchanged;
- human GOP recording remains blocked.
