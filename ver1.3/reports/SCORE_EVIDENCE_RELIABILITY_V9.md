# Score Evidence Reliability v9

## Status

Branch: `score-evidence-reliability-v9`

Base: `scoring-integrity-v8`

This round fixes a consumer-facing reliability semantic problem without changing ProductScore.

## Problem

The previous UI could show one reliability label plus a percentage-like scalar. In direct free speech, that scalar could be high because the recording was clean and endpointing was stable even while several headline dimensions were still neutral priors rather than measured evidence.

That presentation made two different questions look like one:

1. Can the recording be analysed reliably?
2. How much of the four-dimensional score is actually supported by measured or broad evidence?

A clean microphone signal does not prove that clarity, rhythm, fluency, and intonation were all measured strongly.

## New contract

`score_evidence_quality_v1` exposes two independent summaries.

### `recording_analyzability`

Answers whether the waveform is usable for analysis. It currently combines recording-quality and endpointing evidence.

It is explicitly:

`can_the_recording_be_analyzed_not_learner_ability`

### `score_evidence`

Summarises how much of the four public dimensions is supported by actual evidence rather than a neutral placeholder.

Evidence-state strength for this UI summary is:

- `measured_proxy`: 1.0
- `broad_proxy`: 0.60
- `neutral_prior`: 0.0
- `unavailable`: 0.0

The four dimensions retain the existing ProductScore weights when evidence coverage is summarised.

This quantity is explicitly:

`evidence_coverage_not_probability_score_is_correct`

It must not be interpreted as a statistical confidence interval, probability of correctness, or learner ability score.

## Important allowed state

The product now deliberately allows:

- recording analyzability: `high`
- score evidence: `low`

This is expected in a clean direct-free-speech recording when only one dimension has broad evidence and the remaining dimensions are neutral priors.

## Established HF Space UI

The formal `debug_ui/index.html` remains the public UI baseline.

The old percentage-like reliability presentation is removed from the public result panel. The panel now presents separate consumer states for:

- recording status
- score evidence

The per-dimension evidence badges remain visible, so the aggregate evidence state does not replace construct-specific provenance.

No consumer redesign or alternate formal UI was introduced.

## ProductScore boundary

This round does **not** change:

- `consumer_four_score_v2`
- component weights
- display transform
- free-speech neutral priors
- fixed-reference score formulas
- shadow promotion status
- history comparability rules

Therefore no score-contract bump is required.

The v8 partial-evidence aggregate remains shadow-only. It is not promoted merely because excluding neutral priors is semantically attractive.

## Remaining gate

The largest unresolved scoring issue is still the free-speech numerical contract: neutral priors remain part of the official headline ProductScore until a replacement mapping is supported by held real audio and construct-matched human criteria.

The next score-changing step should therefore use the existing frozen validation infrastructure rather than another heuristic remap:

- short valid Japanese
- long valid Japanese
- learner and native Japanese
- English and Mandarin negative controls
- silence/noise/unusable recordings
- same-source channel variants
- listener ratings for clarity/comprehensibility, fluency, rhythm naturalness, and utterance-level intonation naturalness

Only after that evidence should a candidate replace neutral priors or change the ProductScore contract.
