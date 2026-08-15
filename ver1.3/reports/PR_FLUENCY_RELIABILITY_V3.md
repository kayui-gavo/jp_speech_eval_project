# Fluency / Reliability Evolution v3 — PR summary

This branch keeps valid-Japanese ProductScore weights and score mappings unchanged while fixing free-speech language eligibility, adding spontaneous-fluency shadow evidence, and making legacy reliability caps auditable.

## Main changes

- Free-speaking ASR is unforced and language-aware before Japanese scoring.
- Confident non-Japanese input is no-score/retry, never a low Japanese score.
- Spontaneous fluency shadow separates speed, breakdown and repair evidence.
- Faster-whisper word timestamps feed only low-confidence pause-location candidates; ASR punctuation is not a syntactic clause label.
- Runtime reliability caps remain unchanged.
- Historical cap replay detects scorer/config drift and cap censoring rather than calling every current-scorer delta a cap effect.
- Historical 40-row acceptance: 37 applicable, 6 fully compatible with strict replay, 31 drifted, 4 compatible-but-censored, 2 trustworthy uncensored counterfactuals; the 2 trustworthy cases show zero cap effect.
- New fixed-reference evaluator results persist exact pre-cap and post-cap score telemetry in the same run.
- Fresh acceptance protocol and manifest are frozen before any cap decision.

## Non-goals

- no ProductScore weight change;
- no valid-Japanese threshold retuning;
- no spontaneous-fluency `/100` mapping;
- no reliability-cap removal;
- no promotion of WavLM / CTC / GOP / rhythm-DTW;
- no use of historical scorer drift as evidence for product improvement.

## Decision

No reliability-cap policy change is justified from the historical acceptance. The next decision gate is a fresh held acceptance using evaluator-native pre/post-cap telemetry plus the already defined human criteria.
