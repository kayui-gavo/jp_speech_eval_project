# Pitch Naturalness v2 Cross-Dataset Audit

- activation decision: passed the documented guardrails and enabled for weak-reference practice only
- naturalness model: continuous ridge calibration, not binary probability
- accent target: automatic OpenJTalk accent-phrase chain used as an OJAD-style weak mismatch penalty at at most 8% weight
- the accent hint can never boost a low naturalness score
- official OJAD output is not scraped or treated as ground truth
- sentence-final 2 morae (3 for questions) are excluded from accent-hint matching

## Distribution

| group | condition | n | v1 mean | v2 mean | v2 p10 | v2 p50 | v2 p90 | accent hint mean | unavailable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh_jvs_parallel100 | flat | 300 | 9.0 | 22.1267 | 21.0 | 22.0 | 23.0 | 58.6364 | 0 |
| fresh_jvs_parallel100 | low_f0 | 0 | None | None | None | None | None | None | 300 |
| fresh_jvs_parallel100 | normal | 300 | 89.92 | 89.54 | 80.0 | 92.0 | 97.0 | 86.8496 | 0 |
| fresh_jvs_parallel100 | shuffled | 300 | 50.49 | 50.2267 | 43.0 | 49.0 | 58.1 | 85.9035 | 0 |
| fresh_jvs_parallel100 | wrong_drop | 300 | 84.9733 | 85.11 | 67.0 | 89.0 | 95.1 | 86.5582 | 0 |
| janon_learner_external | observed | 1686 | 74.8238 | 67.4543 | 37.0 | 73.0 | 86.0 | 81.1284 | 6 |
| janon_native_external | observed | 284 | 80.7746 | 80.2887 | 61.0 | 85.0 | 93.0 | 84.0135 | 0 |

## Product Checks

- Fresh JVS normal: mean `89.54`, p10 `80.0`, ceiling rate `0.0167`.
- Flat: mean `22.1267`; shuffled: `50.2267`.
- Wrong-drop: mean `85.11`. It is not required to be low for broad naturalness and is not used as a training negative.
- JANON native: mean `80.2887`, p10 `61.0`.
- JANON learner: mean `67.4543`. No teacher labels exist, so this is not an accuracy metric.
- Low-F0 controls must remain unavailable internally; the UI numeric fallback is a low-confidence practice estimate.

## Decision Rule

Activation requires: fresh JVS normal clearly above flat/shuffle, no 0/100 probability collapse, JANON native without catastrophic collapse, and automatic accent hints changing the total only mildly. Strict accent-error claims remain prohibited.
