# Pitch v2 Objective Validation (No Human Ratings)

This audit adds objective reliability and generalization checks without changing runtime scoring. JVS is used for native baselines and paired F0 controls; JANON is external-only and is never used as a training label.

## Main Results

| check | result | interpretation |
|---|---:|---|
| JVS native mean (95% bootstrap CI) | 89.7567 (88.96–90.5367) | Unseen speakers/sentences remain high |
| Native vs flat paired delta | +64.9333 (AUC 1.0) | Flat F0 is clearly separated |
| Native vs shuffled paired delta | +36.69 (AUC 0.9978) | Unstable F0 is clearly separated |
| Median same-sentence SD across speakers | 4.2134 | Remaining speaker variability on the 0–100 scale |
| Speaker macro mean range | 80.7–95.1 | No single fresh JVS speaker collapses |
| JANON native mean | 80.9225 | External microphone/timing domain is lower than JVS |
| JVS–JANON native mean gap | 8.8341 | Domain/timing gap, not an accuracy claim |

## Speaker-Disjoint Cross-Validation

| fold | held-out speakers | normal | shuffled | flat | ordering |
|---|---:|---:|---:|---:|---|
| fold_1 | 6 | 87.4797 | 51.2332 | 22.1149 | PASS |
| fold_2 | 6 | 86.0721 | 47.7093 | 22.616 | PASS |
| fold_3 | 6 | 89.9278 | 52.3156 | 22.3239 | PASS |
| fold_4 | 5 | 88.0698 | 50.4853 | 22.2686 | PASS |
| fold_5 | 5 | 85.5682 | 48.34 | 22.6174 | PASS |

All folds keep native > shuffled > flat. This checks speaker leakage and coefficient stability against the predefined proxy anchors; it does not create human validity.

## Same-Sentence Cross-Speaker Stability

- 10 parallel sentences observed for at least 24 of 30 unseen JVS speakers were compared (actual coverage 28–30).
- Median within-sentence SD: `4.2134`; maximum: `8.004`.
- Highest-variance sentence: `VOICEACTRESS100_100` (SD `8.004`, p10 `71.8`, p90 `92.1`).
- This is cross-speaker consistency, not test-retest reliability; JVS does not provide repeated takes for this exact check.

## External JANON Audit

| group | n | mean | p10 | p50 | p90 |
|---|---:|---:|---:|---:|---:|
| JANON native | 284 | 80.9225 | 62.0 | 85.0 | 93.0 |
| JANON learner (descriptive only) | 1686 | 68.7456 | 41.0 | 74.0 | 86.0 |

Learner identity is not an error label. The learner distribution is useful only as an external trend and failure-analysis pool because JANON has no teacher pitch score and currently uses approximate equal-mora timing.

## What This Establishes

- The broad pitch-naturalness score generalizes across held-out JVS speakers and a different JVS reading set.
- Flat and shuffled controls remain strongly lower without forcing wrong accent-drop contours to be errors.
- Score mass is not collapsed to only 0 or 100.
- JANON exposes a real domain gap that should be addressed through alignment/device robustness, not by fitting learner identity.

## What It Does Not Establish

- Teacher-grade pitch-accent correctness.
- Agreement with human perception or educational usefulness.
- Test-retest reliability for the same person and sentence.
- Device/noise robustness on real phone recordings.
- Scientific calibration of the other three dimensions.

## Next Step Without Human Ratings

Run waveform-level robustness tests on held-out JVS audio (codec, room noise, gain and microphone filtering), then audit all four dimensions for score drift. Keep the clean score as the reference and require small drift for pronunciation/rhythm/fluency and stable pitch ordering for pitch controls.
