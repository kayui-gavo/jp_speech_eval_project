# Practice dimensions v2 engineering sanity validation

- JVS native rows: 300 (speaker-disjoint fresh parallel100 audit set)
- JANON sentence rows: 1976 (external fluency distribution only)
- Human ratings: none
- Purpose: validate implementation behavior and controlled negative-control separation, not teacher-grade or phoneme-error accuracy

## Distribution

| metric | n | mean | p10 | p50 | p90 | ceiling rate |
|---|---:|---:|---:|---:|---:|---:|
| pronunciation_clarity_normal | 300 | 95.0 | 95.0 | 95.0 | 95.0 | 0.0 |
| pronunciation_clarity_5db_noise | 60 | 82.0 | 82.0 | 82.0 | 82.0 | 0.0 |
| pronunciation_clarity_low_gain | 60 | 87.6 | 88.0 | 88.0 | 88.0 | 0.0 |
| rhythm_normal | 300 | 93.69 | 90.0 | 95.0 | 96.0 | 0.0 |
| rhythm_timing_jitter | 300 | 67.04 | 58.0 | 68.0 | 75.1 | 0.0 |
| rhythm_equal_fallback | 300 | 72.7733 | 71.0 | 73.0 | 74.0 | 0.0 |
| fluency_normal | 300 | 92.1133 | 82.0 | 95.0 | 96.0 | 0.0 |
| fluency_slow | 300 | 80.7733 | 68.0 | 83.0 | 89.0 | 0.0 |
| fluency_fast | 300 | 80.7133 | 77.0 | 80.5 | 86.0 | 0.0 |
| fluency_hesitation | 300 | 52.9267 | 31.0 | 52.5 | 74.1 | 0.0 |
| pitch_normal | 300 | 89.7567 | 81.0 | 92.0 | 97.0 | 0.0 |
| overall_four_dimension | 300 | 92.6833 | 88.9 | 94.0 | 95.0 | 0.0 |
| janon_native_fluency_external | 284 | 93.7324 | 90.0 | 95.0 | 96.0 | 0.0 |
| janon_learner_fluency_external | 1692 | 93.5721 | 91.0 | 96.0 | 96.0 | 0.0 |

## Paired separation

| comparison | mean normal-control delta | AUC |
|---|---:|---:|
| pronunciation_clarity_normal_vs_pronunciation_clarity_5db_noise | 13.0 | 1.0 |
| pronunciation_clarity_normal_vs_pronunciation_clarity_low_gain | 7.4 | 1.0 |
| rhythm_normal_vs_rhythm_timing_jitter | 26.65 | 1.0 |
| fluency_normal_vs_fluency_slow | 11.34 | 0.8995 |
| fluency_normal_vs_fluency_fast | 11.4 | 0.9146 |
| fluency_normal_vs_fluency_hesitation | 39.1867 | 0.9835 |

## Interpretation

- Pronunciation clarity no longer uses global mora-duration CV. It reacts to recording/evidence degradation but still cannot detect a cleanly recorded phone substitution.
- The 5 dB noise and low-gain conditions are synthetic channel controls. Their AUC values show deterministic response to those controls, not real pronunciation-error validity.
- Rhythm uses robust log-duration dispersion with reliable phone-label timing. Equal fallback is a low-confidence neutral estimate, not perfect rhythm.
- Fluency uses a continuous rate curve and excess-pause penalty, removing the previous broad 100-point plateau while allowing natural phrase pauses.
- The practice overall is computed from the same four dimensions shown in the UI. There is no hidden high-score floor.
- Special-mora user feedback is safe by default. Long-vowel/nasal candidate feedback requires explicit opt-in and reliable non-fallback boundaries; sokuon and yoon remain blocked.

## Remaining scientific limits

- Pronunciation clarity is an intelligibility/judgeability proxy, not phone-level GOP or phoneme correctness.
- JVS negative controls are synthetic paired perturbations; they validate sensitivity, not educational validity.
- JANON has no teacher score and is not used as a bad-pronunciation label.
- Special-mora learner feedback still needs reliable phone boundaries and real learner error labels.
- Pitch remains broad naturalness with a soft accent hint, not strict pitch-accent correctness.
