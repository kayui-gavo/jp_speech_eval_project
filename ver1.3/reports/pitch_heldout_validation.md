# Held-Out Pitch Naturalness Validation

- scoring formula: frozen at commit `95b9510`; no tuning performed after test selection
- JVS locked internal test: 600 real recordings, 60 unseen speakers, nonparallel unseen texts
- JANON external test: 284 native + 316 learner sentence recordings
- total real recordings: 1,200
- JVS counterfactual score rows: 3,000; actual independent JVS recordings remain 600

## Score Summary

| group | condition | n scored | speakers | mean | p10 | p50 | p90 | unavailable | <80 | >=80 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| janon_learner_external | learner_observed | 315 | 24 | 75.9746 | 50.0 | 81.0 | 93.6 | 1 | 0.4698 | 0.5302 |
| janon_native_external | native_normal | 284 | 4 | 80.7746 | 54.0 | 87.0 | 95.0 | 0 | 0.3521 | 0.6479 |
| jvs_native_heldout | flat | 600 | 60 | 9.0 | 9.0 | 9.0 | 9.0 | 0 | 1.0 | 0.0 |
| jvs_native_heldout | low_f0 | 0 | 60 | None | None | None | None | 600 | 0.0 | 0.0 |
| jvs_native_heldout | native_normal | 600 | 60 | 88.6533 | 78.0 | 92.0 | 95.0 | 0 | 0.13 | 0.87 |
| jvs_native_heldout | shuffled | 600 | 60 | 50.2583 | 42.0 | 48.0 | 63.0 | 0 | 0.9817 | 0.0183 |
| jvs_native_heldout | wrong_drop | 600 | 60 | 82.87 | 68.0 | 86.0 | 93.0 | 0 | 0.2917 | 0.7083 |

## Speaker-Cluster Confidence Intervals

| group | speaker-macro mean | cluster-bootstrap 95% CI |
|---|---:|---|
| jvs_native_heldout | 88.6533 | [87.8683, 89.4217] |
| janon_native_external | 80.7746 | [73.338, 88.2113] |
| janon_learner_external | 75.9487 | [70.7637, 81.4629] |

## Paired JVS Counterfactual Separation

| comparison | speaker-macro margin | cluster-bootstrap 95% CI | normal wins |
|---|---:|---|---:|
| normal - flat | 79.6533 | [78.8683, 80.4217] | 1.0 |
| normal - shuffled | 38.395 | [37.2467, 39.4467] | 0.99 |
| normal - wrong_drop | 5.7833 | [5.195, 6.3583] | 0.8217 |

## Findings

- Held-out JVS native speech is generally high, but not uniformly high: mean 88.65, p10 78, and 13% of rows are below 80.
- Flat separation is strong: all 600 paired native rows beat the flat control.
- Shuffle separation is strong but imperfect: native wins 99% of pairs, while 1.83% of shuffled controls still score at least 80.
- Wrong-drop remains weak: mean margin is only about 5.78 and 70.83% of wrong-drop controls still score at least 80. Strict pitch-accent correctness is unsupported.
- Low-F0 gating behaves correctly: 600/600 controls are unavailable rather than assigned a formal score.
- JANON external native mean is 80.77 and 35.21% fall below 80. This shows material domain/alignment sensitivity; do not interpret those rows as low native ability.
- JANON learner and native distributions overlap substantially. Without teacher labels, the score is not a calibrated learner-ability measure.

## Scientific Boundary

- JVS test speakers `jvs041`–`jvs100` and `nonpara30` texts were not used by the original 24-item weak-score development audit.
- Counterfactuals alter mora-level F0 only; they are paired semi-audio controls, not independently recorded learner errors.
- JANON native/learner rows use endpointed equal-mora timing because JANON lacks phone labels. Treat JANON as external robustness/trend evidence, not exact pitch-error ground truth.
- JANON learner prompts have no teacher pitch labels; lower scores cannot automatically be interpreted as incorrect Japanese.
- Wrong-drop separation remains the key criterion for whether strict accent claims are unsupported.
- The test report is frozen evidence. Any later formula change requires a new untouched test set or external human-labeled corpus.
