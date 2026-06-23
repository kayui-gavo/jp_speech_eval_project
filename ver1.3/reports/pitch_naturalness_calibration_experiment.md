# Pitch Naturalness Lightweight Calibration Experiment

## Scope

- Offline experiment only; the runtime scoring formula is unchanged.
- Model: standardized logistic regression.
- Training target: distinguish real JVS native contours from paired flat/shuffled F0 controls.
- This is a native-likeness proxy, not teacher-rated pitch-accent correctness.
- No JVS/JANON speaker appears in more than one internal split.

## Data Split

| split | speakers | real native recordings | training use |
|---|---:|---:|---|
| train (`jvs001–030`) | 28 | 280 | fit scaler and coefficients |
| dev (`jvs031–040`) | 10 | 100 | choose decision threshold |
| locked test (`jvs041–100`) | 60 | 600 | final internal evaluation only |
| JANON external | 28 | 600 | distribution audit only; no labels |

Each available JVS real recording has paired flat and shuffled controls. Wrong-drop is never used for fitting and remains a challenge set.
The local nonpara/lab requirements excluded jvs006 and jvs028 from training, so the actual training set has 28 speakers rather than the planned 30.

## Candidate Score Distribution

| split | condition | n | mean | p10 | p50 | p90 |
|---|---|---:|---:|---:|---:|---:|
| dev | flat | 100 | 0.99 | 1.0 | 1.0 | 1.0 |
| dev | native_normal | 100 | 92.7 | 72.0 | 100.0 | 100.0 |
| dev | shuffled | 100 | 9.3 | 1.0 | 2.0 | 18.9 |
| dev | wrong_drop | 100 | 82.03 | 15.9 | 98.0 | 100.0 |
| external_janon_learner | learner_observed | 315 | 64.5016 | 1.0 | 90.0 | 99.0 |
| external_janon_native | native_normal | 284 | 81.3486 | 13.6 | 98.0 | 100.0 |
| locked_test | flat | 600 | 0.9883 | 1.0 | 1.0 | 1.0 |
| locked_test | low_f0 | 0 | None | None | None | None |
| locked_test | native_normal | 600 | 94.855 | 88.9 | 99.0 | 100.0 |
| locked_test | shuffled | 600 | 9.1817 | 1.0 | 2.0 | 23.0 |
| locked_test | wrong_drop | 600 | 86.9633 | 51.0 | 98.0 | 100.0 |
| train | flat | 280 | 0.9786 | 1.0 | 1.0 | 1.0 |
| train | native_normal | 280 | 96.1857 | 92.0 | 99.0 | 100.0 |
| train | shuffled | 280 | 6.9893 | 1.0 | 2.0 | 20.1 |
| train | wrong_drop | 280 | 87.4179 | 60.7 | 98.5 | 100.0 |

## Locked-Test Results

- ROC AUC for native vs flat/shuffled: `0.993487`.
- Balanced accuracy at the dev-selected threshold `0.17`: `0.965417`.
- Native beats paired flat: `0.9983`.
- Native beats paired shuffled: `0.99`.
- Native beats paired wrong-drop: `0.5633`.
- Locked JVS native candidate mean: `94.855`.
- Flat candidate mean: `0.9883`; shuffled mean: `9.1817`.
- Wrong-drop candidate mean: `86.9633`.

## Existing Formula vs Trained Candidate

| evaluation group | existing mean | trained candidate mean | observation |
|---|---:|---:|---|
| held-out JVS native | 88.6533 | 94.855 | native is higher |
| paired flat | 9.0 | 0.9883 | stronger rejection |
| paired shuffled | 50.2583 | 9.1817 | stronger rejection |
| wrong-drop challenge | 82.87 | 86.9633 | separation is not improved |
| JANON native | 80.7746 | 81.3486 | mean similar, but candidate p10 is 13.6 |
| JANON learner | 75.9746 | 64.5016 | no teacher labels; not accuracy evidence |

## Interpretation

- Training is useful if held-out native remains high while flat/shuffled remain low.
- The trained probabilities are strongly saturated near 0 or 100; they are classifier confidence, not calibrated 0–100 educational scores.
- JANON native/learner overlap is expected because JANON has no teacher pitch labels and uses approximate equal-mora timing. The very low JANON-native p10 shows domain sensitivity remains.
- A high wrong-drop score means this model still measures broad contour naturalness, not lexical pitch-accent correctness.
- Do not activate this candidate in the product until human ratings are collected and score calibration is validated.

## Next Data Needed

1. Teacher/listener ratings for naturalness and intelligibility on real learner recordings.
2. Independently recorded flat, unstable, and wrong-accent speech rather than F0-only counterfactuals.
3. More female/mixed-condition external speech with reliable mora timing.
