# Scoring System v2 Upgrade

## Product Contract

For a complete Japanese utterance that passes the content and minimum-evidence gates, the API returns four numeric practice dimensions:

| dimension | meaning | low-evidence behavior |
|---|---|---|
| pronunciation | recording clarity, mora alignment and pronunciation stability proxy | keep a coarse score; lower confidence |
| rhythm | mora timing and available special-mora evidence | use overall timing when a specific special mora cannot be judged |
| fluency | speech rate, pauses and continuity | keep a numeric score |
| pitch | broad F0 naturalness plus a weak Tokyo-style accent hint | keep a conservative score; prohibit strict accent-error feedback |

`dimension_scores` and `dimension_confidence` are separate fields. Alignment fallback and low F0 no longer erase an otherwise valid Japanese result. Non-Japanese content, silence/bad recording, and very short evidence remain no-score rather than four zeros.

## Pitch v2

The previous binary classifier was rejected because its probabilities concentrated near 0 and 100. Runtime v2 uses a regularized continuous ridge model:

- training anchors: native normal 95, flat 20, shuffled 45;
- model inputs: F0 coverage, log-F0 range, local movement, smoothness, flatness penalty and instability penalty;
- output range: 10–98 before UI evidence fallback;
- wrong-drop is not a training negative;
- strict pitch-accent correctness is not claimed.

### Accent hint

The arbitrary-sentence path currently has an automatic OpenJTalk accent-phrase prediction, not an official manually verified OJAD target. It is treated as an OJAD-style soft linguistic hint:

- salient phrase-initial rises and accent-nucleus drops only;
- ±1 mora timing tolerance;
- final 2 morae excluded, or final 3 morae for questions;
- automatic target weight at most 8%;
- manually verified OJAD/human targets may use at most 20%;
- mismatch penalty only: the hint can never boost a flat or unstable contour.

This separation prevents sentence-final intonation, focus and affect from being treated as automatic errors.

## Data Separation

| role | corpus | speakers / recordings | use |
|---|---|---:|---|
| train | JVS nonpara30, jvs001–030 | 28 / 280 available | fit continuous coefficients |
| development | JVS nonpara30, jvs031–040 | 10 / 100 | choose regularization and inspect distribution |
| prior locked test | JVS nonpara30, jvs041–100 | 60 / 600 | held-out comparison |
| fresh confirmation | JVS parallel100, jvs071–100 | 30 / 300 different recordings | final activation check |
| external audit | JANON sentences | 28 / 1,976 | domain audit only; never training labels |

JVS flat/shuffled samples are F0-only paired controls, not independent recordings. JANON learner identity is not treated as an error label because teacher pitch ratings are unavailable.

## Final Audit

| group | pitch v2 mean | p10 | p50 | p90 |
|---|---:|---:|---:|---:|
| fresh JVS normal | 89.54 | 80 | 92 | 97 |
| fresh JVS flat | 22.13 | 21 | 22 | 23 |
| fresh JVS shuffled | 50.23 | 43 | 49 | 58.1 |
| fresh JVS wrong-drop | 85.11 | 67 | 89 | 95.1 |
| JANON native | 80.29 | 61 | 85 | 93 |
| JANON learner | 67.45 | 37 | 73 | 86 |

Only 1.67% of fresh JVS normal rows hit the 98-point ceiling, compared with the rejected binary candidate's heavy near-100 concentration. Low-F0 controls remain internally unavailable; the UI supplies a low-confidence coarse practice estimate so a valid Japanese sentence still has four numeric dimensions.

## End-to-End Local Recordings

| confirmed text | pronunciation | rhythm | fluency | pitch | overall |
|---|---:|---:|---:|---:|---:|
| 私は東京大学の学生です | 80 | 100 | 96 | 84 | 88 |
| レポードはまだやってないです | 80 | 100 | 100 | 95 | 92 |
| おはようございます | 66 | 83 | 100 | 76 | 75 |

The first two use fallback alignment and therefore expose low confidence for pronunciation/rhythm/pitch while retaining numeric practice scores.

## Remaining Limits

- The 0–100 scale is anchored with proxy targets, not human teacher ratings.
- Wrong lexical accent cannot be reliably detected in arbitrary emotional/focused speech.
- Official OJAD outputs are not dynamically imported or used as ground truth.
- JANON has only four native speakers and approximate equal-mora timing.
- Human-rated learner recordings are still required for educational score calibration.
