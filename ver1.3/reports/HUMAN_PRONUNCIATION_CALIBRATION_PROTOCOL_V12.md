# Human Pronunciation Calibration Protocol v1.2

```text
protocol_version: v1.2
rating_collection_started: false
study_stage: pilot_seed / final_calibration (future only)
```

## Pre-collection amendment

This is a pre-collection amendment to v1.1. No formal listener ratings have started. Its amendment reason is: **separate isolated-word construct-validation dataset from long-sentence channel-bias controls before collection**.

It does not change ProductScore, WavLM features or distances, layer/fusion selection, production enablement, or introduce a /100 mapping.

## Two independent study components

### Primary pronunciation calibration

```text
study_component: primary_pronunciation_calibration
validation_scope: janon_7target_isolated_word_validation_v1
```

This is the only construct-validation cohort and the only cohort that may eventually support pronunciation-calibration research. It consists only of clean JANON isolated-word recordings for these seven targets:

```text
うっとうしい, がっしり, さっさと, ばっちり, オイル, バグ, 酸味
```

The final target is 28 native anchors (4 × 7) plus 70 real learner recordings (10 × 7), for **98 clean isolated-word clips**. The current pilot seed has 28 native and 14 learner clips: **42 primary clips**. Fifty-six learner clips from eight new real learner speakers are still required. No learner audio is fabricated by this protocol.

Primary analyses, after formal collection only, are: analyzability; ordinal Krippendorff alpha; mixed-effects/generalizability-theory reliability; hidden-repeat consistency; overall and within-target Spearman association to frozen WavLM evidence; leave-one-target-out; leave-one-speaker-out; speaker/rater effects; and human versus frozen WavLM evidence. A future mapping, if justified after held-out validation, may use only this component.

### Channel-bias control

```text
study_component: channel_bias_control
validation_scope: jvs_channel_bias_long_sentence_v1
```

This is a separate JVS long-sentence robustness-control study. A channel set (channel quartet) is one source utterance presented in:

```text
clean, RIR, noise15dB, codec
```

It provides the three within-source contrasts `RIR - clean`, `noise15dB - clean`, and `codec - clean`. The current pilot seed contains 14 complete JVS quartets: **56 channel clips**. The formal target is at least 15 complete quartets: **at least 60 channel clips**. One additional real quartet is required.

Channel analyses are separate: analyzability by condition; reliability separately from the primary study; within-source human deltas for each contrast; corresponding WavLM deltas; and paired/mixed-effects channel analysis. Long-sentence absolute ratings must not be compared directly with isolated-word absolute ratings. The channel component must never enter a future pronunciation /100 mapping training set; it is solely robustness/correction validation.

## Pilot versus final study

The current 98-clip `pilot_seed` is deliberately mixed only for UI/usability purposes:

| Pilot component | Clips |
|---|---:|
| Primary JANON isolated-word seed | 42 |
| JVS long-sentence channel-control seed | 56 |
| Total | 98 |

Pilot collection may assess instruction clarity, player reliability, blind metadata, save behavior, analyzability control, and randomization. Pilot ratings do not enter formal calibration. Pilot analyses must retain the `study_component` boundary and must not report a mixed 98-clip WavLM-human correlation as pronunciation validity.

When real learner recordings and the fifteenth channel quartet exist, the projected final cohort is 98 primary + 60 channel = **158 unique clips**. At five independent ratings per clip, this is **790 base ratings**. With ten listener slots and at least five hidden repeats per slot, this projects **50 repeat presentations** and approximately **840 presentations**. Exact final numbers are computed only from the populated real manifest.

## Construct, rating, and rater cohort

The primary construct is ordinal 1–7 **pronunciation accuracy**. It excludes fluency, rate, intonation, emotion, native-likeness, accentedness, recording quality, and speaker attractiveness. `analyzable_yes_no` remains independent: `no` permits a null pronunciation rating and never automatically becomes 1.

The primary rater cohort is native Japanese listeners. If near-native listeners are used, their language and training background must be recorded under anonymous IDs and analyzed separately as a sensitivity cohort.

The rater-facing manifest reveals only opaque asset/clip identifiers and target text/kana. It does not reveal group, dataset, source path, channel condition, WavLM values, calibration eligibility, or study component.

## Reliability and assignment

The incomplete rater matrix has five ratings per clip drawn from ten listener slots. Primary ordinal agreement is Krippendorff alpha with ordinal distance. Reliability of an averaged rating is estimated with a mixed-effects/generalizability-theory variance-component approach appropriate to incomplete observational designs, including clip and rater effects and documented target-effect treatment.

Formal assignments require at least five undisclosed, non-adjacent hidden duplicate presentations per listener slot, preferably 5–10% of workload. Duplicates have distinct presentation IDs, retain the underlying clip, and span components/targets where feasible. The current one-repeat pilot plan remains explicitly insufficient for stable per-rater repeatability.

## Mapping and scope freeze

There is no user-score mapping in v1.2: no linear, min-max, sigmoid, percentile, or /100 conversion. `PronunciationCalibration.mapping` remains `None` and `production_enabled` remains `False`.

Even successful primary validation supports only controlled same-target, multi-human-reference, JANON seven-target isolated-word pronunciation evidence. It does not validate arbitrary Japanese sentence correctness, spontaneous speech, or a general educational pronunciation score.

## Gates

| Gate | Status | Reason |
|---|---|---|
| PILOT UI/DESIGN READY | PASS | Seed is labeled, blind, and assignment/schema infrastructure is test-covered. |
| FINAL PRIMARY DATA READY | BLOCK | 56 real learner isolated-word clips are missing. |
| CHANNEL-BIAS DATA READY | BLOCK | Only 14 of the required 15 real channel quartets exist. |
| PRODUCT SCORE MAPPING READY | BLOCK | No formal ratings, reliability, channel analysis, or held-out validation exist. |
