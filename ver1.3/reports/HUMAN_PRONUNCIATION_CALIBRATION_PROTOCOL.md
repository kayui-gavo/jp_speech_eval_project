# Human Pronunciation Calibration Protocol v1 (pre-registration)

## Status and scope

This protocol is frozen before collecting listener ratings. It governs validation of the existing `ssl_pronunciation_evidence_index`; it does **not** define a ProductScore mapping, change WavLM layers/distances, or enable ProductScore v3.

The frozen research baseline is same-target multi-reference `microsoft/wavlm-large`, **layer 12 + median reference aggregation**. Layer 24 remains a recorded secondary diagnostic. Any future layer/fusion comparison must be selected on training/development data only and evaluated on held-out targets/speakers.

## Primary construct

The only primary human construct is **pronunciation accuracy**: how accurately the sounds in the displayed target word or sentence are realized.

Rater-facing Japanese instruction:

> 画面に示された語を基準として、発音そのものがどの程度正確に実現されているかを評価してください。話す速さ、声の高さ、感情表現、録音音質は、可能な限り評価に含めないでください。

The ordinal response scale is 1–7: 1 = 非常に不正確, 4 = 中程度, 7 = 非常に正確. The scale must not be described as naturalness, native-likeness, accentedness, or comprehensibility.

Explicit exclusions from the primary label are fluency, speech rate, intonation, emotion, recording quality, and speaker attractiveness. Comprehensibility is not collected in v1; if added later, it must be stored and analyzed separately, never averaged with pronunciation accuracy.

## Independent analyzability judgment

Each presentation has a separate `analyzable_yes_no` response. If the clip cannot be analyzed as speech, the rater can select `no` and leave the pronunciation rating null. Unanalyzable audio must never be converted into a low pronunciation score.

## Listeners and blinding

Each clip receives at least five independent ratings. Recruit native Japanese listeners where possible. If a near-native listener is used, record anonymized `rater_id`, native language, Japanese proficiency, and phonetics/speech-training experience, then examine rater effects separately.

Raters see only the target text/kana, audio asset, and rating controls. They must not see speaker identity, native/learner membership, L1, dataset, channel condition, WavLM evidence, current product scores, or source audio paths. The study delivery service resolves the blinded `audio_asset_id` server-side; it must not expose the master manifest.

## Dataset plan

The committed seed manifest contains:

| Subset | Clips | Notes |
|---|---:|---|
| Human-native anchors | 28 | 4 native speakers × 7 targets; raters still score them, never auto-assign 7. |
| Existing same-target learner clips | 14 | 2 real learner speakers × 7 targets; frozen v3.2 WavLM evidence exists. |
| Channel-paired controls | 56 | 14 clean/RIR/15 dB-noise/codec pairs from real JVS utterances. |
| Total unique clips | 98 | No synthetic speech is used. |

The channel subset is one pair below the pre-registered 15-pair minimum: only 14 complete four-condition pairs already exist. Do not manufacture a fifteenth pair during rating. Add one provenance-checked real clean/RIR/noise/codec quartet before treating the channel test as promotion-grade.

The learner quota is deliberately separate from native anchors and channel copies. The usable fixed-seven-target learner cohort is only 14 recordings from two speakers. `learner_recording_needed.csv` plans eight additional real learner speakers × seven targets = 56 clips, yielding 70 real learner clean recordings across ten speakers. Recruitment is required before a mapping-validation study; the present 14-clip learner subset is suitable only as a seed/pilot.

## Assignment and duplicate quality control

`listener_assignment_v1.csv` assigns every unique clip to five of ten anonymous listener slots, yielding 490 base rating slots. One hidden duplicate per listener slot yields 500 planned presentations. Duplicate presentations have a different `presentation_id` but retain the same `sample_id`, so intra-rater ordinal consistency can be measured.

Order is deterministically randomized with seed 33017. Channel-pair members cannot be adjacent for the same listener, and recent-target balancing avoids target blocks. Pair condition and duplicate status are unavailable to the rater UI.

## Frozen analysis plan

Before any mapping is considered, report all of the following:

1. Analyzability rate overall and by channel condition, separate from accuracy.
2. Inter-rater reliability: ICC(A,k) for averaged pronunciation ratings and an ordinal agreement statistic (weighted agreement or Krippendorff's alpha with the chosen ordinal distance documented).
3. Intra-rater duplicate consistency.
4. Rater-normalized pronunciation ratings, with both raw and normalized results preserved.
5. Overall Spearman correlation between frozen WavLM evidence and averaged human pronunciation accuracy.
6. Within-target Spearman correlations, target variance, and target-wise plots/tables.
7. Leave-one-target-out and leave-one-speaker-out validation; no held-out partition may choose a layer, fusion, normalizer, or mapping.
8. Speaker random-effect analysis (and rater random effect where model assumptions permit).
9. Paired channel analysis: within `pair_id`, compare human accuracy delta against frozen WavLM evidence-index delta for clean vs RIR/noise/codec. Report human stability and WavLM channel bias rather than conflating either with pronunciation change.

Native-vs-learner identity is only a sampling label. It cannot be used as the primary validation outcome or as a substitute for listener ratings.

## Mapping protocol freeze

There is no /100 mapping in v1: no linear, min-max, sigmoid, percentile, or categorical conversion is allowed. `PronunciationCalibration.mapping` remains `None` and `production_enabled=False`.

If the human dataset later supports mapping research, choose a train/development partition and a strict held-out target or held-out speaker partition before fitting. Prefer nested target/speaker splits when cohort size permits. A mapping selected on all data and reported on the same data is invalid.

## Promotion evidence, not a single correlation threshold

Promotion requires a coherent package of evidence:

- listener ratings show usable agreement and analyzability is not collapsed by channel condition;
- WavLM has a stable relation to pronunciation accuracy overall **and within target**;
- that relation is not explained only by target or native/learner identity;
- channel bias is acceptably small or explicitly modeled and validated; and
- target/speaker hold-outs support the relation.

Failure of any one part blocks a user-facing pronunciation mapping but does not invalidate the existing candidate-only telemetry.
