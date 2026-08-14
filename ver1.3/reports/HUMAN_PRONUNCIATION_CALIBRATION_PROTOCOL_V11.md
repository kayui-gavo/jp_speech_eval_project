# Human Pronunciation Calibration Protocol v1.1

> Historical pre-collection record. Superseded by [Protocol v1.2](HUMAN_PRONUNCIATION_CALIBRATION_PROTOCOL_V12.md), which separates JANON isolated-word calibration from JVS long-sentence channel-bias controls before rating collection.

```text
protocol_version: v1.1
rating_collection_started: false
study_stage: pilot_seed / final_calibration (future only)
primary_construct: pronunciation_accuracy
```

## Amendment status

This is a pre-collection amendment to the v1 protocol, not a post-hoc change: no formal listener ratings have started. It supersedes v1 for future collection and records five corrections:

1. seed/pilot and final calibration datasets are distinct;
2. channel data are four-condition **channel sets** (channel quartets), not generic pairs;
3. incomplete-rater reliability uses ordinal agreement plus variance components, not an unexplained standard ICC;
4. final duplicate quality control is at least five repeats per rater; and
5. target scope is restricted to `janon_7target_isolated_word_validation_v1`.

This branch does not alter ProductScore v2, WavLM features/distances, layer/fusion selection, any /100 mapping, ProductScore v3 enablement, or production behavior.

## Study stages

### Pilot seed

The existing 98 unique real clips are explicitly `study_stage=pilot_seed`:

| Component | Clips |
|---|---:|
| Native anchors | 28 |
| Current real learner isolated words | 14 |
| Channel-set clips | 56 (14 channel sets × 4 conditions) |
| Total | 98 |

The 500-presentation pilot assignment has five base ratings per clip and only one hidden repeat per listener slot. It is permitted only for UI/usability checks: Japanese instruction clarity, analyzability control clarity, audio playback and saving, blindness, and randomization behavior. Pilot ratings must not enter the final calibration dataset or a future mapping fit.

One repeat per rater is explicitly **insufficient for stable per-rater repeatability estimation**. The pilot assignment is therefore not a final calibration assignment.

### Final calibration study (not yet generated)

The final study requires 56 new **real** learner clean recordings: eight additional learners × the seven fixed targets. Combined with the current 14, that produces 70 learner clips from ten speakers. No placeholder learner audio, final manifest rows, or final assignment is generated before the files and source metadata exist.

If current seed clips are retained unchanged, the learner expansion projects approximately 154 unique clips (98 + 56). The separate channel-set shortfall may increase the final count further.

`data/human_eval/final_template/pronunciation_listener_manifest_final_template.csv` is header-only. The final assignment function refuses an incomplete cohort and requires all of:

- 70 real clean learner clips;
- 10 learner speakers;
- a `final_calibration` manifest stage;
- source audio paths that exist; and
- at least 15 complete channel sets.

## Primary construct and rating contract

The primary construct is **pronunciation accuracy**: whether the sounds in the displayed target word or sentence are accurately realized. It excludes fluency, speed, pitch/intonation, emotion, native-likeness, accentedness, recording quality, and speaker attractiveness.

Rater-facing instruction:

> 画面に示された語を基準として、発音そのものがどの程度正確に実現されているかを評価してください。話す速さ、声の高さ、感情表現、録音音質は、可能な限り評価に含めないでください。

Ratings are ordinal 1–7 (1 = 非常に不正確; 7 = 非常に正確). `analyzable_yes_no` is independent. When `analyzable=no`, the accuracy rating may be null; it must never be auto-converted to 1.

`human_rating_schema.json` is declared as `schema_kind=study_contract`. Its companion `human_rating_json_schema_v1.json` is a formal JSON Schema with `if/then`: analyzable `yes` requires an integer 1–7; analyzable `no` permits null. Comprehensibility is not collected in v1 and may not be averaged with pronunciation accuracy if later added.

## Raters and blinding

The primary cohort is native Japanese listeners. Near-native listeners, if operationally necessary, must have native language, Japanese proficiency, and phonetics/speech-training experience recorded under anonymous IDs and must receive a separate sensitivity analysis. They are not assumed exchangeable with the primary cohort.

The rater-facing blind manifest and assignment expose only opaque clip/asset IDs, target text/kana, and rating controls. They do not expose speaker identity or group, L1, dataset, channel condition, WavLM evidence, product score, alignment, recording quality, or source path. A server-side resolver maps opaque asset IDs to audio; the internal master manifest must not be sent to raters.

## Channel-set protocol

A complete `channel_set` / channel quartet contains exactly:

```text
clean, RIR, noise15dB, codec
```

Each set yields three within-source paired contrasts:

```text
RIR - clean
noise15dB - clean
codec - clean
```

There are currently 14 complete channel sets. Promotion-grade validation requires at least 15. The missing fifteenth set is a real-data collection gap; it must not be manufactured by pretending a partial set is complete.

For each contrast, compare human pronunciation-accuracy delta with the frozen WavLM evidence-index delta. This tests channel bias: a stable human rating with a large WavLM shift is evidence against treating the index as pronunciation-only. It does not assert that the channel transform changed a speaker's pronunciation.

## Assignment and repeatability plan

The design is an incomplete rater matrix: each clip has five ratings drawn from ten listener slots. Final-study assignments are generated only after the final real manifest exists.

Final assignment requirements:

- at least five independent ratings per clip;
- at least five hidden duplicate presentations per rater, preferably about 5–10% of that rater's workload;
- duplicate presentation IDs differ while retaining the same underlying clip;
- duplicates are not adjacent, not disclosed, and spread across targets when possible; and
- duplicate selection includes native anchors, learner clips, and channel-set clips when those groups are assigned to the rater.

With an approximately 154-clip final manifest, five ratings per clip imply approximately 770 base ratings. Ten rater slots with five repeats each would imply approximately 820 presentations. These are projections only; the exact count is calculated from the populated final manifest and must never be hard-coded before recordings exist.

## Frozen analysis plan

Keep raw ordinal ratings as the primary data. Do not silently treat 1–7 as a fully interval-scale outcome.

1. Report analyzability rate separately, overall and by channel condition.
2. Primary inter-rater agreement: Krippendorff's alpha with ordinal distance, which allows incomplete rating matrices.
3. Reliability of the averaged rating: use a generalizability-theory or mixed-effects variance-component approach appropriate to incomplete observational ratings. Estimate clip and rater random effects; include target as a fixed or random effect as justified and document the choice. Report the resulting ICC-like reliability with its model specification.
4. Evaluate hidden-repeat intra-rater consistency after the final five-repeat minimum is met.
5. Produce rater-normalized accuracy ratings alongside raw ordinal results, preserving both.
6. Compute overall and within-target Spearman association between the frozen SSL evidence index and averaged human pronunciation accuracy.
7. Run leave-one-target-out and leave-one-speaker-out validation. Held-out units may not select layers, fusion, normalization, or mappings.
8. Estimate speaker and rater effects, and report target variance.
9. Analyze each channel contrast within channel set: human delta versus frozen SSL-index delta.

Native/learner membership remains a sampling descriptor only, not a human-rating substitute or primary validation outcome.

## Scope and mapping freeze

The only contemplated validation scope is:

```text
janon_7target_isolated_word_validation_v1
```

Even a successful study cannot support arbitrary Japanese sentence pronunciation correctness, spontaneous-speech pronunciation, or a general educational /100 pronunciation score.

No mapping is permitted in this protocol: no linear, min-max, sigmoid, percentile, or /100 conversion. `PronunciationCalibration.mapping` stays `None`; `production_enabled` stays `False`. Any future mapping must be selected on train/development data and evaluated on strict held-out targets or speakers, preferably with nested target/speaker splits.

## Gates

The pilot is ready for usability-only collection. The final human study is not ready until the learner and channel requirements are met. Product score mapping remains blocked until reliable ratings, channel-bias evidence, and held-out validation have been completed.
