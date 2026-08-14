# Product Score v3.3 Readiness

## Artifacts delivered

- `data/human_eval/pronunciation_listener_manifest_v1.csv`: internal master manifest with source and frozen-study metadata.
- `data/human_eval/pronunciation_listener_blind_v1.csv`: rater-safe manifest with opaque audio asset IDs only.
- `data/human_eval/listener_assignment_v1.csv`: balanced five-rating assignment plan with hidden duplicates.
- `data/human_eval/human_rating_schema.json`: primary accuracy construct and independent analyzability schema.
- `data/human_eval/learner_recording_needed.csv`: recruitment plan, not fabricated data.
- `reports/HUMAN_PRONUNCIATION_CALIBRATION_PROTOCOL.md`: pre-rating construct, analysis, and mapping freeze.

## Current real-audio coverage

| Measure | Count | Status |
|---|---:|---|
| Same-target native anchors | 28 / 4 speakers / 7 targets | Available; must be rated, not assumed perfect. |
| Same-target real learner recordings with frozen v3.2 evidence | 14 / 2 speakers / 7 targets | Pilot only. |
| JANON rows from those two learner speakers across all stimuli | 578 | Not a substitute for the seven-target learner quota. |
| Required added learner recordings | 56 / 8 new speakers / 7 targets | Recruitment required; combined cohort becomes 70 clips / 10 speakers. |
| Complete clean/RIR/noise/codec channel pairs | 14 | One pair short of the pre-registered 15-pair minimum. |
| Unique seed-study clips | 98 | 28 anchors + 14 learners + 56 channel controls. |
| Planned base ratings | 490 | 5 independent slots per unique clip. |
| Planned presentations incl. duplicates | 500 | 10 hidden duplicates, one per listener slot. |

The blind manifest omits `speaker_group_hidden`, dataset, condition, WavLM fields, alignment, recording quality, and real source paths. The assignment file also omits condition and group; it exposes only rater-safe target and opaque asset information.

## Channel-bias readiness

The v3.2 evidence showed WavLM deltas of +0.0639 (15 dB noise), +0.0935 (RIR), and +0.0536 (codec), larger than the pooled native/learner median separation. The paired study is therefore mandatory. Its primary comparison is human pronunciation delta versus frozen WavLM delta within the same original utterance, not a claim that the synthetic condition caused pronunciation to change.

Current channel coverage is insufficient for promotion because the complete-pair count is 14, not 15. No new audio was synthesized or modified in this branch merely to satisfy the count.

## Interfaces and safety

`PronunciationCalibration` is a provenance-only contract:

- frozen default: `microsoft/wavlm-large`, `layer12`, `median`;
- `mapping=None`;
- `production_enabled=False`;
- no score-mapping method exists.

ProductScore v2 is unchanged. ProductScore v3 remains default-off and candidate-only. The branch does not load WavLM in ordinary pytest and does not change the existing WavLM distances.

## Gates

### CODE MERGE GATE: PASS

The v3.2 parity audit remains 45 real audio cases with zero user-facing regressions, v3 remains default-off, and this v3.3 work adds only disabled calibration metadata, study construction, documentation, and tests.

### PRODUCT V3 PROMOTION: BLOCK

Human ratings have not yet been collected; the current learner cohort is 14 rather than the planned 60–80 real learner clean clips; the channel cohort is 14 rather than at least 15 complete pairs; and no held-out human-validation result exists. No /100 mapping or product enablement is justified.

## Next controlled actions

1. Recruit the eight planned learners and collect one clean recording of each target.
2. Add one provenance-checked real channel quartet to reach 15 pairs.
3. Deploy the blind manifest and schema to at least five independent Japanese listeners per clip.
4. Execute the frozen analysis protocol before considering any mapping experiment.
