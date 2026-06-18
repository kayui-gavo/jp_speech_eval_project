# Prosody component diagnostics: JVS native

- generated_at: 2026-06-18T15:36:46.367185+00:00
- jvs_root: `/Users/ryukayuiii/Documents/jp_speech_eval_project/JVS`
- samples: 17
- speakers: 1
- utterances_per_speaker: 17
- alignment_mode: `equal`

## Score and Component Summary

| field | n | mean | min | p10 | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| prosody_score | 17 | 66.0588 | 52.0 | 53.6 | 67.0 | 81.0 | 82.0 |
| f0_coverage | 17 | 0.9009 | 0.7857 | 0.8026 | 0.9302 | 0.9671 | 1.0 |
| contour_corr | 17 | 0.1943 | -0.2535 | -0.1097 | 0.2116 | 0.5967 | 0.6999 |
| transition_agreement | 17 | 0.7377 | 0.5588 | 0.5982 | 0.725 | 0.8688 | 0.9241 |
| hl_match | 17 | 0.574 | 0.4167 | 0.4309 | 0.5909 | 0.7105 | 0.8 |
| accent_drop_match | 17 | 0.5117 | 0.0 | 0.0 | 0.5 | 1.0 | 1.0 |
| final_intonation_score | 17 | 92.3529 | 45.0 | 75.0 | 100.0 | 100.0 | 100.0 |
| component_contour | 17 | 0.5791 | 0.3972 | 0.4541 | 0.5841 | 0.7491 | 0.7968 |
| component_transition | 17 | 0.7377 | 0.5588 | 0.5982 | 0.725 | 0.8688 | 0.9241 |
| component_final | 17 | 0.9235 | 0.45 | 0.75 | 1.0 | 1.0 | 1.0 |
| component_hl | 17 | 0.574 | 0.4167 | 0.4309 | 0.5909 | 0.7105 | 0.8 |

## Approximate Component Loss

| component | mean_component | mean_weight | approx_score_loss |
|---|---:|---:|---:|
| contour | 0.5791 | 0.55 | 23.15 |
| transition | 0.7377 | 0.25 | 6.56 |
| hl | 0.574 | 0.08 | 3.41 |
| final | 0.9235 | 0.12 | 0.92 |

## Evidence and Gate Summary

- alignment fallback count: 0/17
- low F0 coverage count (<0.50): 0/17
- pitch_target_source counts: `{'openjtalk_accent_phrase_chain': 17}`
- hl_target_source counts: `{'openjtalk_accent_phrase_chain': 17}`

## Lowest Prosody Samples

| sample_id | prosody_score | f0_coverage | contour_corr | transition_agreement | final_intonation_score | pitch_target_source | gate_reason |
|---|---:|---:|---:|---:|---:|---|---|
| jvs001:VOICEACTRESS100_016 | 52 | 0.9714 | -0.011325836642889765 | 0.5989304812834225 | 45 | openjtalk_accent_phrase_chain | debug_only_by_profile;no_correction_needed;blocked_by_profile;missing_or_invalid_threshold_metadata |
| jvs001:VOICEACTRESS100_009 | 53 | 0.9487 | -0.25353334058752225 | 0.6138613861386139 | 100 | openjtalk_accent_phrase_chain | debug_only_by_profile;no_correction_needed;missing_or_invalid_threshold_metadata |
| jvs001:VOICEACTRESS100_013 | 54 | 0.8 | -0.18804808777072377 | 0.5970149253731342 | 100 | openjtalk_accent_phrase_chain | no_correction_needed;missing_or_invalid_threshold_metadata;debug_only_by_profile |
| jvs001:VOICEACTRESS100_001 | 57 | 0.8478 | -0.038551011109396634 | 0.5588235294117646 | 100 | openjtalk_accent_phrase_chain | no_correction_needed;debug_only_by_profile;missing_or_invalid_threshold_metadata |
| jvs001:VOICEACTRESS100_005 | 59 | 0.806 | -0.057549047598983244 | 0.704081632653061 | 100 | openjtalk_accent_phrase_chain | no_correction_needed;debug_only_by_profile;missing_or_invalid_threshold_metadata |

## Interpretation

- alignment fallback is unlikely to explain the low native prosody scores in this run.
- F0 coverage is generally sufficient, so missing F0 is not the primary explanation.
- pitch target/reference quality is a likely factor because targets are tool-generated or heuristic.
- contour similarity is a major score limiter.
- mora-to-mora transition agreement is a major score limiter.
- raw score scale/calibration is not ready for user-facing pitch scoring.
- This report is diagnostic only; it does not change runtime scoring.
- Native high-score calibration is not justified yet without flat/random/wrong-accent negative controls.

## Calibration Readiness

Not ready. This diagnostic identifies component behavior on native audio, but calibration should wait until counterfactual or real negative controls prove that native-correct contours rank above flat/random/wrong-accent contours.
