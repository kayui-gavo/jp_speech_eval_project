# Special mora rollout gate decision

## long_vowel
- rollout_status: blocked_pending_manual_inspection
- jvs_false_alarm_rate: 0.0342
- manual_annotated_count: 0
- manual_should_allow_rate: 0
- manual_false_alarm_rate: 1
- reason: ['blocked_pending_manual_inspection', 'manual_allow_rate_below_80_percent', 'manual_false_alarm_rate_above_10_percent']

## moraic_nasal
- rollout_status: blocked_pending_manual_inspection
- jvs_false_alarm_rate: 0.0455
- manual_annotated_count: 0
- manual_should_allow_rate: 0
- manual_false_alarm_rate: 1
- reason: ['blocked_pending_manual_inspection', 'manual_allow_rate_below_80_percent', 'manual_false_alarm_rate_above_10_percent']

## sokuon
- rollout_status: blocked_insufficient_native_evidence

## yoon
- rollout_status: blocked_debug_only_duration_not_valid

## Limitations
- manual inspection is not a full listener study
- JVS controls native false alarm risk but does not prove learner benefit
- JANON trend is not ground truth
- counterfactual feature perturbation is not human validation
- limited_candidate is not full rollout