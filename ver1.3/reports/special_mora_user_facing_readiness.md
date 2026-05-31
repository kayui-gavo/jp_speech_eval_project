# Special mora user-facing readiness

This report recommends rollout state only. limited_candidate does not mean full rollout.
## long_vowel
- recommendation: limited_candidate
- false_alarm_v1: 0.0859
- false_alarm_v2: 0.0342
- sensitivity summary: {'detection_rate_25': 0.8376, 'detection_rate_40': 0.5812, 'detection_rate_60': 0.2479, 'detection_rate_80': 0.1111}
- requirements before actual rollout: profile=v2_limited_candidate, explicit flag enabled, strong_reference only, too_short only, near-boundary suppressed, manual inspection recommended
- allowed modes if future enabled: fixed-reference strong target only; weak-reference remains mild candidate only
- recommended wording: short, non-accusatory too-short suggestion

## moraic_nasal
- recommendation: limited_candidate
- false_alarm_v1: 0.12
- false_alarm_v2: 0.0455
- sensitivity summary: {'detection_rate_25': 0.9318, 'detection_rate_40': 0.5909, 'detection_rate_60': 0.2727, 'detection_rate_80': 0.1364}
- requirements before actual rollout: profile=v2_limited_candidate, explicit flag enabled, strong_reference only, too_short only, near-boundary suppressed, manual inspection recommended
- allowed modes if future enabled: fixed-reference strong target only; weak-reference remains mild candidate only
- recommended wording: short, non-accusatory too-short suggestion

## sokuon
- recommendation: blocked
- false_alarm_v1: 0.0
- false_alarm_v2: 0.0
- sensitivity summary: None
- requirements before actual rollout: profile=v2_limited_candidate, explicit flag enabled, strong_reference only, too_short only, near-boundary suppressed, manual inspection recommended
- allowed modes if future enabled: fixed-reference strong target only; weak-reference remains mild candidate only
- recommended wording: short, non-accusatory too-short suggestion

## yoon
- recommendation: blocked
- false_alarm_v1: 0.0
- false_alarm_v2: 0.0
- sensitivity summary: None
- requirements before actual rollout: profile=v2_limited_candidate, explicit flag enabled, strong_reference only, too_short only, near-boundary suppressed, manual inspection recommended
- allowed modes if future enabled: fixed-reference strong target only; weak-reference remains mild candidate only
- recommended wording: short, non-accusatory too-short suggestion

## Required limitations
- JVS native calibration controls false alarm but does not prove user-level effectiveness
- counterfactual feature perturbation is not human validation
- JANON has no teacher/native listener rating
- pronunciation_score ceiling effect remains unresolved
- sokuon threshold insufficient
- yoon duration threshold debug_only
- limited_candidate does not mean full rollout

## Rollout gate status
- long_vowel: blocked_pending_manual_inspection
- moraic_nasal: blocked_pending_manual_inspection
- sokuon: blocked_insufficient_native_evidence
- yoon: blocked_debug_only_duration_not_valid

## Required manual checks before actual rollout
- profile must be v2_limited_candidate
- flag must be explicitly enabled
- strong_reference only
- feedback direction must be too_short only
- near-boundary cases must be suppressed
- manual inspection should annotate at least 10 items with >=80% should_allow_user_facing and <=10% false_alarm
