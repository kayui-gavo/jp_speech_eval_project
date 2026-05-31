# Special mora user-facing readiness

This report recommends rollout state only. It does not automatically enable user-facing calibrated special mora feedback.

## long_vowel
- status: keep_shadow
- reason: native_false_alarm_proxy_all=0.0846, shortened_40_detection_rate=0.6379
- required_flag: enable_user_facing_calibrated_special_mora=True
- allowed_modes: fixed-reference strong target only; weak-reference mild candidate only
- feedback_wording: short, non-accusatory length/nasal-hold suggestion

## moraic_nasal
- status: keep_shadow
- reason: native_false_alarm_proxy_all=0.1154, shortened_40_detection_rate=0.6364
- required_flag: enable_user_facing_calibrated_special_mora=True
- allowed_modes: fixed-reference strong target only; weak-reference mild candidate only
- feedback_wording: short, non-accusatory length/nasal-hold suggestion

## sokuon
- status: blocked
- reason: insufficient native reliable count
- next_step: collect/search more JVS sokuon or improve closure evidence

## yoon
- status: blocked/debug_only
- reason: duration threshold inappropriate
- next_step: design mora_count_consistency evidence

## Required limitations
- counterfactual feature perturbation is not human validation
- JANON has no teacher/native listener rating
- pronunciation_score ceiling effect still unresolved
- sokuon threshold insufficient
- yoon duration threshold debug_only
- JVS native calibration reduces false alarm risk but does not prove user-facing effectiveness
