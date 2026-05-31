# Runtime special mora JVS validation

JVS is treated as native speech for false-alarm sanity checks. This reduces false-alarm risk but does not prove user-facing effectiveness.

- total utterances: 37
- total special mora instances: 275
- equal/non-phone fallback rate: 0.1309
- low evidence rate: 0.1309
- missing threshold rate: 0.0
- sokuon/yoon user-facing leakage: 0
- display_score impact check: safe; shadow validation does not modify display_score

## Threshold status
- long_vowel: status=active, sample_count=117, source=JVS
- moraic_nasal: status=active, sample_count=44, source=JVS
- sokuon: status=insufficient, sample_count=12, source=JVS
- vowel_lengthening_candidate: status=invalid, sample_count=None, source=None
- yoon: status=debug_only, sample_count=66, source=JVS

## False alarm proxy by type
- long_vowel: instances=130, active=130, user_feedback_allowed=11, false_alarm_proxy=11 (all=0.0846, active=0.0846), statuses=active
- moraic_nasal: instances=52, active=52, user_feedback_allowed=6, false_alarm_proxy=6 (all=0.1154, active=0.1154), statuses=active
- sokuon: instances=19, active=0, user_feedback_allowed=0, false_alarm_proxy=0 (all=0.0, active=None), statuses=insufficient
- yoon: instances=74, active=0, user_feedback_allowed=0, false_alarm_proxy=0 (all=0.0, active=None), statuses=debug_only

## Suppression reasons
- allowed: 17
- debug_only_threshold: 74
- insufficient_native_evidence: 19
- no_correction_needed: 165

## Answer
- 長音 native false alarm should be judged from the table above; <= 5% is the rollout target.
- 撥音 native false alarm should be judged from the table above; <= 5% is the rollout target.
- sokuon/yoon must remain zero leakage before any user-facing release.
- Concentrated transcript/mapping errors should be inspected in `jvs_shadow_decisions.csv`.
