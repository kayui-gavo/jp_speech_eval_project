# Runtime special mora JVS validation v2

V2 uses stricter user-facing thresholds and too_short-only feedback. It does not automatically enable rollout.
## v2 false alarm by type
- long_vowel: false_alarm=4/117 (0.0342), rollout=limited_candidate
- moraic_nasal: false_alarm=2/44 (0.0455), rollout=limited_candidate
- sokuon: false_alarm=0/12 (0.0), rollout=blocked
- yoon: false_alarm=0/66 (0.0), rollout=blocked
- display_score impact: safe; validation does not modify display_score.
- sokuon/yoon leakage remains zero if false_alarm count is zero for those rows.
