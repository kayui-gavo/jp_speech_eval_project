# Runtime special mora profile validation

- default_safe flag=False: candidates=0, long=0, nasal=0, sokuon/yoon leakage=0, too_long leakage=0, near-boundary leakage=0
- v2_shadow flag=False: candidates=0, long=0, nasal=0, sokuon/yoon leakage=0, too_long leakage=0, near-boundary leakage=0
- v2_limited_candidate flag=False: candidates=0, long=0, nasal=0, sokuon/yoon leakage=0, too_long leakage=0, near-boundary leakage=0
- v2_limited_candidate flag=True: candidates=2, long=1, nasal=1, sokuon/yoon leakage=0, too_long leakage=0, near-boundary leakage=0
- default profile and flag-off profiles must keep user-facing feedback disabled.
- display_score impact: safe; profile validation does not modify display_score.
