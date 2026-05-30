# Special mora threshold update

Native P05/P95 thresholds are proposed per special-mora type only when coverage is sufficient.

- long_vowel: coverage=117, reliable=0, sufficient=False, low_ratio=None, high_ratio=None
- sokuon: coverage=12, reliable=0, sufficient=False, low_ratio=None, high_ratio=None
- moraic_nasal: coverage=44, reliable=0, sufficient=False, low_ratio=None, high_ratio=None
- yoon: coverage=66, reliable=0, sufficient=False, low_ratio=None, high_ratio=None

Runtime policy remains ok / too_short / too_long / uncertain. Insufficient types must not silently update thresholds.
