# Special mora threshold update

Native P05/P95 thresholds are proposed per special-mora type only when coverage is sufficient.

- long_vowel: coverage=117, reliable=117, sufficient=True, low_ratio=0.2307, high_ratio=1.0948
- sokuon: coverage=12, reliable=12, sufficient=False, low_ratio=None, high_ratio=None
- moraic_nasal: coverage=44, reliable=44, sufficient=True, low_ratio=0.2742, high_ratio=1.1429
- yoon: coverage=66, reliable=66, sufficient=True, low_ratio=0.8804, high_ratio=2.2489

Runtime policy remains ok / too_short / too_long / uncertain. Insufficient types must not silently update thresholds.
