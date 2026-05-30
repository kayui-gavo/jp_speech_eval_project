# JVS native sanity check

This report is a native false-alarm audit. It does not prove scoring validity, but it checks whether native speakers are being punished by current engineering thresholds.

## Summary
- n: 17
- display mean: 98.7647
- pronunciation mean: 100.0
- fluency mean: 94.2941
- rhythm timing mean: 97.4706
- delivery fluency mean: 91.7647
- retry rate: 0.0
- unscorable rate: 0.0
- alignment fallback rate: 0.0
- special mora false alarm proxy: 0.0

## Acceptance checks
- PASS: mora_clarity native mean >= 90
- PASS: rhythm timing native mean >= 85
- PASS: delivery fluency native mean >= 85
- PASS: unscorable rate <= 5%
- PASS: special mora false alarm proxy <= 5%

## Interpretation
- `score_pronunciation` is still a mora-timing/acoustic proxy, not phoneme correctness.
- Ceiling warning: many native pronunciation scores are exactly 100, so this proxy may be too blunt.
- Special mora false-alarm checks are meaningful only if the calibration sample includes enough long vowels, sokuon, moraic nasals, and yoon.
- F0/pitch failures should usually suppress pitch feedback instead of lowering native ability claims.
- If alignment fallback is frequent, detailed mora/special-mora feedback must stay gated.
