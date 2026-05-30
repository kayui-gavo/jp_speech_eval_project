# JVS native sanity check

This report is a native false-alarm audit. It does not prove scoring validity, but it checks whether native speakers are being punished by current engineering thresholds.

## Summary
- n: 10
- display mean: 89.7
- pronunciation mean: 100.0
- fluency/rhythm proxy mean: 79.9
- retry rate: 0.0
- unscorable rate: 0.0
- alignment fallback rate: 0.0
- special mora false alarm proxy: 0.0

## Acceptance checks
- PASS: mora_clarity native mean >= 90
- FAIL: rhythm/fluency native mean >= 85
- PASS: unscorable rate <= 5%
- PASS: special mora false alarm proxy <= 5%

## Interpretation
- `score_pronunciation` is still a mora-timing/acoustic proxy, not phoneme correctness.
- F0/pitch failures should usually suppress pitch feedback instead of lowering native ability claims.
- If alignment fallback is frequent, detailed mora/special-mora feedback must stay gated.
