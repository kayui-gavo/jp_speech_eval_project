# Special mora alignment calibration report

This report answers whether the current offline alignment evidence is reliable enough to calibrate special-mora thresholds.

## Backend status
- existing_label: usable_special_mora_rate=1.0, failure=
- mfcc_dtw: usable_special_mora_rate=0.0, failure=not reliable enough for special mora threshold calibration

## Reliable counts by type
- long_vowel: coverage=117, reliable_count=117
- sokuon: coverage=12, reliable_count=12
- moraic_nasal: coverage=44, reliable_count=44
- yoon: coverage=66, reliable_count=66

## Threshold decision
- long_vowel: threshold can be generated from reliable alignment evidence.
- sokuon: insufficient reliable alignment evidence; do not update threshold.
- moraic_nasal: threshold can be generated from reliable alignment evidence.
- yoon: threshold can be generated from reliable alignment evidence.

## Next data need
- Aim for at least 30 reliable JVS instances per special mora type.
- Prioritize sentences with real sokuon, long vowels, moraic nasals, and yoon after MFA/TextGrid alignment is available.
