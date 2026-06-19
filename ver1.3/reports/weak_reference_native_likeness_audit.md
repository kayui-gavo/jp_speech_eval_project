# Weak-reference native-likeness audit

- generated_at: 2026-06-19T04:23:00+00:00
- jvs_root: `/Users/ryukayuiii/Documents/jp_speech_eval_project/JVS`
- janon_root: `/Users/ryukayuiii/Documents/jp_speech_eval_project/JANON`
- scope: diagnostic/practice scoring only; no strict pitch calibration is activated.

## Weak Prosody Naturalness

| case | n | mean | min | p50 | max |
|---|---:|---:|---:|---:|---:|
| flat_pitch_control | 24 | 9.0 | 9.0 | 9.0 | 9.0 |
| janon_english | 5 | 61.2 | 10.0 | 77.0 | 95.0 |
| janon_japanese | 3 | 70.3333 | 54.0 | 76.0 | 81.0 |
| jvs_native | 24 | 92.3333 | 80.0 | 94.0 | 97.0 |
| low_f0_coverage_control | 0 | None | None | None | None |
| shuffled_random_pitch_control | 24 | 48.9167 | 44.0 | 48.0 | 67.0 |
| wrong_accent_drop_control | 24 | 88.875 | 75.0 | 91.0 | 96.0 |

## Weak Overall Practice Score

| case | n | mean | min | p50 | max |
|---|---:|---:|---:|---:|---:|
| flat_pitch_control | 0 | None | None | None | None |
| janon_english | 6 | 79.3333 | 68.0 | 78.5 | 94.0 |
| janon_japanese | 6 | 71.8333 | 65.0 | 73.5 | 78.0 |
| jvs_native | 0 | None | None | None | None |
| low_f0_coverage_control | 0 | None | None | None | None |
| shuffled_random_pitch_control | 0 | None | None | None | None |
| wrong_accent_drop_control | 0 | None | None | None | None |

## OpenJTalk Strict Target Comparison

| case | n | mean | min | p50 | max |
|---|---:|---:|---:|---:|---:|
| flat_pitch_control | 0 | None | None | None | None |
| janon_english | 6 | 54.1667 | 32.0 | 50.5 | 81.0 |
| janon_japanese | 6 | 63.5 | 50.0 | 51.0 | 98.0 |
| jvs_native | 24 | 70.75 | 56.0 | 71.5 | 83.0 |
| low_f0_coverage_control | 0 | None | None | None | None |
| shuffled_random_pitch_control | 0 | None | None | None | None |
| wrong_accent_drop_control | 0 | None | None | None | None |

## Interpretation

- JVS native weak prosody mean: 92.3333.
- Flat control mean: 9.0; random/shuffled control mean: 48.9167.
- Native is no longer evaluated by strict OpenJTalk contour mismatch in weak-reference practice mode.
- Flat/random controls are not lifted together with native when the weak score is used.
- Low-F0 unavailable rows: 24/24.
- Wrong-drop mean: 88.875; this remains a known limitation and is not treated as strict accent correctness.
- OpenJTalk is used only as kana/mora/accent hint in arbitrary-sentence practice, not as reliable pitch target.
- JANON rows are a small trend sanity only, not calibration.

## Product Policy

- Arbitrary confirmed text can receive weak-reference native-likeness practice scores.
- Pitch feedback should say 音高变化参考 / may be flat / sentence-final intonation may be unclear.
- It must not claim teacher-grade pitch accent correctness without reliable reference audio.
- Verified fixed-reference scoring remains separate and unchanged.

## Calibration Readiness

Not ready. This establishes a safer arbitrary-sentence practice path, but does not calibrate strict pitch accent correctness.
