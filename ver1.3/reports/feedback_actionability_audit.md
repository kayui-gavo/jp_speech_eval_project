# Feedback actionability audit

- generated_at: 2026-06-19T18:49:56+00:00
- cases: 12
- passed: 12/12
- scope: feedback selection and wording only; scoring, aggregate, calibration, and providers are unchanged.
- smoke inputs are policy/component fixtures, not a new real-audio corpus.

## Existing feedback audit

- Previous rendering selected legacy strings without a common evidence/location/practice schema.
- Fluency rate/pause and high-confidence long-vowel/nasal timing have usable evidence.
- Raw pitch-reference strings could be too strong for weak-reference arbitrary speech if surfaced directly.
- Generic advice such as 'try again' did not always explain the evidence or the next practice action.
- Content/no-score and fallback gates already existed, but structured feedback now makes their suppression explicit.
- Sokuon/yoon and weak special-mora details remain gated; no unsupported location is invented.

## Smoke results

| case | expected | actual | location | tip | caveat | overclaim | result |
|---|---|---|---:|---:|---:|---:|---|
| normal_japanese_native | clear_recording | clear_recording | False | True | True | False | PASS |
| learner_japanese_ok | clear_recording | clear_recording | False | True | True | False | PASS |
| too_short_japanese | too_short | too_short | False | True | True | False | PASS |
| random_english_latin | content_mismatch | content_mismatch | False | True | True | False | PASS |
| flat_pitch | flat_pitch | flat_pitch | False | True | True | False | PASS |
| random_pitch | unstable_pitch | unstable_pitch | False | True | True | False | PASS |
| low_f0_coverage | low_f0_coverage | low_f0_coverage | False | True | True | False | PASS |
| long_pause_many | long_pause | long_pause | False | True | True | False | PASS |
| fast_rate | fast_rate | fast_rate | False | True | True | False | PASS |
| special_mora_long_vowel_short | special_mora_duration_issue | special_mora_duration_issue | True | True | True | False | PASS |
| special_mora_nasal_short | special_mora_duration_issue | special_mora_duration_issue | True | True | True | False | PASS |
| alignment_fallback | alignment_uncertain | alignment_uncertain | False | True | True | False | PASS |

## Remaining maturity limits

- Feedback wording is evidence-aware, but learner usefulness still needs listening tests with Chinese-L1 learners.
- Weak pitch can describe flatness or instability only; wrong accent drop remains insufficiently separated.
- Pause aggregates do not provide a reliable phrase-level time location, so no pause location is claimed.
- Special-mora location is shown only when the runtime decision is user-facing allowed and alignment evidence is strong.
- Normal positive feedback remains a practice observation, not proof of phoneme correctness.
