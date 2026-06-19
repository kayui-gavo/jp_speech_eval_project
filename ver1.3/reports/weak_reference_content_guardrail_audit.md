# Weak-reference content guardrail audit

- generated_at: 2026-06-19T06:19:02+00:00
- source_csv: `/Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/results/calibration/weak_reference_native_likeness_audit.csv`
- scope: weak-reference arbitrary-sentence practice score guardrails only.
- strict fixed-reference, pitch scoring formula, aggregate weights, UI, ASR/TTS providers, and tone core dimensions are unchanged.

## Weak Overall Before Guardrail

| case | n | mean | min | p50 | max |
|---|---:|---:|---:|---:|---:|
| asr_confirmed_japanese_sentence | 1 | 91.0 | 91.0 | 91.0 | 91.0 |
| flat_pitch_control | 24 | 9.0 | 9.0 | 9.0 | 9.0 |
| janon_english | 6 | 79.3333 | 68.0 | 78.5 | 94.0 |
| janon_japanese | 6 | 71.8333 | 65.0 | 73.5 | 78.0 |
| jvs_native | 24 | 92.3333 | 80.0 | 94.0 | 97.0 |
| kana_mora_failed | 1 | 82.0 | 82.0 | 82.0 | 82.0 |
| latin_dominant_transcript | 1 | 88.0 | 88.0 | 88.0 | 88.0 |
| low_f0_coverage_control | 0 | None | None | None | None |
| random_english_latin | 1 | 93.0 | 93.0 | 93.0 | 93.0 |
| short_japanese_control | 1 | 95.0 | 95.0 | 95.0 | 95.0 |
| shuffled_random_pitch_control | 24 | 48.9167 | 44.0 | 48.0 | 67.0 |
| wrong_accent_drop_control | 24 | 88.875 | 75.0 | 91.0 | 96.0 |

## Weak Overall After Guardrail

| case | n | mean | min | p50 | max | no_score | capped |
|---|---:|---:|---:|---:|---:|---:|---:|
| asr_confirmed_japanese_sentence | 1 | 91.0 | 91.0 | 91.0 | 91.0 | 0/1 | 0/1 |
| flat_pitch_control | 24 | 9.0 | 9.0 | 9.0 | 9.0 | 0/24 | 0/24 |
| janon_english | 4 | 78.0 | 68.0 | 78.5 | 87.0 | 2/6 | 1/6 |
| janon_japanese | 2 | 70.0 | 70.0 | 70.0 | 70.0 | 4/6 | 2/6 |
| jvs_native | 24 | 92.3333 | 80.0 | 94.0 | 97.0 | 0/24 | 0/24 |
| kana_mora_failed | 0 | None | None | None | None | 1/1 | 0/1 |
| latin_dominant_transcript | 0 | None | None | None | None | 1/1 | 0/1 |
| low_f0_coverage_control | 0 | None | None | None | None | 0/24 | 24/24 |
| random_english_latin | 0 | None | None | None | None | 1/1 | 0/1 |
| short_japanese_control | 0 | None | None | None | None | 1/1 | 0/1 |
| shuffled_random_pitch_control | 24 | 48.9167 | 44.0 | 48.0 | 67.0 | 0/24 | 0/24 |
| wrong_accent_drop_control | 24 | 88.875 | 75.0 | 91.0 | 96.0 | 0/24 | 0/24 |

## Interpretation

- JANON English means English-L1 learners reading Japanese stimuli, not random English input. Its previous high mean was partly from short isolated-word rows where fluency/rhythm-like proxies could dominate weak overall.
- Latin-dominant, random-English, and kana/mora-failed controls now receive no weak overall practice score.
- Very short Japanese controls are treated as insufficient evidence rather than high-scoring arbitrary-sentence practice; this also explains most JANON no-score/capped rows.
- JVS native rows are long Japanese utterances and remain unaffected by the guardrail.
- Flat/random pitch controls remain below native because the pitch scoring formula was not changed.
- This is a content/language/evidence guardrail, not pitch calibration.

## Product Rule

- Show weak-reference practice score only when confirmed text is Japanese-like and has enough mora/F0 evidence.
- Hide weak overall for English, Latin-dominant, kana/mora extraction failures, content mismatch, and too-short utterances.
- Cap weak overall for low evidence cases instead of letting fluency/rhythm proxies create a high score.
