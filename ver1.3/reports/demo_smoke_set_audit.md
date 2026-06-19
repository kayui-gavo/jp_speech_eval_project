# Demo smoke set audit

- generated_at: 2026-06-19T18:28:35+00:00
- cases: 10
- passed: 10
- scope: product visibility and wording readiness; no calibration or scoring-formula change.
- evidence mixes real JVS/JANON audio audits, F0-only counterfactuals, and explicit policy fixtures.

## Results

| case | evidence | gate | visibility | pitch behavior | result |
|---|---|---|---|---|---|
| native_japanese_normal | JVS native audit | pass | practice scores visible | mean=92.33 | PASS |
| learner_japanese_normal | JANON English-L1 speakers reading Japanese | pass when evidence sufficient | visible=4/6; no_score=2; capped=1 | mean=61.2 | PASS |
| learner_japanese_with_special_mora_issue | existing product guardrail fixture | pass | practice score visible; special-mora hint gated | weak naturalness only | PASS |
| random_english_latin | weak-reference guardrail | no_score | no scores | hidden | PASS |
| latin_dominant_transcript | weak-reference guardrail | no_score | no scores | hidden | PASS |
| very_short_japanese | weak-reference guardrail | no_score | no scores | hidden | PASS |
| flat_pitch_control | JVS timing/content with flat F0 | pass | practice scores visible | mean=9.0 | PASS |
| random_pitch_control | JVS timing/content with shuffled F0 | pass | practice scores visible | mean=48.92 | PASS |
| low_f0_coverage | JVS timing/content with sparse F0 | pass with evidence cap | overall capped=24/24; pitch unavailable | unavailable | PASS |
| wrong_japanese_or_content_mismatch | content mismatch veto | veto | no scores | hidden | PASS |

## Product interpretation

- Normal native and evidence-sufficient learner Japanese can show practice scores.
- English, Latin-dominant, content-mismatch, and too-short controls do not surface misleading scores.
- Flat and shuffled F0 controls score below native pitch naturalness; low-F0 pitch remains unavailable.
- Special-mora learner behavior is policy-fixture coverage here, not new real-audio validation.
- Wrong accent-drop separation remains a known limitation, so weak-reference feedback must not claim a pitch-accent error.

## Readiness

Suitable for a practice demo with the documented wording and gates. Not suitable as an examination system or teacher-grade pitch-accent assessment.
