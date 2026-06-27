# Demo flow smoke test report

- scenarios: 12
- passed: 12
- failed: 0

## Guardrails

- no raw score leakage to learner fields: True
- no Kanade scoring leakage: True
- no strong special mora correction by default: True
- weak-reference remains conservative: True
- fixed-reference is the most reliable path: documented in `reports/fixed_reference_demo_flow.md`

## Scenarios

- fixed_normal_pass: PASS (pass)
- fixed_fallback_degrades_to_four_practice_scores: PASS (practice_suggestion)
- fixed_poor_recording_retry: PASS (retry)
- fixed_near_boundary_special_mora_accepted: PASS (pass)
- fixed_clear_long_vowel_default_safe: PASS (pass)
- fixed_clear_long_vowel_explicit_gentle: PASS (practice_suggestion)
- weak_asr_unconfirmed: PASS (debug_only)
- weak_asr_confirmed: PASS (pass)
- weak_special_mora_suppressed: PASS (pass)
- kanade_reference_mocked: PASS (debug_only)
- kanade_excluded_from_scoring: PASS (debug_only)
- kanade_notice_visible: PASS (debug_only)
