# Demo flow smoke test report

- scenarios: 11
- passed: 11
- failed: 0

## Guardrails

- no raw score leakage to learner fields: True
- no Kanade scoring leakage: True
- no strong special mora correction by default: True
- weak-reference remains conservative: True
- fixed-reference is the most reliable path: documented in `reports/fixed_reference_demo_flow.md`

## Scenarios

- fixed_normal_pass: PASS (pass) 
- fixed_poor_recording_retry: PASS (retry) 
- fixed_near_boundary_special_mora_accepted: PASS (pass) 
- fixed_clear_long_vowel_flag_off: PASS (pass) 
- fixed_clear_long_vowel_flag_on_gentle: PASS (practice_suggestion) 
- weak_asr_unconfirmed: PASS (debug_only) 
- weak_asr_confirmed: PASS (pass) 
- weak_special_mora_suppressed: PASS (pass) 
- kanade_reference_mocked: PASS (debug_only) 
- kanade_excluded_from_scoring: PASS (debug_only) 
- kanade_notice_visible: PASS (debug_only) 
