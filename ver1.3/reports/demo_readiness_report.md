# Demo readiness report

- status: warning
- checks: 23
- ready: 22
- warning: 1
- blocked: 0

## Current demo scope

- fixed-reference reading practice is the most reliable path.
- ASR-confirmed weak-reference can be shown as conservative practice support.
- ASR+Kanade can be shown as personalized playback reference; it is not a scoring ground truth.
- Normal UI should read `response.user_facing`; debug/raw metrics stay hidden by default.

## Do not claim

- Do not claim validated pronunciation ability scoring.
- Do not claim raw total/prosody scores are scientific correctness.
- Do not claim Kanade evaluates pronunciation correctness.
- Do not claim ASR-generated reference is a strong target before user confirmation.
- Do not claim special-mora feedback is fully validated.

## Checks

- ready: file:demo_fixed_targets /Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/data/demo_fixed_targets.json
- ready: file:user_facing_messages /Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/configs/user_facing_messages_ja.json
- ready: file:api_contract /Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/docs/api_user_facing_contract.md
- ready: file:fixed_reference_flow /Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/reports/fixed_reference_demo_flow.md
- ready: file:asr_confirmed_flow /Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/reports/asr_confirmed_reference_flow.md
- ready: file:asr_kanade_flow /Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/reports/asr_kanade_demo_flow.md
- ready: file:frontend_checklist /Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/reports/frontend_user_facing_checklist.md
- ready: file:known_limitations /Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/reports/demo_known_limitations.md
- ready: file:presentation_script /Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/reports/demo_presentation_script.md
- ready: file:api_examples_report /Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/reports/demo_api_response_examples.md
- ready: file:api_examples_json /Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/results/demo_readiness/demo_api_examples.json
- ready: file:human_validation_plan /Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/reports/minimal_human_validation_plan.md
- ready: file:freeze_checklist /Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/reports/demo_freeze_checklist.md
- ready: demo_targets_nonempty 5 targets
- ready: pitch_feedback_requires_verified_target 1 verified targets
- ready: user_facing_messages_nonempty 9 messages
- ready: api_examples_contain_user_facing
- ready: no_raw_debug_score_leakage
- ready: demo_smoke_tests_pass
- warning: pytest_command_available Run `../.venv/bin/python -m pytest tests` before demo freeze.
- ready: kanade_playback_only_policy Kanade must stay demo_only/playback_only and excluded from correctness scoring.
- ready: special_mora_default_hidden Special mora remains shadow/debug unless explicitly enabled.
- ready: weak_reference_conservative ASR-generated reference requires user confirmation and weak-reference notice.
