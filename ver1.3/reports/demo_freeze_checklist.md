# Demo freeze checklist

## Before demo

- Run `../.venv/bin/python -m pytest tests`.
- Run `../.venv/bin/python scripts/run_demo_flow_smoke_tests.py`.
- Run `../.venv/bin/python scripts/generate_demo_api_examples.py`.
- Run `../.venv/bin/python scripts/run_demo_readiness_check.py`.
- Check `reports/demo_flow_smoke_test_report.md`.
- Check `reports/demo_readiness_report.md`.
- Check `reports/demo_api_response_examples.md`.
- Confirm user-facing examples do not show raw total/prosody/DTW/F0.
- Confirm Kanade is not scoring.
- Confirm special mora default is off.
- Confirm weak-reference notices are visible.
- Read `reports/demo_known_limitations.md`.

## During demo

- Start with fixed-reference.
- Then show ASR-confirmed weak-reference.
- Finally show ASR+Kanade.
- Do not emphasize raw `total_score`.
- Explain `practice_score` carefully as practice guidance.
- Do not claim scientific validation.
- If a result is retry/debug-only, frame it as reliability control, not user failure.

## After demo

- Collect feedback.
- Note confusing UI wording.
- Note false suggestions.
- Note ASR confirmation problems.
- Note Kanade experience comments.
- Decide which feedback remains debug-only.
