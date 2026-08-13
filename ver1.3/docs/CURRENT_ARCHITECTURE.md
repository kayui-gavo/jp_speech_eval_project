# Current Architecture

## Request and result flow

1. `SpeechEvaluationClient.evaluate()` dispatches to `evaluate_mode()`.
2. Fixed-reference evaluation produces acoustic metrics, alignment evidence,
   content state, raw dimension scores, and research/debug fields.
3. A fixed-target mismatch with a plausible Japanese ASR transcript routes to
   `transcript_assisted_light` and is labeled
   `reference_mismatch_general_japanese`. The original target result remains
   under `fixed_reference_debug` and `task_content_match`.
4. Optional shadow candidates run behind disabled-by-default feature flags and
   write only to `details.shadow`.
5. `policy_from_result()` defines fixed versus broad claim permissions.
6. `evaluate_reliability_gate()` controls dimension-local detail availability.
7. `apply_user_score_policy()` is the only product total-score authority.
8. `render_user_facing_result()` maps that decision into the stable public/UI
   contract. It never recomputes a total.

## Fixed and broad modes

Fixed modes have a known target/reference and may expose verified target-local
diagnostics when evidence supports them. Broad modes can expose pronunciation
clarity proxies, rhythm, fluency, and broad intonation, but not lexical pitch
correctness or target-local special-mora correction.

## Reliability domains

The gate reports independent audio, content, alignment, pitch, and
special-mora reliability. Broad modes do not depend on fixed-reference
alignment. F0 coverage affects pitch only.

## Shadow boundary

SSL reference distance, special-mora v2 ROI evidence, phrase intonation, and
accent-nucleus candidates are audit-only. They are lazy, exception-isolated,
disabled by default, separately timed, and never feed ProductScorePolicy.
