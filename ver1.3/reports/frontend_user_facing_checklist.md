# Frontend user-facing checklist

## Normal learner view

- Show `user_facing.status`.
- Show `user_facing.practice_score.label` and optionally `practice_score.value` when present.
- Show `user_facing.summary_text`.
- Show at most one `user_facing.primary_suggestion_text`.
- Show `user_facing.mode_notice` in weak-reference and Kanade modes.
- Treat `retry` as recording/content reliability, not pronunciation failure.

## Do not show by default

- raw `total_score`
- raw `prosody_score`
- raw DTW distance
- raw F0 statistics
- threshold values
- special-mora evidence cards
- debug suppression reasons

## Developer / teacher panel

Debug can be hidden behind developer mode:

- alignment confidence
- F0 coverage
- special-mora shadow decisions
- reliability gate detail
- scoring policy detail

## Mode-specific notes

- Fixed-reference is the current most reliable scoring path.
- ASR-confirmed weak-reference must show the confirmation/edit step before scoring.
- ASR+Kanade should emphasize playback reference; Kanade similarity is not a score.
