# Scoring Policy

## Practice-score eligibility

A practice score is available when speech is analyzable and is not rejected by
the recording/transcript sanity guard. ProductScorePolicy combines broad
pronunciation clarity, mora rhythm, and delivery fluency using one continuous
mapping. Pitch-accent and special-mora candidates do not drive the product
total.

`display_score`, `display_score_before_cap`, and `practice_score.value` are
derived from the same policy result. Raw evaluator totals are debug fields.

## Degradation matrix

| Condition | Practice score | Broad dimensions | Lexical pitch | Target-local special mora |
| --- | --- | --- | --- | --- |
| Normal fixed Japanese | yes | yes | only with verified target/evidence | only with fixed target/evidence |
| Fallback/low alignment | yes | yes | hidden | hidden |
| Low F0 coverage | yes | pronunciation/rhythm/fluency remain | hidden | alignment-dependent |
| Moderate recording degradation | yes, lower confidence | yes | commonly hidden | commonly hidden |
| Weak/TTS reference | yes | yes | hidden | hidden |
| Plausible Japanese target mismatch | yes, broad fallback | yes | hidden | hidden |
| Short valid Japanese | yes | yes | usually hidden | conservative |
| Silence/unusable audio | no | no | no | no |
| Transcript sanity rejection | no | no | no | no |

Renderer status has four meanings: `pass`, `practice_suggestion`, `retry`, and
`debug_only`. Missing local detail alone is never a reason for `debug_only`.
Demo/playback-only modes remain `debug_only`; unusable recordings are `retry`.
