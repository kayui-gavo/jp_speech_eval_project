# Fixed-reference demo flow

Fixed-reference is the most reliable path in the current demo because the target text is known before recording.

## Input

- `target_id`
- `target_text`
- `user_audio`

Demo targets live in `data/demo_fixed_targets.json`.

Initial targets:

- ラーメンをください
- コーヒーをください
- すみません
- もう一度お願いします
- 駅までお願いします

## Flow

1. Load target metadata: text, kana, mora sequence, verified level, optional reference audio.
2. Generate or load a pseudo-reference if needed.
3. Analyze user audio: recording quality, content match, alignment, rhythm, F0 coverage, special-mora shadow decisions.
4. Apply reliability gate before learner-facing feedback.
5. Render `response.user_facing`.
6. Keep raw scores and debug evidence available only for developer/teacher inspection.

## User-facing output

The C-end UI should show:

- status
- practice score label
- summary text
- one suggestion at most
- mode notice when needed

The C-end UI should not show raw total score, raw DTW, raw F0, thresholds, or debug scores by default.

## Policy

- `practice_score` is a training guide, not a validated pronunciation ability score.
- `total_score`, `prosody_score`, and special-mora score are debug/proxy values.
- Special-mora output is conservative. Borderline native-like variation should be accepted.
- Pitch-accent feedback requires a verified target (`ojad_checked` or `human_checked`) plus reliable F0 and alignment.
- If a target is only `auto_pyopenjtalk`, pitch feedback is suppressed.
