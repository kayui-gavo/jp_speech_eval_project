# API user-facing contract

Product UI should read `response.user_facing` first. Raw scores remain available for developers, teachers, and manual review, but they should not be the default learner display.

## Response shape

```json
{
  "ok": true,
  "mode": "reference",
  "user_facing": {
    "status": "pass | practice_suggestion | retry | debug_only",
    "practice_score": {
      "value": 93,
      "label": "良好",
      "explanation": "練習用の目安です。発音能力そのものを厳密に評価するものではありません。"
    },
    "confidence": "high | medium | low | unscorable",
    "summary_text": "全体としてよくできています。",
    "primary_suggestion_text": null,
    "suggestion_type": "none | rhythm | fluency | special_mora | recording | content",
    "mode_notice": "fixed-reference mode: verified target に基づく練習確認です。",
    "suppressed_reasons": []
  },
  "raw_result": {
    "total_score": 88,
    "prosody_score": 80,
    "details": {}
  }
}
```

## User-facing fields

- `status`: learner-facing state. `retry` means recording/content reliability failed, not pronunciation failure.
- `practice_score`: practice guidance for this recording. It is not a validated pronunciation ability score.
- `confidence`: reliability gate result.
- `summary_text`: one short safe message.
- `primary_suggestion_text`: at most one actionable suggestion.
- `suggestion_type`: category of that suggestion.
- `mode_notice`: required in weak-reference and Kanade modes.
- `suppressed_reasons`: developer-friendly list explaining why risky feedback was hidden.

## Debug fields

`raw_result` and `user_facing.debug` may include:

- raw total/prosody/pronunciation/fluency proxies
- alignment mode and confidence
- F0 coverage
- special-mora shadow decisions
- reliability gate detail
- scoring policy detail

These are for inspection, not normal C-end display.

## Product rules

- UI should not show raw `total_score` as the main score.
- UI should not show `prosody_score` as pronunciation correctness.
- UI should not show raw DTW, raw F0, thresholds, or debug evidence to normal users.
- Weak-reference results must keep a weak-reference notice visible.
- Kanade results must keep a playback-only notice visible.
- Debug panels should be hidden behind a developer/teacher toggle.
# Authority note

The learner-facing total has one authority:
`apply_user_score_policy() -> display_score -> practice_score.value`.
`raw_result.total_score` is debug/research output and is never a UI fallback.
Unavailable target-local detail does not imply an unavailable practice score.
