# User-facing evaluation policy

This demo exposes raw acoustic/debug metrics, but product UI should read `response.user_facing` first.

## UserFacingResult

- `status`: `pass`, `practice_suggestion`, `retry`, or `debug_only`
- `practice_score`: practice guidance for this recording, not validated pronunciation ability
- `confidence`: high / medium / low
- `summary_text`: one safe summary for learners
- `primary_suggestion_text`: at most one non-punitive suggestion
- `suggestion_type`: content / rhythm / fluency / special_mora / recording_quality / none
- `mode_notice`: explains reference mode limits
- `suppressed_reasons`: debug list explaining why risky feedback was hidden
- `debug`: raw scores and evidence for developers

## Practice score

`total_score` is kept internally as a debug proxy. The UI should display `practice_score`, described as:

> このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。

Formula:

```text
practice_score = sum(w_i * score_i) / sum(w_i)
for available and reliable dimensions only
```

Current user-facing weights exclude `pitch_accent_score` and exclude `special_mora_score` by default.

## Special mora policy

- `near_boundary=True`: accepted; hidden from user-facing feedback
- none / mild variation: accepted
- `too_long`: debug-only
- `too_short`: at most a gentle tip
- `sokuon`: blocked for strong user feedback
- `yoon`: debug-only for duration-based feedback
- weak-reference and Kanade/demo modes do not emit correctness feedback

Allowed wording:

- 全体としては問題ありません。より自然にするなら，「ラー」を少し長めに意識するとよいです。
- これは練習のための軽いアドバイスです。

Forbidden wording:

- 長音が間違っています
- 撥音が発音できていません
- 日本人のようではありません

## Limits

- practice_score is demo guidance, not validated pronunciation ability score
- total/prosody scores are proxy
- special mora is conservative suggestion, not correctness penalty
- JVS controls native false alarm risk but does not prove learner benefit
- JANON trend is not ground truth
- pitch accent feedback requires verified target and reliable F0
