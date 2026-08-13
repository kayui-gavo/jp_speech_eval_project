# Product Truth

## Product contract

This repository implements a consumer Japanese speaking-practice product. It
is not a formal educational measurement instrument. The highest-level rule is:

> Meaningful, analyzable Japanese speech receives a 0–100 practice score.

Silence/no speech, unusable audio, obvious non-Japanese speech, meaningless
vocalization, and an ASR hallucination that fails transcript sanity may remain
unscored. F0 extraction failure, weak or TTS reference identity, unstable DTW,
fallback alignment, moderate recording noise, a short valid sentence, or a
plausible Japanese sentence that differs from the displayed target must not by
themselves remove the practice score.

## Score versus detail

Score availability and diagnostic-detail availability are separate decisions.
Reliability primarily controls confidence, wording, and the granularity of
feedback. Low F0 evidence suppresses lexical pitch claims only. Low alignment
evidence suppresses target-local and mora-local claims only. If a fixed target
mismatches but ASR finds plausible Japanese, the API uses broad Japanese
scoring and preserves the target mismatch as task metadata.

The 0–100 number is a product practice score. It is not a certified ability
score, native-likeness score, phone-correctness probability, or human rater
equivalent. Research reports provide evidence and limitations; they do not
override this product contract.

## Authority

The sole learner-facing total-score path is:

`apply_user_score_policy()` → `display_score` → `practice_score.value`

`raw_result.total_score` remains available for debug and research comparison
and must not be used as a UI fallback.
