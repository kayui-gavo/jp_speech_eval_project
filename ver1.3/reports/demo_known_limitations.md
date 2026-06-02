# Demo known limitations

This document is part of the demo readiness pack. It prevents over-claiming.

## Current limitations

- `practice_score` is demo guidance, not validated pronunciation ability.
- `total_score` and `prosody_score` are proxy metrics.
- fixed-reference is currently the most reliable mode.
- ASR-generated reference requires user confirmation.
- ASR raw text must not be used as a scoring target directly.
- Kanade is playback reference only, not scoring ground truth.
- Similarity to Kanade audio is not pronunciation correctness.
- TTS reference is a pseudo-reference, not ground truth.
- Special mora feedback is conservative and mostly suppressed by default.
- long_vowel / moraic_nasal may become gentle tips only after manual inspection.
- sokuon remains blocked for user-facing correction.
- yoon remains debug-only / blocked for duration-based correction.
- pitch accent requires verified target and reliable F0.
- Targets generated only by pyopenjtalk should suppress pitch-accent feedback.
- JANON trend is not ground truth.
- JVS sanity checks reduce native false-alarm risk but do not prove learner benefit.
- Human validation is still insufficient.

## What the demo can safely claim

- It can run a fixed-reference Japanese reading practice flow.
- It can return a learner-safe `response.user_facing` object.
- It can keep raw acoustic/debug metrics out of the normal learner UI.
- It can require ASR confirmation before weak-reference scoring.
- It can use Kanade as personalized reference playback without scoring voice similarity.

## What the demo should not claim

- It should not claim scientific pronunciation ability measurement.
- It should not claim native-level scoring.
- It should not claim full pitch-accent diagnosis for unverified targets.
- It should not claim special-mora correction is fully validated.
- It should not claim ASR-generated text is a reliable target without confirmation.
