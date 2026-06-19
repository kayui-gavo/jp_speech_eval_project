# Product scoring status summary

## Product position

This is an arbitrary-Japanese speaking practice demo. It provides cautious practice feedback after the user confirms Japanese text. It is not an examination system and does not claim teacher-grade pronunciation or pitch-accent correctness.

## What can be shown safely

| Dimension | Practice meaning | Main limitation |
|---|---|---|
| Pronunciation clarity | A practice reference based on recording clarity, mora alignment, and pronunciation stability. | It is a proxy, not phoneme-level GOP or a teacher judgement. |
| Rhythm / special mora | A practice reference based on timing evidence for long vowels, sokuon, moraic nasal, and related mora behavior. | A special-mora hint is shown only with enough evidence; weak evidence is suppressed. |
| Fluency | A practice reference based on speaking rate, pauses, speech duration, and continuity. | Short fluent-looking fragments can be misleading, so short inputs are gated. |
| Pitch movement | A native-likeness reference based on F0 coverage, movement range, smoothness, and stability. | It is not strict pitch-accent correctness and cannot reliably diagnose a wrong accent drop. |

## No-score and limited-score cases

- Random English, Latin-dominant confirmed text, failed kana/mora extraction, and explicit content mismatch receive no formal practice score.
- Very short Japanese receives no score; marginally short or low-evidence input may be capped.
- Low F0 coverage hides pitch naturalness and may cap overall practice feedback without hiding otherwise usable timing feedback.
- Debug/raw values remain available to developers but must not appear as formal user-facing scores.

## Feedback confidence

Relatively useful practice feedback includes recording quality, speech continuity, obvious long pauses, broad speaking-rate behavior, and strongly evidenced special-mora timing. Pitch feedback may cautiously describe flatness, unstable movement, or insufficient F0 evidence. It must not say that a specific pitch accent is wrong in weak-reference mode.

## Weak versus verified reference

ASR-confirmed arbitrary speech uses weak-reference native-likeness scoring. OpenJTalk may help derive kana/mora hints but is not a strict pitch ground truth. Verified fixed-reference mode remains separate: strict reference-based pitch comparison is allowed only when reliable human/native reference audio and provenance are available.

## Known limitations

- Wrong accent-drop counterfactuals remain close to native in the current weak score.
- JVS/JANON audits are useful sanity checks but are not broad demographic calibration.
- Learner special-mora feedback still needs more real-audio manual validation.
- ASR hallucinated Japanese is guarded more safely than before but is not exhaustively solved across languages and accents.
- The four dimensions share some timing and recording evidence, so they are not statistically independent abilities.

## Data needed next

The next meaningful improvement needs a balanced end-to-end smoke corpus: multiple native speakers, learners at several levels, verified transcripts, manually reviewed special-mora errors, real flat/random/wrong-accent controls, and teacher or native listener labels. Verified human reference audio is also required for strict fixed-sentence pitch work.
