# ASR/TTS Upgrade Priority Summary

## Mode-Specific Decision

| product path | first provider priority | why | provider upgrade cannot solve alone |
|---|---|---|---|
| arbitrary speech / ASR-confirmed weak-reference | ASR | Transcript/language evidence controls Japanese-content eligibility, kana/mora stability, correction burden, and possible feedback localization | F0 extraction, rhythm proxy validity, learner calibration, feedback pedagogy, strict accent correctness |
| fixed-reference / TTS pseudo-reference | TTS | Synthetic reference audio, F0, and timing can constrain reference-based alignment/prosody and imitation quality | Reliable human/native provenance, wrong-drop validation, strict-score calibration |
| verified human/native fixed-reference | Neither as the strict baseline | Verified reference audio/timing is already the scoring reference; ASR/TTS are auxiliary | Cross-speaker validation, calibration, and instructional validity |

## Evidence Summary

- The ASR combination audit is an offline fixture replay. Its candidate slots look better than current ASR, but this is a hypothesis, not an empirical provider ranking.
- The current TTS inventory has one packaged pyopenjtalk fixture with weak provenance and approximate equal-mora timing.
- No same-sentence best-candidate TTS output is available, so the project cannot yet claim that a candidate TTS materially raises the fixed-reference ceiling.
- Test-only JVS verified reference audio produces a stronger upper-bound result than heuristic OpenJTalk targets, but it cannot be transferred to unrelated packaged sentences.

## Answers

1. **Main arbitrary-speech product:** test ASR first. It most directly affects whether valid Japanese receives a practice score and whether bad input is rejected safely.
2. **Fixed pseudo-reference path:** test TTS first, while preserving synthetic/weak provenance.
3. **Not solved by either provider:** strict pitch-accent truth, reliable wrong accent-drop detection, score calibration, teaching effectiveness, and verified packaged reference assets.
4. **One real external A/B:** run the ASR A/B first on the exact smoke audio, preserving raw transcript, language/confidence, and timestamps. This targets the main product path.
5. **One minimum offline replay next:** collect same-audio ASR outputs from current and one candidate, then replay the existing content gate. The present hypothetical transcript plan is not enough for provider selection.
6. **Strict pitch accent:** still not solvable by a provider upgrade alone. Better TTS may improve the pseudo-reference ceiling and UX, but synthetic audio is not human/native ground truth.

## Release Boundary

No runtime provider integration, scoring formula, feedback policy, aggregate, UI structure, or calibration is changed by these audits.
