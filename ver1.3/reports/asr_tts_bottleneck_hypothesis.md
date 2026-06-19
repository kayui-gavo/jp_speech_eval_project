# ASR/TTS Bottleneck Hypothesis

## Scope

This document separates likely ASR/TTS effects from acoustic scoring limitations. It is a benchmark hypothesis for offline A/B work, not a provider integration proposal. No external provider is called and no API key is required.

## Current Pipeline

The arbitrary-speech product path is:

1. audio recording and speech-region detection;
2. ASR candidate generation with automatic language detection;
3. user confirmation or correction of Japanese text;
4. kana/mora extraction and content/language/evidence guardrails;
5. acoustic feature extraction, mora alignment, weak-reference practice scoring;
6. evidence-based feedback rendering.

TTS is used as a synthetic practice/reference-audio aid in selected demo paths. It is not a verified human/native scoring ground truth.

## ASR Hypothesis

### ASR may improve

- arbitrary-speech transcript accuracy;
- kana and mora extraction stability;
- the quality of the candidate text shown for confirmation;
- Japanese/non-Japanese language evidence;
- random-English and hallucinated-Japanese guardrail decisions;
- weak-reference scoring eligibility after confirmation;
- fixed-target wrong-content rejection;
- coarse feedback location when reliable segments or word timestamps exist;
- alignment initialization for long pauses and special-mora sentences.

### ASR does not directly solve

- F0 extraction quality;
- pitch-range, local-movement, or smoothness feature quality;
- wrong accent-drop sensitivity;
- special-mora duration evidence when mora/phone alignment remains coarse;
- phoneme-level GOP or teacher-grade pronunciation correctness;
- learner-score calibration.

### Main risk

A stronger decoder can still emit fluent Japanese-looking hallucinations. Provider confidence is not permission to score. User confirmation, Japanese-likeness checks, minimum evidence, and content mismatch gates must remain in place.

## TTS Hypothesis

### TTS may improve

- demo reference-audio naturalness;
- shadowing and imitation UX;
- phrase-level rhythm and sentence-final intonation in playback;
- learner willingness to imitate the reference;
- synthetic smoke fixture realism;
- pseudo-reference timing quality when a provider supplies stable timestamps or phoneme durations.

### TTS must not be treated as solving

- verified human/native pitch ground truth;
- strict pitch-accent correctness;
- learner scoring calibration;
- wrong accent-drop detection;
- teacher-grade pronunciation assessment;
- reliable special-mora error labels for learner audio.

All TTS outputs must keep `provenance=synthetic_tts`, weak/unverified scoring status, and `strong_pitch_reference_allowed=false`.

## Dimension-Level Expected Impact

| Product area | Better ASR | Better TTS | Neither alone |
|---|---|---|---|
| Content confirmation | High impact | None | User confirmation still required |
| English/hallucination gate | Medium-high impact | None | Language/content guardrails remain required |
| Kana/mora extraction | High impact | Low | Confirmed text quality controls the frontend |
| Pronunciation clarity proxy | Indirect | Low | Needs phoneme/acoustic evidence for teacher-grade claims |
| Fluency | Low-medium with timestamps | Low | Acoustic pause/rate features remain primary |
| Special mora | Medium only with alignment timestamps | Low-medium for playback | Needs reliable learner-audio mora/phone alignment |
| Weak pitch naturalness | Eligibility only | UX/playback only | F0 feature design remains primary |
| Strict pitch accent | No direct solution | No direct solution | Needs verified references, labels, and accent-target validation |

## Working Priority

ASR deserves the first upper-bound experiment because current product safety and usability depend on candidate text, language detection, kana/mora stability, and hallucination handling. TTS is the second experiment for reference-audio UX. If the immediate target changes to special-mora localization, evaluate WhisperX/forced-alignment timestamps rather than ordinary transcription alone.
