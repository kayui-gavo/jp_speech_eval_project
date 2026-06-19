# ASR/TTS Upgrade Decision

## Decision Summary

Do not integrate a new provider into the formal runtime yet. Run an offline, same-audio replay first. ASR is the higher-priority upper-bound experiment; TTS is useful primarily for demo/reference-audio UX.

## 1. What an ASR upgrade is most likely to improve

- fewer manual corrections to arbitrary Japanese candidate text;
- better kana/mora stability after transcription;
- more reliable rejection of English, Latin-dominant, and wrong-sentence input;
- lower missing-transcript rate on noisy but intelligible Japanese;
- better coarse time anchors for pauses and feedback location when word timestamps are available.

It may indirectly increase the number of valid weak-reference practice evaluations. It should not directly raise acoustic dimension scores for the same confirmed text and audio.

## 2. What a TTS upgrade is most likely to improve

- naturalness and listenability of demo reference audio;
- shadowing and imitation experience;
- phrase rhythm and sentence-final intonation in playback;
- perceived product quality.

It may improve pseudo-reference timing if detailed timing metadata is supplied. The output remains synthetic and weak for scoring provenance.

## 3. What ASR/TTS cannot solve

- strict pitch-accent correctness;
- weak wrong-accent-drop sensitivity;
- F0 extraction failures;
- teacher-grade phoneme correctness;
- special-mora learner labels without reliable alignment and human review;
- large-scale calibration or proof of learning improvement.

## 4. Which should be evaluated first

Evaluate ASR first. The current bottleneck with the largest product blast radius is transcript/language/content evidence, not reference-audio timbre. Within ASR, compare ordinary transcription against a timestamp-capable alignment candidate because timestamps may help pause and mora-location diagnostics.

Evaluate TTS second if users report that reference audio sounds unnatural or is hard to imitate.

## 5. Is formal integration justified now?

No. Current benchmark rows are offline planning replays plus local-cache inventory. They do not establish real provider accuracy, latency, cost, privacy behavior, or stability.

## 6. Offline fixtures required before integration

For each ASR candidate, store results for the same WAV set:

- transcript, detected language, confidence;
- segment and word timestamps;
- model/provider/version;
- failure status and raw non-sensitive metadata;
- user-confirmed transcript for comparison.

For each TTS candidate, store:

- WAV file, sample rate, duration;
- provider/model/voice/style prompt;
- generation timestamp and synthetic provenance;
- optional phoneme/word timing;
- a human listening annotation for naturalness and imitation usefulness.

The audio set should contain native Japanese, learners, pronunciation variants, English, code switching, short/noisy speech, wrong sentences, hallucination controls, long pauses, and special-mora sentences.

## 7. Cost, privacy, and key risks

- External ASR/TTS sends user audio or text to a third party unless local inference is used.
- API keys must remain server-side and outside logs, fixtures, and the repository.
- Cost and latency can make confirmation UX worse even when accuracy improves.
- Provider/model revisions can silently change output and invalidate benchmark expectations.
- Retention, regional processing, and consent requirements must be reviewed before real user audio is transmitted.

## 8. Minimum upper-bound experiment

1. Select 20-30 existing local WAV files spanning the benchmark cases.
2. Run current ASR, one stronger ASR, and one timestamp-capable ASR offline or in a controlled one-time export.
3. Save provider-result JSON only; do not wire providers into runtime.
4. Replay the JSON through `audit_asr_bottleneck.py`.
5. Compare good-Japanese accept rate, bad-input reject rate, hallucination risk, kana similarity, correction burden, and timestamp usefulness.

For TTS, generate the ten planned sentences with current TTS and one candidate, then run `audit_tts_reference_quality.py` and conduct blinded listening review. Do not promote either TTS output to reliable pitch reference.

## Final Recommendation

Proceed with an ASR offline upper-bound A/B before any formal integration. Keep TTS as a separate UX experiment. Preserve all existing content gates, user confirmation, weak-reference wording, and synthetic-reference provenance.
