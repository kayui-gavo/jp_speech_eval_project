# TTS pseudo-reference combo audit

- generated_at: 2026-06-20T08:13:10+00:00
- mode: fixed-reference pseudo-reference
- external provider calls: 0
- API keys required: no
- ASR setting: fixed/current or oracle transcript; ASR is not the comparison variable.

## Results

| combo | fixture | readable | duration | F0 coverage | timing | range | movement | final | native score | reliability | evidence |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| current_tts_pseudo_reference | local_fixture | True | 1.088 | 1.0 | approximate | 0.349 | 0.9172 | 0.98 | None | weak | audio_metrics_only_no_same_sentence_native_vs_tts_f0_score |
| best_candidate_tts_pseudo_reference | not_supplied | None | None | None | None | None | None | None | None | weak | not_measured |
| verified_native_reference_oracle | test_only_fixture | True | None | 0.9886 | lab_phone_mora | None | None | 86.25 | 85.75 | reliable | test_only_jvs_cross_speaker_oracle |

## Evidence boundary

- Current packaged pyopenjtalk fixture metrics are available for one sentence: duration 1.088 s, F0 coverage 1.0, timing `approximate`.
- The existing OpenJTalk symbolic-target JVS proxy mean is 67.5. It is not a TTS-audio-F0 pseudo-reference score and is not used as proof that current TTS reaches that ceiling.
- The test-only verified native cross-speaker oracle mean is 85.75. It is an upper-bound proxy, not packaged-demo evidence.
- No same-sentence best-candidate TTS WAV is present. Therefore better TTS versus current TTS, and either TTS versus the verified oracle, cannot yet be quantified.

## Answers

- Current TTS is a plausible strict/reference-based ceiling bottleneck because its provenance is weak and its mora timing is approximate. The present files do not prove the size of that bottleneck.
- Better TTS is not demonstrated to be materially better in this offline audit; the candidate fixture is missing.
- No synthetic TTS can become a reliable human/native pitch baseline solely by improving audio quality. It may improve imitation UX, timing/alignment behavior, and possibly pseudo-reference score consistency.
- A decisive A/B needs the same texts synthesized by current and candidate TTS, the same native/user recordings scored against each, plus listening review. Verified human/native reference remains the strict baseline.
