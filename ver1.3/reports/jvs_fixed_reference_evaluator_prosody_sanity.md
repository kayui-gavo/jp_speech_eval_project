# JVS fixed-reference evaluator prosody sanity

- generated_at: 2026-06-19T02:42:16+00:00
- test_cache_prefix: `/Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/outputs/test_jvs_prosody_reference_cache/jvs001_VOICEACTRESS100_001`
- scope: test-only verified native reference path; no packaged demo target is promoted.

## Evaluator Path Checks

| case | prosody | pitch_target_source | reliability | content_status | alignment | display_score | visible_prosody |
|---|---:|---|---|---|---|---:|---:|
| verified_jvs_sidecar_fixed_reference | 46 | reference_audio_f0_cache | reliable | pass | cached_dtw_fallback_equal | None | None |
| same_audio_text_openjtalk_evaluator_baseline | 45 | openjtalk_accent_phrase_chain | heuristic | unknown | dtw | 89 | None |

## Prosody Score Summary

| case | n | mean | min | p50 | max |
|---|---:|---:|---:|---:|---:|
| flat_pitch_correct_content | 1 | 47.0 | 47.0 | 47.0 | 47.0 |
| low_f0_coverage_correct_content | 1 | 50.0 | 50.0 | 50.0 | 50.0 |
| native_cross_speaker_reference_audio_f0_cache | 1 | 91.0 | 91.0 | 91.0 | 91.0 |
| same_audio_text_openjtalk_evaluator_baseline | 1 | 45.0 | 45.0 | 45.0 | 45.0 |
| shuffled_random_pitch_correct_content | 1 | 60.0 | 60.0 | 60.0 | 60.0 |
| verified_jvs_sidecar_fixed_reference | 1 | 46.0 | 46.0 | 46.0 | 46.0 |
| wrong_accent_drop_correct_content | 1 | 85.0 | 85.0 | 85.0 | 85.0 |

## Interpretation

- Evaluator successfully reads the test-only verified JVS sidecar as `reference_audio_f0_cache`.
- Content gate passes for this cross-speaker JVS fixed-reference sample.
- The OpenJTalk row is an evaluator-level text baseline, not a fixed-reference cache path.
- Flat/random/low-F0 rows are semi-audio lab-F0 controls, not waveform-manipulated audio.
- This audit does not enable calibration or change runtime scoring.
