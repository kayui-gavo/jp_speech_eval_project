# JVS Verified Pitch Demo Summary

- generated_at: 2026-06-19T03:11:42+00:00
- test_only_cache_dir: `/Users/ryukayuiii/Documents/jp_speech_eval_project/ver1.3/results/test_fixtures/jvs_verified_pitch_demo`
- selected_items: 4
- scope: test-only JVS verified fixed-reference pitch demo; no packaged demo target is promoted.
- selection_note: JVS parallel100 has no very short sentence in this local set; selected items are the shortest medium-length candidates with good F0 coverage and pitch movement.

## Selected JVS Items

| utterance_id | target_text | mora_count | reference_speaker | user_speaker |
|---|---|---:|---|---|
| VOICEACTRESS100_014 | クィーンズアベニューアルファに所属している。 | 20 | jvs001 | jvs002 |
| VOICEACTRESS100_091 | 同母姉に、スウェーデン王妃、ジョゼフィーヌがいる。 | 22 | jvs001 | jvs002 |
| VOICEACTRESS100_033 | 芸能プロダクション、アミューズのグループ企業。 | 22 | jvs001 | jvs002 |
| VOICEACTRESS100_034 | 長母音を省略して、エリュシオンとも表記される。 | 25 | jvs001 | jvs002 |

## Semi-Audio Prosody Score Summary

| mode | n | mean | min | p50 | max |
|---|---:|---:|---:|---:|---:|
| flat_pitch_correct_content | 4 | 48.0 | 43.0 | 51.0 | 51.0 |
| low_f0_coverage_correct_content | 4 | 50.0 | 50.0 | 50.0 | 50.0 |
| native_cross_speaker_openjtalk_target | 4 | 67.5 | 55.0 | 75.0 | 77.0 |
| native_cross_speaker_reference_audio_f0_cache | 4 | 85.75 | 78.0 | 88.0 | 91.0 |
| shuffled_random_pitch_correct_content | 4 | 55.75 | 49.0 | 59.0 | 60.0 |
| wrong_accent_drop_correct_content | 4 | 82.5 | 75.0 | 84.0 | 88.0 |

## Interpretation

- reference_audio_f0_cache mean: 85.75; OpenJTalk mean: 67.5.
- Reference-audio F0 target beats OpenJTalk on this selected JVS demo set.
- Flat mean: 48.0; random/shuffled mean: 55.75.
- Flat/random controls are lower than the native reference contour.
- Wrong-drop mean: 82.5.
- Wrong-drop remains weakly separated; this is a known limitation of the current component design.
- Low-F0 rows are marked as insufficient evidence when voiced mora coverage is below threshold.
- Evaluator rows confirm the sidecar path reads `reference_audio_f0_cache`; user-facing gates such as fallback alignment remain active.
- `tone_score` is not part of the core four dimensions.

## Product Conclusion

This demonstrates the pitch/prosody pipeline upper bound when a real human/native reference contour with lab timing exists. It cannot be directly used for the packaged short-sentence demo because those targets still lack verified human/native reference audio and reliable timing sidecars.

## Next Minimal Product Action

Record/import verified native or teacher reference audio for the packaged targets, add non-fallback mora timing, then rebuild reliable sidecars. Calibration remains inactive.
