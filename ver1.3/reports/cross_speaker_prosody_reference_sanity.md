# Cross-speaker prosody reference sanity

- generated_at: 2026-06-18T16:57:20+00:00
- jvs_root: `/Users/ryukayuiii/Documents/jp_speech_eval_project/JVS`
- requested_pairs: 12
- rows: 72
- method: semi-audio; real JVS audio and lab timing provide mora-level F0, then counterfactuals replace F0 only.

## Prosody Score by Mode

| mode | n | mean | min | p50 | max |
|---|---:|---:|---:|---:|---:|
| flat_pitch_correct_content | 12 | 47.5 | 44.0 | 47.0 | 54.0 |
| low_f0_coverage_correct_content | 12 | 50.0 | 50.0 | 50.0 | 50.0 |
| native_cross_speaker_openjtalk_target | 12 | 71.5833 | 58.0 | 72.0 | 83.0 |
| native_cross_speaker_reference_audio_f0_cache | 12 | 87.4167 | 81.0 | 87.0 | 96.0 |
| shuffled_random_pitch_correct_content | 12 | 54.8333 | 44.0 | 56.0 | 63.0 |
| wrong_accent_drop_correct_content | 12 | 83.5833 | 75.0 | 85.0 | 89.0 |

## Paired Deltas

| comparison | n | mean | min | p50 | max |
|---|---:|---:|---:|---:|---:|
| native_cross_speaker_reference_audio_f0_cache - native_cross_speaker_openjtalk_target | 12 | 15.8333 | 1.0 | 17.0 | 33.0 |
| native_cross_speaker_reference_audio_f0_cache - flat_pitch_correct_content | 12 | 39.9167 | 29.0 | 40.0 | 50.0 |
| native_cross_speaker_reference_audio_f0_cache - shuffled_random_pitch_correct_content | 12 | 32.5833 | 22.0 | 33.0 | 40.0 |
| native_cross_speaker_reference_audio_f0_cache - wrong_accent_drop_correct_content | 12 | 3.8333 | 0.0 | 4.0 | 11.0 |
| native_cross_speaker_reference_audio_f0_cache - low_f0_coverage_correct_content | 12 | 37.4167 | 31.0 | 37.0 | 46.0 |

## Interpretation

- Cross-speaker reference-audio F0 target scores higher than OpenJTalk target in this sample.
- Flat counterfactual is lower than native reference target.
- Shuffled/random counterfactual is lower than native reference target.
- Wrong-drop counterfactual is lower than native reference target.
- Content is controlled by same JVS transcript; no ASR/TTS/content-gate behavior is changed by this audit.
- This audit does not make calibration active.

## Calibration Readiness

Not ready. This sanity check is stronger than component-only tests, but calibration should wait for verified packaged reference audio and broader cross-speaker/audio-level negative controls.
