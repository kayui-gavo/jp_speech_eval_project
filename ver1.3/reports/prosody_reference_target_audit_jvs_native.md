# Prosody reference/target/alignment audit: JVS native

- generated_at: 2026-06-18T15:58:56.862550+00:00
- jvs_root: `/Users/ryukayuiii/Documents/jp_speech_eval_project/JVS`
- rows: 136
- speakers: 1
- utterances_per_speaker: 17
- sample_rate: 16000
- pitch_target_source counts: `{'openjtalk_accent_phrase_chain': 34, 'tts_reference': 102}`
- alignment_mode counts: `{'equal': 17, 'lab_or_reference_alignment': 119}`

## Prosody Score by Target Mode

| target_mode | n | mean | min | p10 | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_openjtalk_target | 17 | 66.0588 | 52.0 | 53.6 | 67.0 | 81.0 | 82.0 |
| current_openjtalk_target_lab_alignment | 17 | 74.0588 | 55.0 | 68.0 | 75.0 | 79.8 | 81.0 |
| flat_user_contour_vs_self_target | 17 | 46.5294 | 41.0 | 42.2 | 47.0 | 50.0 | 54.0 |
| self_oracle_contour_target | 17 | 99.4706 | 97.0 | 97.0 | 100.0 | 100.0 | 100.0 |
| shifted_self_target_+1_mora | 17 | 71.0 | 63.0 | 64.0 | 70.0 | 77.6 | 83.0 |
| shifted_self_target_-1_mora | 17 | 72.0588 | 63.0 | 65.2 | 73.0 | 79.8 | 82.0 |
| shuffled_user_contour_vs_self_target | 17 | 55.1765 | 41.0 | 45.2 | 57.0 | 63.0 | 64.0 |
| smoothed_self_target | 17 | 92.9412 | 87.0 | 89.4 | 93.0 | 96.4 | 97.0 |

## F0 Coverage by Target Mode

| target_mode | n | mean | min | p10 | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_openjtalk_target | 17 | 0.9009 | 0.7857 | 0.8026 | 0.9302 | 0.9671 | 1.0 |
| current_openjtalk_target_lab_alignment | 17 | 0.9928 | 0.9667 | 0.9752 | 1.0 | 1.0 | 1.0 |
| flat_user_contour_vs_self_target | 17 | 0.9928 | 0.9667 | 0.9752 | 1.0 | 1.0 | 1.0 |
| self_oracle_contour_target | 17 | 0.9928 | 0.9667 | 0.9752 | 1.0 | 1.0 | 1.0 |
| shifted_self_target_+1_mora | 17 | 0.9928 | 0.9667 | 0.9752 | 1.0 | 1.0 | 1.0 |
| shifted_self_target_-1_mora | 17 | 0.9928 | 0.9667 | 0.9752 | 1.0 | 1.0 | 1.0 |
| shuffled_user_contour_vs_self_target | 17 | 0.9928 | 0.9667 | 0.9752 | 1.0 | 1.0 | 1.0 |
| smoothed_self_target | 17 | 0.9928 | 0.9667 | 0.9752 | 1.0 | 1.0 | 1.0 |

## Contour Correlation by Target Mode

| target_mode | n | mean | min | p10 | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_openjtalk_target | 17 | 0.1943 | -0.2535 | -0.1097 | 0.2116 | 0.5967 | 0.6999 |
| current_openjtalk_target_lab_alignment | 17 | 0.4387 | 0.0652 | 0.2746 | 0.4718 | 0.5793 | 0.6147 |
| flat_user_contour_vs_self_target | 17 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| self_oracle_contour_target | 17 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| shifted_self_target_+1_mora | 17 | 0.556 | 0.2514 | 0.3152 | 0.5839 | 0.7235 | 0.7874 |
| shifted_self_target_-1_mora | 17 | 0.556 | 0.2514 | 0.3152 | 0.5839 | 0.7235 | 0.7874 |
| shuffled_user_contour_vs_self_target | 17 | -0.0365 | -0.3986 | -0.2938 | -0.0193 | 0.1942 | 0.2198 |
| smoothed_self_target | 17 | 0.9529 | 0.8963 | 0.9236 | 0.9543 | 0.9795 | 0.984 |

## Transition Agreement by Target Mode

| target_mode | n | mean | min | p10 | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_openjtalk_target | 17 | 0.7377 | 0.5588 | 0.5982 | 0.725 | 0.8688 | 0.9241 |
| current_openjtalk_target_lab_alignment | 17 | 0.813 | 0.6531 | 0.6877 | 0.7941 | 0.9387 | 1.0 |
| flat_user_contour_vs_self_target | 17 | 0.2543 | 0.0564 | 0.1067 | 0.277 | 0.3229 | 0.4742 |
| self_oracle_contour_target | 17 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| shifted_self_target_+1_mora | 17 | 0.5977 | 0.4749 | 0.5183 | 0.5905 | 0.6951 | 0.7308 |
| shifted_self_target_-1_mora | 17 | 0.5881 | 0.518 | 0.5408 | 0.5755 | 0.6386 | 0.7554 |
| shuffled_user_contour_vs_self_target | 17 | 0.5865 | 0.409 | 0.5014 | 0.5892 | 0.6657 | 0.6884 |
| smoothed_self_target | 17 | 0.8941 | 0.8227 | 0.8354 | 0.8934 | 0.9566 | 0.9797 |

## HL Match by Target Mode

| target_mode | n | mean | min | p10 | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_openjtalk_target | 17 | 0.574 | 0.4167 | 0.4309 | 0.5909 | 0.7105 | 0.8 |
| current_openjtalk_target_lab_alignment | 17 | 0.7018 | 0.5217 | 0.5807 | 0.7097 | 0.7795 | 0.8571 |
| flat_user_contour_vs_self_target | 17 | 0.5308 | 0.4571 | 0.4636 | 0.5357 | 0.5875 | 0.5962 |
| self_oracle_contour_target | 17 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| shifted_self_target_+1_mora | 17 | 0.7097 | 0.5714 | 0.6437 | 0.6957 | 0.7773 | 0.85 |
| shifted_self_target_-1_mora | 17 | 0.7084 | 0.5714 | 0.6223 | 0.7059 | 0.7773 | 0.85 |
| shuffled_user_contour_vs_self_target | 17 | 0.4887 | 0.2857 | 0.3539 | 0.4857 | 0.6128 | 0.6538 |
| smoothed_self_target | 17 | 0.9488 | 0.8627 | 0.9043 | 0.9552 | 1.0 | 1.0 |

## Paired Score Deltas

| comparison | n | mean | min | p50 | max |
|---|---:|---:|---:|---:|---:|
| self_oracle_contour_target - current_openjtalk_target | 17 | 33.4118 | 18.0 | 33.0 | 48.0 |
| self_oracle_contour_target - current_openjtalk_target_lab_alignment | 17 | 25.4118 | 19.0 | 24.0 | 45.0 |
| self_oracle_contour_target - smoothed_self_target | 17 | 6.5294 | 3.0 | 6.0 | 13.0 |
| self_oracle_contour_target - shifted_self_target_-1_mora | 17 | 27.4118 | 18.0 | 27.0 | 37.0 |
| self_oracle_contour_target - shifted_self_target_+1_mora | 17 | 28.4706 | 17.0 | 29.0 | 37.0 |
| self_oracle_contour_target - flat_user_contour_vs_self_target | 17 | 52.9412 | 46.0 | 53.0 | 59.0 |
| self_oracle_contour_target - shuffled_user_contour_vs_self_target | 17 | 44.2941 | 34.0 | 43.0 | 59.0 |

## Interpretation

- self-oracle scores are high, so score_prosody can reward a matching mora-level F0 contour.
- OpenJTalk/tool-generated pitch targets are a major limiter versus self-reference.
- lab/reference mora timing improves current OpenJTalk-target scoring, so equal alignment is a material risk.
- one-mora shifts heavily reduce score, so pitch scoring is alignment-sensitive.
- flat pitch counterfactuals score lower than self-oracle.
- shuffled/randomized pitch counterfactuals score lower than self-oracle.
- low F0 coverage is not the main explanation for these rows.
- This report is diagnostic only; it does not change runtime scoring, UI, aggregate, or content gates.

## Calibration Readiness

Not ready for a score calibration transform yet. The audit supports target/reference and alignment work first, because self-reference behaves well while the current OpenJTalk target remains much lower.
