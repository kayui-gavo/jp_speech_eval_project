# Fixed-reference Scoring Audit Summary

## Experiment Overview

- manifest_path: `data/audit/fixed_reference_manifest_v0.csv`
- total_samples: 24
- expected_behavior_counts: native_should_score_high:10, clear_learner_should_score:6, content_mismatch_should_not_score:5, weak_reference_should_not_score:1, pitch_unverified_should_suppress_pitch:1, alignment_bad_should_not_score:1
- audio_type_counts: janon_learner_clear:6, jvs_native_clear:5, jvs_native_same_text_diff_speaker:5, wrong_japanese_sentence:4, partial_target:1, weak_reference:1, pitch_unverified_target:1, fallback_alignment_case:1
- reference_type_counts: tts_reference:13, human_reference:10, weak_reference:1

## Main Diagnostic Questions

- Are negative controls blocked? CHECK; affected_sample_id=neg_wrong_janon_mandarin_001
- Are weak-reference samples blocked? OK; affected_sample_id=-
- Are fallback alignment samples blocked from pitch feedback? OK; affected_sample_id=-
- Are native samples rejected too often? CHECK; affected_sample_id=jvs_native_jvs001_001, jvs_native_jvs001_002, jvs_native_jvs001_003, jvs_native_jvs001_004, jvs_native_jvs001_005, jvs_native_jvs002_001, jvs_native_jvs002_002, jvs_native_jvs002_003, jvs_native_jvs002_004, jvs_native_jvs002_005
- Are native scores too low? OK; affected_sample_id=-
- Are bad learner samples still suspiciously high? OK; affected_sample_id=-
- Is special mora warning shown for native? OK; affected_sample_id=-
- Is display cap frequently applied to native? OK; affected_sample_id=-

## Group Summary

| audio_type | reference_type | expected | n | score_available | display mean/median/p10/p90 | pron mean/median | raw prosody mean/median | pitch allowed | pitch leakage | cap rate | avg cap reduction | fallback | content fail | pron evidence fail | special shown | special suppressed |
|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fallback_alignment_case | tts_reference | alignment_bad_should_not_score | 1 | 0.0 | /// | / | 50.0/50.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| janon_learner_clear | tts_reference | clear_learner_should_score | 6 | 1.0 | 58.67/58.5/51.0/62.0 | 53.67/53.5 | 72.67/76.5 | 0.0 | 0.0 | 1.0 | 8.33 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| jvs_native_clear | human_reference | native_should_score_high | 5 | 0.0 | /// | / | 0.0/0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| jvs_native_same_text_diff_speaker | human_reference | native_should_score_high | 5 | 0.0 | /// | / | 0.0/0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| partial_target | tts_reference | content_mismatch_should_not_score | 1 | 0.0 | /// | / | 50.0/50.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| pitch_unverified_target | tts_reference | pitch_unverified_should_suppress_pitch | 1 | 0.0 | /// | / | 50.0/50.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| weak_reference | weak_reference | weak_reference_should_not_score | 1 | 0.0 | /// | / | 50.0/50.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| wrong_japanese_sentence | tts_reference | content_mismatch_should_not_score | 4 | 0.25 | 61.0/61.0/61.0/61.0 | 56.0/56.0 | 32.0/25.0 | 0.0 | 0.0 | 0.5 | 2.0 | 0.25 | 0.5 | 0.75 | 0.0 | 1.0 |

## Failures and Warnings

- WARN_native_score_available_rate: group=('jvs_native_clear', 'human_reference', 'native_should_score_high'), rate=0.0
- WARN_native_score_available_rate: group=('jvs_native_same_text_diff_speaker', 'human_reference', 'native_should_score_high'), rate=0.0
- FAIL_negative_user_facing_score: group=('wrong_japanese_sentence', 'tts_reference', 'content_mismatch_should_not_score'), sample_id=neg_wrong_janon_mandarin_001, score_available=True, display=61

## Pitch Text Leakage

- pitch_text_leakage_count: 0

## Display Cap Reductions

- display_cap_applied_rate: 0.5
- average_cap_reduction: 2.42
- reduction_gte_10: janon_mandarin_clear_005(10.0)

## Suspicious Sample List

- negative_controls_with_display_score: neg_wrong_janon_mandarin_001
- negative_controls_with_pitch_feedback_allowed: -
- weak_reference_with_display_score: -
- fallback_with_pitch_feedback_allowed: -
- bad_learner_display_score_gte_80: -
- native_display_score_lt_75: -
- native_rejected_by_gate: jvs_native_jvs001_001, jvs_native_jvs001_002, jvs_native_jvs001_003, jvs_native_jvs001_004, jvs_native_jvs001_005, jvs_native_jvs002_001, jvs_native_jvs002_002, jvs_native_jvs002_003, jvs_native_jvs002_004, jvs_native_jvs002_005
- native_with_special_mora_user_facing_warning: -
- native_with_large_display_cap_reduction: -

## Decision Hints

- If negative controls still show scores or pitch feedback, fix gates or message leakage first.
- If native samples are often rejected, check reference audio, alignment, VAD, and sampling rate before changing score mapping.
- If native score availability is acceptable but display scores are low, inspect score mapping and display cap behavior next.
- If bad learner samples remain 80+, calibrate pronunciation_score in the next round after gate behavior is confirmed.
- This audit does not auto-tune thresholds.

## Recommended Next Action

- Fix reliability/content gates or user-message leakage first. Do not tune pronunciation_score yet. affected_sample_id=neg_wrong_janon_mandarin_001
- Native score availability is low. Check reference audio, VAD, sample rate, kana/mora parsing, and alignment thresholds before changing score mapping. score_available_rate=0.0; affected_sample_id=jvs_native_jvs001_001, jvs_native_jvs001_002, jvs_native_jvs001_003, jvs_native_jvs001_004, jvs_native_jvs001_005, jvs_native_jvs002_001, jvs_native_jvs002_002, jvs_native_jvs002_003, jvs_native_jvs002_004, jvs_native_jvs002_005

## Warning Code Counts

- score_policy_warnings: content_match_failed_no_pronunciation_score:12, pronunciation_under_60_display_cap:5, alignment_fallback_cap:5, alignment_fallback_no_display_score:5, pronunciation_under_70_display_cap:1, pronunciation_under_50_display_cap:1
- suppressed_reasons: shadow_mode_user_facing_disabled:17, content_mismatch:12, no_correction_needed:7, fallback_alignment:5, low_f0_coverage:5
- user_message_type: content_mismatch:12, alignment_limited:5
- special_mora_evidence_level: high:18, medium:4, low:2
- special_mora_suppression_reason: shadow_mode_user_facing_disabled:17, no_correction_needed:7
- rhythm_timing_penalty_reason: -
