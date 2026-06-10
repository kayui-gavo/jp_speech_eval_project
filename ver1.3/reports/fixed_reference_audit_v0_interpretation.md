# Fixed-reference Audit v0 Interpretation

本报告解释第一轮真实路径审计结果。它只用于定位系统风险，不代表最终性能评测。

## 1. Audit Overview

- manifest: `data/audit/fixed_reference_manifest_v0.csv`
- audit csv: `outputs/fixed_reference_scoring_audit.csv`
- summary: `reports/fixed_reference_scoring_audit_summary.md`
- total samples: 24

### Expected behavior counts

- `native_should_score_high`: 10
- `clear_learner_should_score`: 6
- `content_mismatch_should_not_score`: 5
- `weak_reference_should_not_score`: 1
- `pitch_unverified_should_suppress_pitch`: 1
- `alignment_bad_should_not_score`: 1

### Audio type counts

- `jvs_native_clear`: 5
- `jvs_native_same_text_diff_speaker`: 5
- `janon_learner_clear`: 6
- `wrong_japanese_sentence`: 4
- `partial_target`: 1
- `weak_reference`: 1
- `pitch_unverified_target`: 1
- `fallback_alignment_case`: 1

### Reference type counts

- `tts_reference`: 13
- `human_reference`: 10
- `weak_reference`: 1

## 2. Main Findings

- Negative controls are not fully blocked. One content-mismatch row still received a learner-visible score: `neg_wrong_janon_mandarin_001` with `display_score=61`.
- Weak-reference display suppression passed in this audit: no weak-reference row showed `display_score`.
- Fallback pitch guard passed: fallback/alignment-bad rows did not allow pitch feedback.
- Pitch feedback leakage was not observed: `pitch_feedback_allowed=False` for all 24 rows, and `pitch_text_leakage_count=0`.
- JVS/native rows were rejected too often: all 10 JVS native rows were rejected by `content_match`.
- JVS/native low-score analysis could not be evaluated because those rows had no displayed score.
- No bad-learner conclusion can be made in this v0 manifest, because JANON rows do not include teacher/native-listener badness labels.
- Special-mora user-facing warnings did not leak to JVS/native rows.
- Large display-cap reduction did not appear on JVS/native rows. One JANON learner row had `display_cap_reduction=10.0`, which is not the highest-priority issue.

## 3. Suspicious Samples

- negative controls with display_score: `neg_wrong_janon_mandarin_001`
- negative controls with pitch_feedback_allowed: none
- weak-reference with display_score: none
- fallback with pitch_feedback_allowed: none
- bad learner with display_score >= 80: not evaluated in this manifest
- native rejected by gate: `jvs_native_jvs001_001`, `jvs_native_jvs001_002`, `jvs_native_jvs001_003`, `jvs_native_jvs001_004`, `jvs_native_jvs001_005`, `jvs_native_jvs002_001`, `jvs_native_jvs002_002`, `jvs_native_jvs002_003`, `jvs_native_jvs002_004`, `jvs_native_jvs002_005`
- native display_score < 75: none, because native display scores were unavailable
- native with special-mora user-facing warning: none
- native with large display cap reduction: none

## 4. Recommended Next Action

Priority 1: fix or inspect the content-match / target-reference path before tuning pronunciation scores.

Reason:

- A negative control leaked a displayed score, so gate behavior is not yet strict enough.
- JVS native rows were all rejected by content match, even though their target_text came from JVS transcripts.
- This suggests the current fixed-reference audit path is still strongly tied to the existing ramen reference/cache behavior or ASR/content-match assumptions. It is not yet safe to interpret native low availability as a pronunciation-scoring problem.

Do not tune `pronunciation_score` or special-mora thresholds yet. The next engineering step should be:

1. Verify whether arbitrary `target_text` rows use a matching reference/cache throughout the pipeline.
2. Inspect ASR transcript and kana similarity for the rejected JVS native rows.
3. Confirm whether `human_reference` rows should bypass TTS cache assumptions or require a per-target cache.
4. Add a small manifest subset where target_text, reference features, and user audio are known to match.
5. Re-run this audit before changing any score mapping.

## 5. Coverage Gaps

- No true `noise_or_silence` sample was included.
- No real `random_speech` sample was included.
- No teacher-labeled `bad_learner_should_not_score_high` JANON subset was included.
- The weak-reference row is a fixed-reference audit label check, not a full ASR-confirmed weak-reference mode run.

These gaps should be filled before treating v0 as a stable benchmark.
