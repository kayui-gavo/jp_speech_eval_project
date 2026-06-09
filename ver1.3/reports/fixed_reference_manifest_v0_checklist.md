# Fixed-reference Manifest v0 检查清单

正式 v0 审计请先复制模板，再填入真实音频路径。

```bash
cp data/audit/fixed_reference_manifest_template.csv data/audit/fixed_reference_manifest_v0.csv
```

模板里的 placeholder 行只作格式示例。正式审计前请删除它们，或替换成真实数据。

## A. Negative Controls

建议数量：10-20 条

`audio_type`:

- `wrong_japanese_sentence`
- `english_or_chinese_speech`
- `random_speech`
- `partial_target`
- `noise_or_silence`

`expected_behavior`:

- `content_mismatch_should_not_score`
- `recording_bad_should_not_score`

期望：

- 不显示 `display_score`
- 不显示 pitch feedback
- user-facing message 里不能漏出正向 pitch/prosody 表扬

## B. JVS Native

建议数量：20-50 条

`audio_type`:

- `jvs_native_clear`
- `jvs_native_same_text_diff_speaker`

`expected_behavior`:

- `native_should_score_high`

期望：

- `score_available_rate` 要高
- `display_score` 的 median / p10 不应过低
- special mora warning 不应频繁误伤母语者
- display cap reduction 不应大面积过大

## C. JANON / Learner

建议数量：20-50 条

`audio_type`:

- `janon_learner_clear`
- `janon_learner_bad`

`expected_behavior`:

- `clear_learner_should_score`
- `bad_learner_should_not_score_high`

期望：

- clear learner 不应被过度拒评
- bad learner 不应大量出现 `display_score >= 80`
- 如果 bad learner 分数仍偏高，下一轮再看 `pronunciation_score` mapping

## D. System Edge Cases

建议数量：5-10 条

`audio_type`:

- `weak_reference`
- `fallback_alignment_case`
- `pitch_unverified_target`

`expected_behavior`:

- `weak_reference_should_not_score`
- `alignment_bad_should_not_score`
- `pitch_unverified_should_suppress_pitch`

期望：

- weak-reference 不显示 `display_score` 和 pitch correctness
- fallback alignment 不显示 pitch feedback
- unverified pitch target 不显示 pitch feedback

## 运行前检查

```bash
../.venv/bin/python scripts/build_fixed_reference_audit_manifest.py \
  --input data/audit/fixed_reference_manifest_v0.csv \
  --dry-run
```

厳密チェック:

```bash
../.venv/bin/python scripts/build_fixed_reference_audit_manifest.py \
  --input data/audit/fixed_reference_manifest_v0.csv \
  --dry-run \
  --strict
```

`--strict` 会在 `audio_path` 不存在或 `expected_behavior` 未知时返回非零退出码。
