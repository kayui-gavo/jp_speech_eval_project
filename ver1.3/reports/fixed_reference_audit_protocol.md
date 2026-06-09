# Fixed-reference Audit Protocol

## 1. 目的

この audit は scoring threshold を調整する前に，fixed-reference 評価が安全に動いているかを確認するためのものです。

主に確認すること:

- native が拒評されすぎていないか
- native が低分になりすぎていないか
- bad learner がまだ 80 点以上の高い表示点を取りすぎていないか
- negative controls が表示点や pitch feedback を受け取っていないか
- special mora feedback が native を誤傷していないか
- display cap が native を過度に下げていないか

## 2. 最小サンプル構成

最初は小さく始めます。

- negative controls: 10-20 件
- JVS native: 20-50 件
- JANON learner: 20-50 件
- weak-reference / fallback / unverified pitch target: 5-10 件

## 3. manifest の書き方

テンプレート:

```text
data/audit/fixed_reference_manifest_template.csv
```

実験用にはコピーして使います。

```bash
cp data/audit/fixed_reference_manifest_template.csv data/audit/fixed_reference_manifest_v0.csv
```

主な `expected_behavior`:

- `native_should_score_high`: 母語話者または明瞭な native reference。拒評率と低分を確認する。
- `clear_learner_should_score`: 明瞭な学習者音声。score が出るかを確認する。
- `bad_learner_should_not_score_high`: 明らかに悪い発音や不自然な読み。80 点以上を suspicious として見る。
- `content_mismatch_should_not_score`: 目標文と違う文，英語，中国語，random speech など。表示点と pitch feedback を出してはいけない。
- `recording_bad_should_not_score`: ノイズ，無音，極端に小さい録音など。細かい発音判定を出してはいけない。
- `alignment_bad_should_not_score`: fallback alignment や不安定 alignment。pitch feedback を出してはいけない。
- `weak_reference_should_not_score`: ASR 確認後などの弱参照。厳密な表示点や pitch correctness を出してはいけない。
- `pitch_unverified_should_suppress_pitch`: OJAD/manual verified ではない target。pitch feedback を出してはいけない。

## 4. 実行方法

手動リストから manifest を作る場合:

```bash
cd ver1.3
../.venv/bin/python scripts/build_fixed_reference_audit_manifest.py \
  --input data/audit/manual_audio_list.csv \
  --out data/audit/fixed_reference_manifest_v0.csv
```

`manual_audio_list.csv` は最低限 `sample_id,audio_path,target_text,group,notes` を持つ小さな CSV/TSV でよいです。

```bash
cd ver1.3
../.venv/bin/python scripts/audit_fixed_reference_scoring.py \
  --manifest data/audit/fixed_reference_manifest_v0.csv \
  --out outputs/fixed_reference_scoring_audit.csv \
  --summary-out reports/fixed_reference_scoring_audit_summary.md
```

## 5. summary の見方

まず以下を見る:

- `Failures and Warnings`
- `Main Diagnostic Questions`
- `Suspicious Sample List`
- `Pitch Text Leakage`
- `Display Cap Reductions`
- `Warning Code Counts`

重点:

- negative controls に display_score や pitch feedback が出ていないか
- weak-reference に display_score や pitch feedback が出ていないか
- fallback alignment で pitch feedback が出ていないか
- `native_should_score_high` の score availability が低すぎないか
- `native_should_score_high` の display median / p10 が低すぎないか
- `bad_learner_should_not_score_high` に 80 点以上が多すぎないか
- native に special mora warning が出すぎていないか
- native に display cap reduction が大きく出すぎていないか

## 6. 判断ルール

- negative controls がまだ表示点や pitch feedback を受け取る場合: 先に gate / message leakage を直す。
- native が大量に拒評される場合: score mapping ではなく，reference audio / alignment / VAD / sampling rate を先に確認する。
- native の score availability は十分だが display が低い場合: 次ラウンドで score mapping または display cap を確認する。
- bad learner がまだ 80 点以上を多く取る場合: gate が通っていることを確認したあと，次ラウンドで pronunciation_score mapping を校正する。
- 本 protocol では threshold を自動調整しない。

## 7. 現時点の制限

- manifest は手動で作る。データセット全体を自動探索するものではない。
- JVS/JANON のラベル品質や target_text の一致は別途確認が必要。
- TTS reference は pseudo-reference であり，ground truth ではない。
- summary は実験判断を助けるためのもの。自動的に scoring threshold を変えない。
