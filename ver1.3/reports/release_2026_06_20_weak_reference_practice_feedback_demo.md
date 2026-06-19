# Release: Weak-reference Practice Scoring and Feedback Demo

- release date: 2026-06-20
- stable commit: `7094448 feat: add evidence-based practice feedback`
- branch: `rollback/stable-packaged-demo-6-2`
- suggested tag: `stable/2026-06-20-weak-reference-practice-feedback-demo`

## Product Position

当前产品是一个面向任意日语发话的 **weak-reference practice scoring + evidence-based feedback demo**。

用户确认 ASR 候选文本后，系统可以显示谨慎的练习参考分，并根据已有证据选择一个最值得练习的问题和一个具体练习动作。它不是教师级评分系统，也不是考试系统。

verified fixed-reference 路径继续独立保留。只有存在可靠人工/母语参考音频及可信 provenance 时，才允许更严格的 reference-based pitch 比较。

## Problems Addressed

- JVS native 在 weak-reference prosody 路径中不再普遍低分，mean 为 `92.33`。
- random English / Latin-dominant confirmed text 进入 no-score。
- short / low-evidence input 会 no-score 或 cap，避免短片段虚高。
- flat-pitch counterfactual mean 为 `9.0`，明显低于 native。
- shuffled/random-pitch counterfactual mean 为 `48.92`，低于 native。
- low-F0 input 的 pitch unavailable，并对 weak overall 使用 evidence cap。
- UI 明确使用 practice / naturalness reference 文案，不把任意句评分包装成严格正确率。
- feedback 从旧的字符串挑选改为 evidence-based candidate。
- no-score 时只解释内容或证据问题，不输出发音、特殊拍或音高细节。
- too-short input 只提示证据不足，并要求说一个完整短句。
- weak pitch feedback 只描述偏平、不稳定或证据不足，不强判高低重音。
- special mora feedback 继续经过 alignment、confidence、type/profile 和 threshold gate。
- 普通用户每次最多看到一个核心问题和一个具体练习动作。

## Current Four Dimensions

1. **发音清晰度**

   录音清晰度、mora 对齐和发音稳定性的 proxy。它不是音素级 GOP，也不是教师听辨结论。

2. **节奏 / 特殊拍**

   长音、促音、拨音等 timing evidence 的练习参考。只有证据充分时才指出具体 mora；证据不足时不编造位置。

3. **流畅度**

   基于语速、停顿、有效发声时长和发声连续性的练习参考。

4. **音高变化**

   基于 F0 coverage、pitch range、local movement、smoothness、flat/random evidence 的 naturalness reference。它不是 strict pitch accent correctness。

## Current Feedback Types

- `content_match`：内容不一致、非日语或确认文本不可用。
- `insufficient_evidence / too_short`：录音过短、质量不足或对齐不稳定。
- `fluency`：语速偏快/偏慢、停顿过多或发声不连续。
- `special_mora`：高置信度的长音、拨音等时长问题，并在可靠时给出具体 mora 位置。
- `pitch_naturalness`：偏平、不稳定或 F0 证据不足；weak-reference 下始终带限制说明。
- `general_encouragement`：没有明确阻塞问题时给出简短练习鼓励，不声称音素完全正确。

每条主要 feedback 包含：维度、严重度、置信度、证据类型、可用时的位置、简短用户文案、具体 practice tip，以及必要的 caveat。raw/debug score 不进入普通用户 feedback。

## Smoke and Test Results

- demo smoke set: `10/10 PASS`。
- feedback actionability smoke set: `12/12 PASS`。
- feedback smoke overclaim risk: 全部为 `False`。
- full test suite: `192 passed, 4 warnings`。

这些 smoke 证据包含真实 JVS/JANON 音频审计、F0-only counterfactual 和 policy/component fixtures。它们不是大规模人口校准，也不表示所有 case 都是新采集的端到端真实录音。

## Still Not Claimed

- 教师级评分；
- 考试级或标准化能力评分；
- strict pitch accent correctness；
- reliable wrong accent drop detection；
- large-scale human calibration；
- proven learner improvement effect。

## Known Limitations

- wrong accent drop 与 native 的区分仍然偏弱。
- 真实 learner special-mora 音频覆盖仍有限，教学提示尚需人工听辨验证。
- short isolated-word mode 需要单独设计，不能直接套用任意句证据门槛。
- 任意句模式仍依赖 ASR-confirmed text；ASR 幻觉风险已降低，但未跨语言、跨口音穷尽验证。
- feedback 的教学有效性尚未通过中文母语学习者用户实验验证。
- 四个维度共享部分 timing 和 recording evidence，不应解释为相互独立的能力测量。

## Suggested Release Commands

Push the current branch after review:

```bash
git push origin rollback/stable-packaged-demo-6-2
```

Create the annotated tag on the stable feature commit and push it:

```bash
git tag -a stable/2026-06-20-weak-reference-practice-feedback-demo 7094448 \
  -m "stable: weak-reference Japanese practice scoring and feedback demo"
git push origin stable/2026-06-20-weak-reference-practice-feedback-demo
```
