# Release: Weak-reference Practice Demo

- release date: 2026-06-20
- stable feature commit: `7c0973b docs: add product demo smoke readiness audit`
- branch: `rollback/stable-packaged-demo-6-2`
- suggested tag: `stable/2026-06-20-weak-reference-practice-demo`

## Product Position

当前产品是一个“任意日语发话 practice scoring demo”。用户确认 ASR 候选文本后，系统可以提供谨慎的练习参考分和反馈。它不是考试级评分系统，也不提供教师级发音或高低重音正确性判定。

verified fixed-reference 路径继续保留，但它与任意句 weak-reference 主线分开。只有具备可靠人工/母语参考音频和可验证 provenance 时，才允许进行更严格的 reference-based pitch 比较。

## Problems Addressed

- JVS native 在 weak-reference 音高自然度路径中不再普遍低分，mean 为 `92.33`。
- random English 和 Latin-dominant confirmed text 进入 no-score，不再显示误导性的 practice overall。
- short / low-evidence 输入会 no-score 或 cap，避免短片段因停顿少而虚高。
- flat-pitch counterfactual mean 为 `9.0`，明显低于 native。
- shuffled/random-pitch counterfactual mean 为 `48.92`，低于 native。
- low-F0 输入的 pitch 会 unavailable，并对 weak overall 进行 evidence cap。
- UI 将核心结果明确表述为 practice / naturalness reference，不再把任意句音高分描述成严格 pitch-accent correctness。

## Four Practice Dimensions

1. **发音清晰度**

   基于音频清晰度、mora 对齐和发音稳定性的 proxy。它不是音素级 GOP，也不是教师听辨结论。

2. **节奏 / 特殊拍**

   基于长音、促音、拨音等 timing evidence 的练习参考。证据不足时不强提示特殊拍问题。

3. **流畅度**

   基于语速、停顿、有效发声时长和发声连续性的练习参考。

4. **音高变化**

   基于 F0 覆盖、音高变化幅度、局部变化和平稳性的 naturalness reference。它不是严格 pitch accent correctness，也不能稳定判定具体 accent drop 是否错误。

## Demo Smoke Set

- product smoke cases: `10/10` 符合预期。
- JVS native weak pitch naturalness mean: `92.33`。
- flat pitch mean: `9.0`。
- shuffled/random pitch mean: `48.92`。
- low-F0: `24/24` capped，pitch unavailable。
- random English / Latin-dominant / very short Japanese / content mismatch: no-score。
- tests: `180 passed, 4 warnings`。

smoke evidence 包含真实 JVS/JANON 音频审计、F0-only counterfactual 和 policy fixture。它不是完整人口校准，也没有把所有 case 冒充为真实端到端录音。

## Known Limitations

- 不是教师级评分。
- 不是考试系统。
- strict pitch accent correctness 尚未解决。
- wrong accent drop 与 native 的区分仍然偏弱，不能给出强错误断言。
- special mora 的真实 learner 音频覆盖仍不足；当前 smoke 中主要验证 policy gate。
- 过短输入需要独立的 isolated-word / short-utterance mode。
- 任意句模式依赖 ASR-confirmed text；ASR hallucinated Japanese 风险已降低，但没有跨语言、跨口音穷尽验证。
- 四个维度共享部分 timing 和 recording evidence，不应解释为彼此独立的能力测量。
- 当前结果不构成 calibration，也不能证明跨人群稳定性。

## Release Boundary

本 release 可以安全地作为练习型 demo 展示，但不得宣传为：

- 标准化考试分数；
- 教师级发音评价；
- 严格的日语高低重音正误判定；
- 经大规模学习者/母语者数据校准的能力分；
- 对所有语言、口音和短输入都稳定可靠的自动评价系统。

## Suggested Release Commands

Review and push the current branch:

```bash
git push origin rollback/stable-packaged-demo-6-2
```

After confirming the pushed commit, create and push the suggested annotated tag:

```bash
git tag -a stable/2026-06-20-weak-reference-practice-demo -m "stable: weak-reference Japanese practice demo"
git push origin stable/2026-06-20-weak-reference-practice-demo
```
