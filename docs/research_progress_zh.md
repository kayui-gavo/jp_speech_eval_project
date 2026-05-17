# 日语语音评价系统研究进度概览

## 当前阶段

这个项目现在更接近一个“有证据门控的研究型原型”，而不是已经训练完成的自动评分器。

目前最重要的进展，不是把指标堆多，而是把系统从“看到一点声学现象就急着下结论”推进到：

- 尽量用符合日语音系的分析单位
- 把不同类型的证据分开
- 在证据不足时主动降级，而不是输出看似精确的强判断

## 已经落地的关键改进

### 1. 从孤立词式思维转向句子级日语韵律

系统已经不再把每个词都当成彼此独立的音调岛。

- 文本前端以 `mora` 为基本单位，而不是英语式音节
- `ラーメン` 这类词会按 `ラ・ー・メ・ン` 分析
- 目标高低型按 `accent phrase` 生成
- 前端使用 OpenJTalk 的连接信息，尽量把助词、助动词和前项合成同一重音短语

这一步对应的价值是：日语连读后的音调变化终于进入了系统，而不只是“查一个词典重音然后硬拼起来”。

### 2. 从绝对音高转向结构比较

当前系统更看重：

- speaker-normalized log-F0 轮廓
- 相邻 mora 之间的上升 / 下降 / 平稳关系
- 重音核下降和短语起始上升
- 句末语调走势

而不是直接拿用户的绝对 Hz 去贴模板。

这更接近 Minematsu / OJAD 一系研究里“比较结构而不是比较声线本身”的思路，也更适合不同性别、年龄和音域的说话人。

### 3. 把“发音”“流利度”“语气”拆开

系统现在明确区分：

- `pronunciation proxy`
  - 主要看 mora 节奏、长音 / 促音 / 撥音等时间结构
- `prosody`
  - 主要看 F0 轮廓、相邻 mora 运动、句末语调
- `fluency`
  - 主要看语速和停顿
- `tone / emotion proxy`
  - 主要看表达风格，不再冒充发音正确性

这件事很重要，因为“语气平”“停顿多”“音高不活跃”并不等于“假名读错了”。

### 4. 新增多层 reliability gate，减少过度自信

已经加入的保护包括：

- endpointing / VAD，避免录音前后静音污染语速判断
- 录音质量估计：噪声、削波、动态范围
- mora-level evidence gate：每个 mora 是否真的有足够证据可判断
- F0 coverage gate：音高证据太稀疏时不做强结论
- equal-time segmentation fallback 检测
- 低可靠性时压低分数上限并改写反馈口径

这部分是目前最值得展示的工程进步之一。它没有让系统“更会装懂”，而是让系统更知道什么时候不该装懂。

### 5. 实时反馈与句末判断分层

当前设计已经分成两层：

- realtime
  - 只看开始说话、音量、停顿、F0 movement、录音状态
- sentence-final
  - 再做 mora 对齐、韵律比较、流利度和反馈汇总

这样既保留了交互速度，也避免把实时层的粗糙信号误当成最终评分。

### 6. 开始具备可实验、可复盘的基础设施

已经补上的研究基础设施包括：

- `UnifiedEvaluationResult`
- JSONL 逐条日志
- CSV 特征导出
- JANON-SPEECH 批量分析脚本
- 多种 evaluation mode 的统一比较
- 可替换 pseudo-reference TTS backend

这意味着下一阶段可以真正做：

- 阈值校准
- 特征消融
- 教师标注拟合
- 小模型回归 / 分类

而不只是靠主观听感改规则。

## 现在可以诚实展示的能力

### 已经可以演示

- 固定句子的 mora / accent phrase 分析
- 实时轻反馈
- 句末 pronunciation / prosody / fluency / tone proxy 输出
- reliability 降级逻辑
- ASR transcript 只是 pseudo-reference，不是 ground truth
- 参考音频 backend 可替换，且来源会被记录
- JANON-SPEECH 上的批量特征分析路径

### 还不能声称已经解决

- 还没有教师标注校准，所以现在的分数不是教学意义上的金标准
- 还没有 CTC / GOP / 专门的日语音素级模型，不能稳定做假名级纠错
- ASR 会把“听错的内容”带进后续链路，不能把 transcript 当真值
- AivisSpeech / OpenJTalk / Kanade 都只能提供 pseudo-reference，不能当母语标准答案
- 当前系统还不能证明优于 SOTA，也没有这个实验结论

## 最值得继续投入的方向

1. 用 native + learner + teacher label 做校准，而不是继续手调阈值
2. 引入更可靠的句子级 accent target 来源，减少 TTS 自身错误对评分的污染
3. 在轻量前提下增加 segmental evidence，例如 CTC posterior / GOP-like 特征
4. 继续把“声学证据不足”和“真的说错了”分开建模
5. 用 JANON-SPEECH 和之后补充的数据集做消融，验证哪些改进真的提高相关性

## 建议队友先看的文件

- [`ver1.3/docs/theory_basis.md`](../ver1.3/docs/theory_basis.md)
- [`ver1.3/docs/evaluation_infrastructure.md`](../ver1.3/docs/evaluation_infrastructure.md)
- [`ver1.3/docs/non_reference_realtime_strategy.md`](../ver1.3/docs/non_reference_realtime_strategy.md)
- [`ver1.3/src/jp_speech_eval/text_frontend.py`](../ver1.3/src/jp_speech_eval/text_frontend.py)
- [`ver1.3/src/jp_speech_eval/scoring.py`](../ver1.3/src/jp_speech_eval/scoring.py)
- [`ver1.3/src/jp_speech_eval/mora_evidence.py`](../ver1.3/src/jp_speech_eval/mora_evidence.py)
- [`ver1.3/src/jp_speech_eval/unified_result.py`](../ver1.3/src/jp_speech_eval/unified_result.py)

## 一句话版本

当前版本最大的进步，不是“分数更花哨”，而是系统开始按日语本身的韵律结构工作，并且在没有足够证据时学会闭嘴。
