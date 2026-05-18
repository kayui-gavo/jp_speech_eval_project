# Demo 使用指南

## 目标

该 demo 用于展示一个研究型日语语音评价原型。系统当前重点是：

- 按日语的 `mora` / `accent phrase` 结构处理韵律
- 区分发音、流利度和表达风格
- 在证据不足时通过 reliability gate 降级
- 明确区分 pseudo-reference、ASR transcript 与 ground truth

## 演示前准备

在项目根目录启动：

```bash
cd ver1.3
python scripts/debug_ui.py \
  --mode asr_pseudo_reference \
  --tts-backend aivis_http \
  --tts-url http://127.0.0.1:10101 \
  --tts-speaker 888753760
```

然后打开：

```text
http://127.0.0.1:8765/
```

如果只是快速展示固定句模式，也可以直接用默认启动命令。

## 推荐演示顺序

### 1. 查看日语分析单位

打开固定句：

```text
ラーメンをください
```

- 系统按 mora 分析，而不是按英语式音节分析
- 长音 `ー`、撥音 `ン` 这类特殊拍会进入时长和节奏分析
- 目标韵律不再只是孤立词拼接，而会考虑 accent phrase

### 2. 查看 realtime 轻反馈

录音时可观察：

- 是否已经开始说话
- 音量是否太弱
- 是否有明显停顿
- F0 是否在运动

realtime 层只提供低风险反馈；完整判断在句末评估阶段完成。

### 3. 查看句末输出和 reliability gate

录音完成后，可查看：

- pronunciation / prosody / fluency / tone proxy
- reliability
- warnings
- mora evidence

- 录音质量不好时，系统会先降可靠性
- F0 coverage 不足时，不会假装自己能稳定判断音调
- equal-time fallback 时，系统会抑制强反馈

### 4. 查看 ASR 相关模式的限制

- ASR 只能作为 transcript hypothesis
- `asr_pseudo_reference` 模式的 reference 也是基于 ASR 假设生成
- 如果 transcript 本身错了，后面的语义和伪参考都会被带歪
- 所以系统明确把它叫 pseudo-reference，并保留 reliability / warning

### 5. 查看后续研究方向

1. 使用 JANON-SPEECH 与后续数据做阈值校准
2. 补充更可靠的 accent target 与 segmental evidence
3. 用教师标注验证规则特征与人类评分之间的相关性

## 推荐样例

### 样例 A：标准固定句

```text
ラーメンをください
```

可用于：

- 展示 mora
- 展示 contour
- 展示固定句评分主流程

### 样例 B：含特殊拍的句子

```text
切符を買って待っています
```

可用于：

- 展示促音、长音、鼻音这类时间结构
- 说明为什么只看“整体像不像”不够

### 样例 C：自由发话模式

```text
私は東京大学の1年生です
```

可用于：

- 展示 ASR transcript 不是 ground truth
- 说明为什么自由说话模式必须更保守

## 相关文档

- [`research_overview_zh.md`](research_overview_zh.md)
- [`../ver1.3/docs/theory_basis.md`](../ver1.3/docs/theory_basis.md)
- [`../ver1.3/docs/evaluation_infrastructure.md`](../ver1.3/docs/evaluation_infrastructure.md)

## 相关实现

- [`../ver1.3/src/jp_speech_eval/text_frontend.py`](../ver1.3/src/jp_speech_eval/text_frontend.py)
- [`../ver1.3/src/jp_speech_eval/mora_evidence.py`](../ver1.3/src/jp_speech_eval/mora_evidence.py)
- [`../ver1.3/src/jp_speech_eval/unified_result.py`](../ver1.3/src/jp_speech_eval/unified_result.py)
