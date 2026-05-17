# 给开发队友看的 5 分钟演示指南

## 演示目标

不要把它讲成“我已经做出了最终自动评分器”。

更准确的说法是：

> 我现在把系统推进到了一个更科学的研究原型：它开始按日语 mora / accent phrase 处理韵律，能区分发音、流利度和表达风格，并且在证据不足时会主动降级，而不是乱给强反馈。

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

## 现场顺序

### 1. 先讲“分析单位变了”

打开固定句：

```text
ラーメンをください
```

可以说：

- 旧思路容易把它当成普通音节串
- 现在系统按 mora 看
- 长音 `ー`、撥音 `ン` 这类特殊拍会进入时长和节奏分析
- 目标韵律不再只是孤立词拼接，而会考虑 accent phrase

### 2. 展示 realtime 只做轻反馈

录音时让队友看实时状态：

- 是否已经开始说话
- 音量是否太弱
- 是否有明显停顿
- F0 是否在运动

可以强调：

> 这里故意不做最终发音判定。实时层只做低风险反馈，句末再做完整分析。

### 3. 展示句末输出和 reliability gate

录完后重点看：

- pronunciation / prosody / fluency / tone proxy
- reliability
- warnings
- mora evidence

建议你现场指出：

- 录音质量不好时，系统会先降可靠性
- F0 coverage 不足时，不会假装自己能稳定判断音调
- equal-time fallback 时，系统会抑制强反馈

这一步最能体现你这段时间的研究不是“加指标”，而是在修正评价逻辑。

### 4. 展示 ASR 不是 ground truth

可以讲你已经遇到的真实例子：

```text
我说：私は東京大学の1年生です
ASR 可能听成：私は東京大学 終始家庭の1年生です
```

然后解释：

- ASR 只能作为 transcript hypothesis
- `asr_pseudo_reference` 模式的 reference 也是基于 ASR 假设生成
- 如果 transcript 本身错了，后面的语义和伪参考都会被带歪
- 所以系统明确把它叫 pseudo-reference，并保留 reliability / warning

这比假装“识别出来的就是用户真实说的”更靠谱。

### 5. 最后讲下一步

建议只讲三件事：

1. 用 JANON-SPEECH 和后续数据做校准，不再只靠手调阈值
2. 补更可靠的 accent target 与 segmental evidence
3. 用教师标注验证哪些规则真的提高了人类评分相关性

## 建议现场准备的三个样例

### 样例 A：标准固定句

```text
ラーメンをください
```

用途：

- 展示 mora
- 展示 contour
- 展示固定句评分主流程

### 样例 B：含特殊拍的句子

```text
切符を買って待っています
```

用途：

- 展示促音、长音、鼻音这类时间结构
- 说明为什么只看“整体像不像”不够

### 样例 C：ASR 容易误解的自由句

```text
私は東京大学の1年生です
```

用途：

- 展示 ASR transcript 不是 ground truth
- 说明为什么自由说话模式必须更保守

## 你可以直接说的三句话

1. “我现在最重视的不是多打几个分，而是让每个分数知道自己凭什么成立。”
2. “日语这里我已经从孤立词重音，推进到句子级 accent phrase 和 mora 结构了。”
3. “目前这些还是 proxy，不是已经被教师标注验证过的最终评分器；下一步重点就是做校准和消融。”

## 如果队友想往下看

先让他们看：

- [`research_progress_zh.md`](research_progress_zh.md)
- [`../ver1.3/docs/theory_basis.md`](../ver1.3/docs/theory_basis.md)
- [`../ver1.3/docs/evaluation_infrastructure.md`](../ver1.3/docs/evaluation_infrastructure.md)

如果他们想直接看实现，再去看：

- [`../ver1.3/src/jp_speech_eval/text_frontend.py`](../ver1.3/src/jp_speech_eval/text_frontend.py)
- [`../ver1.3/src/jp_speech_eval/mora_evidence.py`](../ver1.3/src/jp_speech_eval/mora_evidence.py)
- [`../ver1.3/src/jp_speech_eval/unified_result.py`](../ver1.3/src/jp_speech_eval/unified_result.py)
