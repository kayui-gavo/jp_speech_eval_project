# jp_speech_eval 1.6.0 快速使用说明

这是一个日语口语练习评分模块，可以输出四个参考分：

| 维度 | 主要参考内容 |
|---|---|
| 发音清晰度 | 录音是否清楚、发音是否稳定 |
| 节奏 / 特殊拍 | mora 时长、长音、促音、拨音等 |
| 流畅度 | 语速、停顿和发声连续性 |
| 音高变化 | 音高起伏是否自然、平稳 |

这些分数用于练习参考，不是考试成绩或教师判定。

## 1. 安装

推荐使用 Python 3.11。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install wheels/jp_speech_eval-1.6.0-py3-none-any.whl
```

如果需要模块自己进行 ASR 识别，改用：

```bash
python -m pip install "wheels/jp_speech_eval-1.6.0-py3-none-any.whl[asr]"
```

从 GitHub 源码安装时：

```bash
git clone https://github.com/kayui-gavo/jp_speech_eval_project.git
cd jp_speech_eval_project/ver1.3
python -m pip install -e ".[asr]"
```

## 2. 先运行自带示例

交接包内已经附带示例音频和参考数据：

```bash
python examples/package_api_quickstart.py
```

正常情况下会看到总参考分、四维分数、可信度和一条练习建议。

## 3. 自由说话评分

自由说话需要先识别文本，再让用户确认。这样可以避免把英语或 ASR 识别错误当成日语评分。

```python
from jp_speech_eval import EvaluationRequest, SpeechEvaluationClient

client = SpeechEvaluationClient()

# 第一步：识别文本
prompt = client.build_asr_confirmation("user.wav")
if not prompt.ok:
    raise RuntimeError(prompt.error)

# 在页面上显示这段文字，让用户确认或修改
confirmed_text = prompt.prompt["editable_text"]

# 第二步：根据确认后的日语文本评分
response = client.evaluate(
    EvaluationRequest(
        audio_path="user.wav",
        mode="asr_confirmed_weak_reference",
        user_confirmed_text=confirmed_text,
    )
)

if not response.ok:
    raise RuntimeError(response.error)

print(response.user_facing["dimension_scores"])
```

如果其他系统已经完成 ASR 和文本确认，可以直接从第二步开始。

## 4. 固定句朗读评分

固定句除了文字，还需要同一句话的参考 cache。交接包内附带“ラーメンをください”的示例。

```python
from jp_speech_eval import EvaluationRequest, SpeechEvalConfig, SpeechEvaluationClient

client = SpeechEvaluationClient(
    SpeechEvalConfig(cache_path="sample_assets/cache/ramen_kudasai")
)

response = client.evaluate(
    EvaluationRequest(
        audio_path="sample_assets/data/ramen.wav",
        mode="reference",
        target_text="ラーメンをください",
    )
)

if not response.ok:
    raise RuntimeError(response.error)

print(response.user_facing["dimension_scores"])
```

## 5. 页面应该读取哪些字段

正式页面只读取：

```python
result = response.user_facing

overall = result["display_score"]
scores = result["dimension_scores"]
confidence = result["dimension_confidence"]
summary = result["summary_text"]
suggestion = result["primary_suggestion_text"]
```

`dimension_scores` 的内容如下：

```python
{
    "pronunciation": 78,  # 发音清晰度
    "rhythm": 69,         # 节奏 / 特殊拍
    "fluency": 89,        # 流畅度
    "pitch": 78,          # 音高变化
}
```

`dimension_confidence` 表示每一项的判断依据是否充分。它不是分数，也不代表用户说得好或不好。

## 6. 接入时只要记住三条

1. 自由说话必须使用用户确认后的日语文本。
2. 页面只显示 `user_facing`，不要直接显示 `raw_result`。
3. 分数为 `None` 时显示“无法判断”，不要改成 `0`，也不要拿内部 raw score 补上。

更完整的字段边界、部署建议和已知限制见 `docs/integration_handoff_zh.md`。
