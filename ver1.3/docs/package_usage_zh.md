# jp_speech_eval 1.6.0 使用说明

`jp_speech_eval` 是面向日语口语练习的 Python 评价模块。当前稳定接口可以返回：

- 发音清晰度 `pronunciation`
- 节奏 / 特殊拍 `rhythm`
- 流畅度 `fluency`
- 音高变化 `pitch`

四项均为练习参考分，不是考试分数或教师级发音判定。

## 1. 安装

从交接压缩包安装：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install wheels/jp_speech_eval-1.6.0-py3-none-any.whl
```

自由发话需要本地 ASR 时，安装 ASR extra：

```bash
python -m pip install "wheels/jp_speech_eval-1.6.0-py3-none-any.whl[asr]"
```

从 GitHub 源码安装：

```bash
git clone https://github.com/kayui-gavo/jp_speech_eval_project.git
cd jp_speech_eval_project/ver1.3
python -m pip install -e ".[asr]"
```

支持 Python `3.10`、`3.11`、`3.12`，推荐 `3.11`。

## 2. 任意句练习：推荐接入方式

任意句必须先得到并确认日语文本。不要把未经用户确认的 ASR 结果直接当作评分目标。

```python
from jp_speech_eval import EvaluationRequest, SpeechEvaluationClient

client = SpeechEvaluationClient()

response = client.evaluate(
    EvaluationRequest(
        audio_path="user.wav",
        mode="asr_confirmed_weak_reference",
        user_confirmed_text="今日は大学で勉強しました",
    )
)

if not response.ok:
    raise RuntimeError(response.error)

result = response.user_facing
print(result["display_score"])
print(result["dimension_scores"])
print(result["dimension_confidence"])
print(result["summary_text"])
print(result["primary_suggestion_text"])
```

本地 ASR 确认流程：

```python
prompt = client.build_asr_confirmation("user.wav")
if not prompt.ok:
    raise RuntimeError(prompt.error)

# 在 UI 中显示 prompt.prompt["editable_text"]，让用户确认或修改。
confirmed_text = prompt.prompt["editable_text"]
```

## 3. 固定句朗读

固定句需要对应的 reference cache。交接包中的示例资源包含“ラーメンをください”。

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
```

## 4. UI 应读取哪些字段

只从 `response.user_facing` 渲染正式结果：

```python
user = response.user_facing

overall = user.get("display_score")
dimensions = user.get("dimension_scores", {})
confidence = user.get("dimension_confidence", {})

cards = {
    "发音清晰度": dimensions.get("pronunciation"),
    "节奏 / 特殊拍": dimensions.get("rhythm"),
    "流畅度": dimensions.get("fluency"),
    "音高变化": dimensions.get("pitch"),
}
```

主要字段：

| 字段 | 用途 |
|---|---|
| `display_score` | 本次练习总参考分；内容不匹配时可能为 `None` |
| `dimension_scores` | 四维练习分，值为 `0..100` 或 `None` |
| `dimension_confidence` | 四维证据等级，不等同于分数 |
| `status` | `pass`、`practice_suggestion`、`retry` 或 `debug_only` |
| `summary_text` | 一句总结 |
| `primary_suggestion_text` | 最多一个主要练习建议 |
| `mode_notice` | weak-reference / fixed-reference 的限制说明 |
| `suppressed_reasons` | 隐藏或降级的原因 |

严禁在 `display_score is None` 时回退显示 `raw_result.total_score`，也不要用 raw `prosody_score` 补上缺失的 `pitch`。

## 5. 什么时候不显示正式分数

以下情况会隐藏或降级正式结果：

- 英语、Latin-dominant 或明显非日语内容
- 固定句模式下内容明显不匹配
- 静音、严重噪声或录音无法形成有效语音证据

短句和部分低证据音频会尽量返回四维练习数字，同时通过 `dimension_confidence` 和文案标明证据限制。调用端不要把 `low` confidence 隐藏成“高可信评分”。

## 6. 调试数据边界

`response.raw_result` 与 `response.user_facing["debug"]` 仅用于开发、日志和人工检查。普通用户界面不要直接展示：

- raw total / pronunciation / prosody / fluency score
- F0、DTW、alignment cost、内部阈值
- `tone_score` 或 expression proxy

## 7. 最小验证

```bash
python examples/package_api_quickstart.py
```

预期输出包括总参考分和四维分。接入前建议再运行：

```bash
python -m pytest tests/test_public_api.py tests/test_package_and_ui_contract.py -q
```

更完整的交接说明见 `docs/integration_handoff_zh.md`。
