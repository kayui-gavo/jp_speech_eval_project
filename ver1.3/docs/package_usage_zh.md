# jp_speech_eval Python 包使用说明

`jp_speech_eval` 是日语口语 / 发音练习评价模块。外部项目可以通过 Python API 调用。

## 安装

```bash
git clone https://github.com/kayui-gavo/jp_speech_eval_project.git
cd jp_speech_eval_project/ver1.3
python -m pip install -e .
```

说明：`ver1.3` 只是当前源码目录名，实际包名是 `jp_speech_eval`。

## 固定句朗读评价

这是目前最可靠的模式。

```python
from jp_speech_eval import SpeechEvaluationClient, SpeechEvalConfig, EvaluationRequest

client = SpeechEvaluationClient(
    SpeechEvalConfig(cache_path="cache/ramen_kudasai")
)

response = client.evaluate(
    EvaluationRequest(
        audio_path="data/ramen.wav",
        mode="reference",
        target_text="ラーメンをください",
    )
)

if response.ok:
    print(response.user_facing)
else:
    print(response.error)
```

## 推荐展示字段

产品 / UI 侧优先使用：

```python
response.user_facing
```

主要字段：

- `status`: `pass` / `practice_suggestion` / `retry` / `debug_only`
- `practice_score`: 练习参考分
- `summary_text`: 给用户看的总结
- `primary_suggestion_text`: 最多一个主要建议
- `mode_notice`: 当前模式限制说明

不建议直接展示：

```python
response.raw_result
```

`raw_result` 包含 raw score、F0、DTW、alignment、特殊拍 debug 等，主要用于开发和调试。

## 自由发话模式

自由发话需要先确认 ASR 文本。

```python
prompt = client.build_asr_confirmation("user.wav")
print(prompt.prompt)
```

让用户确认或修改文本后，再评价：

```python
response = client.evaluate(
    EvaluationRequest(
        audio_path="user.wav",
        mode="asr_confirmed_weak_reference",
        user_confirmed_text="ラーメンをください",
    )
)
```

注意：ASR 原始结果不能直接作为评分目标。

## ASR + Kanade

```python
response = client.evaluate(
    EvaluationRequest(
        audio_path="user.wav",
        mode="kanade_asr_voice_reference",
        user_confirmed_text="ラーメンをください",
    )
)
```

Kanade 只用于生成接近用户声线的参考音，不参与发音正确性评分。

## 最小示例

```bash
cd jp_speech_eval_project/ver1.3
../.venv/bin/python examples/package_api_quickstart.py
```

## 当前限制

- `practice_score` 是练习参考，不是严格发音能力评分。
- fixed-reference 是目前最可靠模式。
- ASR-generated reference 必须用户确认。
- Kanade 只是播放参考音，不是评分真值。
- `raw_result` 不建议直接展示给普通用户。
