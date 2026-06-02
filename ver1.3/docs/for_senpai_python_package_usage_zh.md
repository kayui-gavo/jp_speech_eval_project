# jp_speech_eval 语音评价包使用说明

这份文档给项目集成方/前辈使用，目标是让外部 Python pipeline 可以直接调用当前日语口语/发音评价模块。

## 1. 当前定位

`jp_speech_eval` 是一个轻量日语口语/发音练习评价包。

目前最可靠的主线是：

- 固定目标句朗读评价：`mode="reference"`

辅助/产品体验模式是：

- 自由发话后确认文本，再做弱参考反馈：`mode="asr_confirmed_weak_reference"`
- ASR + Kanade 声线参考音播放：`mode="kanade_asr_voice_reference"`

注意：

- `practice_score` 是练习参考分，不是严格科学发音能力评分。
- `raw_result` 是 debug / 研究用，不建议直接展示给 C 端用户。
- Kanade 只用于生成/播放接近用户声线的参考音，不参与发音正确性评分。
- ASR 识别结果必须让用户确认或修改后，才能进入 weak-reference 评价。

## 2. 安装

```bash
git clone https://github.com/kayui-gavo/jp_speech_eval_project.git
cd jp_speech_eval_project/ver1.3
python -m pip install -e .
```

说明：

- 当前源码目录仍叫 `ver1.3`，这是历史兼容目录名。
- 真正的 Python 包名 / import 名是 `jp_speech_eval`。
- 外部代码应该依赖 `jp_speech_eval`，不要依赖目录名。

## 3. 最小调用：固定句朗读评价

固定句模式适合目标句已知的朗读练习，是当前最可靠的模式。

```python
from jp_speech_eval import (
    SpeechEvaluationClient,
    SpeechEvalConfig,
    EvaluationRequest,
)

client = SpeechEvaluationClient(
    SpeechEvalConfig(
        cache_path="cache/ramen_kudasai",
        tts_backend="pyopenjtalk",
    )
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

## 4. 推荐给产品/UI 使用的字段

产品侧建议优先读取：

```python
response.user_facing
```

典型结构：

```python
{
    "status": "pass",
    "practice_score": {
        "value": 94,
        "label": "良好",
        "explanation": "このスコアは...練習用の目安です。"
    },
    "confidence": "high",
    "summary_text": "全体としてよくできています。",
    "primary_suggestion_text": None,
    "suggestion_type": "none",
    "mode_notice": "fixed-reference mode: ...",
    "display_total_score": False
}
```

字段含义：

- `status`
  - `pass`: 本次练习整体可接受
  - `practice_suggestion`: 可以继续，但有一个练习建议
  - `retry`: 录音质量/内容匹配等不适合判断，建议重录
  - `debug_only`: 只作为参考，不做严格判定

- `practice_score`
  - 给用户看的练习参考分
  - 不是严格发音能力分
  - 如果模式不适合打分，`value` 可能是 `None`

- `summary_text`
  - 给用户看的短总结

- `primary_suggestion_text`
  - 最多一个主要练习建议

- `mode_notice`
  - 当前模式的限制说明
  - weak-reference / Kanade 模式尤其需要展示

## 5. 不建议直接展示给普通用户的字段

```python
response.raw_result
```

这里面包含：

- raw `total_score`
- raw `prosody_score`
- alignment 细节
- F0 覆盖率
- DTW / mora 证据
- special mora shadow decisions
- reliability gate debug

这些适合开发者、教师、研究调试使用，不建议直接展示给普通 C 端用户。

## 6. 自由发话：ASR 确认后弱参考评价

自由发话必须两步走。

### Step 1: 生成 ASR 候选，让用户确认

```python
prompt = client.build_asr_confirmation("user_free_speech.wav")

if prompt.ok:
    print(prompt.prompt)
else:
    print(prompt.error)
```

典型返回：

```python
{
    "mode": "asr_confirm",
    "session_id": "...",
    "asr_candidates": [
        {"id": 1, "text": "ラーメンをください", "confidence": 0.82}
    ],
    "editable_text": "ラーメンをください",
    "message": "猜你想说的是哪一句？如果不对，请手动修改。",
    "asr_raw": {...}
}
```

UI 应该让用户选择候选或手动修改 `editable_text`。

### Step 2: 使用用户确认文本进行评价

```python
response = client.evaluate(
    EvaluationRequest(
        audio_path="user_free_speech.wav",
        mode="asr_confirmed_weak_reference",
        user_confirmed_text="ラーメンをください",
    )
)
```

重要规则：

- ASR raw text 不能直接当评分目标。
- 必须使用 `user_confirmed_text`。
- 该模式是 weak-reference，只能输出练习参考反馈。
- 不应声称严格 pronunciation correctness。

## 7. ASR + Kanade 声线参考音

Kanade 模式用于产品体验：让用户听到更接近自己声线的参考音。

```python
response = client.evaluate(
    EvaluationRequest(
        audio_path="user_free_speech.wav",
        mode="kanade_asr_voice_reference",
        user_confirmed_text="ラーメンをください",
    )
)
```

注意：

- Kanade 只用于 playback/reference experience。
- Kanade audio 不参与 pronunciation correctness scoring。
- 不要计算或展示 `similarity_to_kanade`。
- 如果 UI 展示该模式，必须提示：声线相似度不参与评分。

## 8. 常见配置

```python
config = SpeechEvalConfig(
    cache_path="cache/ramen_kudasai",
    sample_rate=16000,
    tts_backend="pyopenjtalk",
    special_mora_threshold_profile="default_safe",
    enable_runtime_special_mora_shadow=True,
    enable_user_facing_calibrated_special_mora=False,
)
client = SpeechEvaluationClient(config)
```

说明：

- `cache_path`: 固定句 reference cache 路径
- `tts_backend`: TTS 后端，目前可用 `pyopenjtalk` 等
- `enable_user_facing_calibrated_special_mora`: 默认应保持 `False`
- `enable_runtime_special_mora_shadow`: 可以保留 `True`，用于 debug 记录

## 9. 最小完整示例

仓库里提供了一个可运行示例：

```bash
cd jp_speech_eval_project/ver1.3
../.venv/bin/python examples/package_api_quickstart.py
```

示例输出类似：

```text
status: pass
practice_score: {'value': 94, 'label': '良好', ...}
summary: 全体としてよくできています。
suggestion: None
```

## 10. 推荐集成方式

如果前辈的项目里有一个后端 pipeline，可以这样封装：

```python
from jp_speech_eval import EvaluationRequest, SpeechEvaluationClient

client = SpeechEvaluationClient()

def evaluate_japanese_speech(audio_path: str, target_text: str | None = None):
    if target_text:
        request = EvaluationRequest(
            audio_path=audio_path,
            mode="reference",
            target_text=target_text,
        )
    else:
        raise ValueError("Free speech requires ASR confirmation step first.")

    response = client.evaluate(request)
    return response.to_dict()
```

产品 UI 建议只使用：

```python
result["user_facing"]
```

开发者调试时再看：

```python
result["raw_result"]
```

## 11. 当前限制

- `practice_score` 是 demo guidance，不是 validated pronunciation ability。
- `total_score` / `prosody_score` 是 proxy/debug metrics。
- fixed-reference 是目前最可靠模式。
- ASR-generated reference 必须用户确认。
- Kanade 是 playback reference only，不是 scoring ground truth。
- 特殊拍 feedback 默认保守，普通用户端不开强纠错。
- pitch accent feedback 需要 verified target 和可靠 F0。
- JANON trend 不是 ground truth。
- 目前 human validation 仍然不足。

## 12. 相关文件

- `docs/python_package_api.md`
- `docs/pipeline_api.md`
- `docs/api_user_facing_contract.md`
- `examples/package_api_quickstart.py`
- `reports/demo_known_limitations.md`
- `reports/demo_readiness_report.md`
