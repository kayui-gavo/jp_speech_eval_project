# 日语四维练习评分模块：接入交接说明

## 交付版本

- Python package：`jp-speech-eval 1.6.0`
- Python：`>=3.10,<3.13`，推荐 `3.11`
- 核心模式：`asr_confirmed_weak_reference`、`reference`
- 正式 UI 数据源：`EvaluationResponse.user_facing`

## 产品边界

模块面向日语练习反馈，不是考试系统。四维分数分别表示：

| 维度 | 当前含义 |
|---|---|
| 发音清晰度 | 录音质量、发音稳定性和可用对齐证据形成的练习 proxy |
| 节奏 / 特殊拍 | mora 时长、节奏稳定性及证据充分时的长音等提示 |
| 流畅度 | 语速、停顿和发声连续性 |
| 音高变化 | F0 覆盖、变化幅度、局部移动和平稳性的自然度参考 |

音高变化不是严格 pitch accent correctness；特殊拍也只在证据充分时给具体建议。

## 推荐架构

```text
录音 WAV
  -> ASR 候选文本
  -> 用户确认/修改日语文本
  -> SpeechEvaluationClient.evaluate(...)
  -> user_facing 四维分 + 一条主要建议
  -> raw_result 仅写开发日志
```

评价是 CPU 密集型同步调用。Web 服务中建议放到 worker thread / job queue，避免阻塞事件循环。

## 五分钟接入

```python
from jp_speech_eval import EvaluationRequest, SpeechEvaluationClient

client = SpeechEvaluationClient()

def evaluate_confirmed_japanese(audio_path: str, confirmed_text: str) -> dict:
    response = client.evaluate(
        EvaluationRequest(
            audio_path=audio_path,
            mode="asr_confirmed_weak_reference",
            user_confirmed_text=confirmed_text,
        )
    )
    if not response.ok:
        return {"ok": False, "error": response.error}

    user = response.user_facing
    return {
        "ok": True,
        "overall": user.get("display_score"),
        "dimensions": user.get("dimension_scores", {}),
        "confidence": user.get("dimension_confidence", {}),
        "summary": user.get("summary_text"),
        "action": user.get("primary_suggestion_text"),
        "notice": user.get("mode_notice"),
    }
```

## 前端映射

```text
pronunciation -> 发音清晰度
rhythm        -> 节奏 / 特殊拍
fluency       -> 流畅度
pitch         -> 音高变化
```

四维卡片只读取 `dimension_scores`。值为 `None` 时显示“无法判断”，不要显示 `0`，也不要回退到 raw score。

## 必须保留的安全规则

1. 英语或明显非日语内容不显示正式练习分。
2. 固定句内容不匹配时不显示正式练习分。
3. `display_score=None` 时，不能展示 raw `total_score`。
4. `dimension_scores.pitch=None` 时，不能展示 raw `prosody_score`。
5. `dimension_confidence` 必须和分数一起保留，不能把低证据包装成高可信结果。
6. `tone_score` 不属于核心四维。

## 固定句资源

固定句模式不能只传文字，还要传相同句子的 cache prefix：

```python
client = SpeechEvaluationClient(
    SpeechEvalConfig(cache_path="/absolute/path/to/cache_prefix")
)
```

例如 cache prefix 为 `/srv/cache/ramen_kudasai` 时，应存在相应的 `.json`、`.npz` 和 reference WAV 资源。交接包自带一套示例，不能把自动 TTS reference 宣称为母语者标准答案。

## 配置与扩展依赖

- wheel 已内置评分运行所需的默认 JSON 配置。
- `.[asr]`：本地 faster-whisper。
- `.[recording]`：本地麦克风录音。
- `.[ssl]`：研究用 SSL 特征，不是普通接入必需。
- `.[google-tts]`：可选 Google TTS。
- `.[full]`：安装全部可选依赖，体积较大。

Kanade 仍是实验性独立 worker，只在完整源码环境中使用，不属于轻量 wheel 的默认接入路径。

## 验收清单

- 正常日语能返回 `display_score` 和四个 `dimension_scores`。
- 英语/Latin-dominant 输入不会出现正式分数。
- `response.ok=False` 时能显示错误而不是伪造 0 分。
- `None` 不被前端转换成 0。
- 日志可保存 `raw_result`，普通用户看不到 raw/debug 指标。
- 固定句 cache 路径由部署侧明确配置。

## 已知限制

- 分数尚未经过大规模人工听评校准。
- wrong accent drop 的可靠识别仍不足。
- 自然对话、方言、情绪和复杂背景噪声覆盖仍有限。
- 特殊拍依赖边界质量，具体错误提示比总练习分更保守。
- 任意句仍依赖用户确认文本；ASR 本身不作为正确答案。

完整示例见 `examples/package_api_quickstart.py`，字段说明见 `docs/package_usage_zh.md`。
