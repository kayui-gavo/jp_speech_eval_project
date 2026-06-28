# Python package API

The stable import name is `jp_speech_eval`. Version `1.6.0` exposes a small
integration API while keeping research/debug metrics separate from learner UI.

## Install

```bash
python -m pip install jp_speech_eval-1.6.0-py3-none-any.whl
# Add local ASR support when needed:
python -m pip install "jp_speech_eval-1.6.0-py3-none-any.whl[asr]"
```

Runtime JSON defaults are bundled in the wheel. Fixed-reference audio caches
remain external assets and must be supplied through `SpeechEvalConfig` or each
request.

## Main import

```python
from jp_speech_eval import (
    EvaluationRequest,
    EvaluationResponse,
    SpeechEvalConfig,
    SpeechEvaluationClient,
    build_asr_confirmation,
    evaluate_speech,
)
```

## ASR-confirmed weak-reference evaluation

```python
client = SpeechEvaluationClient()
response = client.evaluate(
    EvaluationRequest(
        audio_path="user.wav",
        mode="asr_confirmed_weak_reference",
        user_confirmed_text="今日は大学で勉強しました",
    )
)
```

The application must let the user confirm or edit ASR text before evaluation.
Weak-reference results are native-likeness/practice guidance, not strict pitch
accent correctness.

## Fixed-reference evaluation

```python
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

## Public response contract

Product code must render `response.user_facing`, especially:

```python
{
    "display_score": 82,
    "dimension_scores": {
        "pronunciation": 84,
        "rhythm": 78,
        "fluency": 86,
        "pitch": 80,
    },
    "dimension_confidence": {
        "pronunciation": "medium",
        "rhythm": "medium",
        "fluency": "high",
        "pitch": "medium",
    },
    "summary_text": "...",
    "primary_suggestion_text": "...",
    "mode_notice": "...",
}
```

Scores may be `None` when content or recording evidence is invalid. Never fall
back from a missing user-facing score to `raw_result.total_score` or raw
`prosody_score`.

`response.raw_result` is intentionally retained for logs, diagnostics and
research inspection. It is not the learner-facing contract.

## Mode boundary

| Mode | Intended use | Required external input |
|---|---|---|
| `asr_confirmed_weak_reference` | arbitrary Japanese practice | user-confirmed Japanese text |
| `reference` | known-sentence practice | matching reference cache and target text |
| `kanade_asr_voice_reference` | experimental personalized playback | source bundle, Kanade worker environment |

Kanade audio is playback/reference experience only and is excluded from
pronunciation correctness.
