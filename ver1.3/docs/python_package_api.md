# Python package API

This package exposes a small Python interface for integrating the speech evaluation pipeline into another project.

## Install locally

From the repository root, enter the current package source directory and install it editable:

```bash
cd ver1.3
python -m pip install -e .
```

The directory name `ver1.3` is an internal project folder kept for compatibility. External callers should depend on the package/import name `jp_speech_eval`, not on the directory name.

## Main import

```python
from jp_speech_eval import (
    EvaluationRequest,
    SpeechEvalConfig,
    SpeechEvaluationClient,
    build_asr_confirmation,
    evaluate_speech,
)
```

## Fixed-reference evaluation

Use this for known target sentences. This is the most reliable product path.

```python
from jp_speech_eval import EvaluationRequest, SpeechEvalConfig, SpeechEvaluationClient

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
    print(response.user_facing["summary_text"])
    print(response.user_facing["practice_score"])
```

Product UI should use `response.user_facing`.

`response.raw_result` is for developer/debug use only.

## ASR-confirmed weak-reference flow

ASR modes must be two-step.

```python
prompt = client.build_asr_confirmation("user_free_speech.wav")

# Show prompt.prompt["editable_text"] to the user.
# User confirms or edits the text in the app.

response = client.evaluate(
    EvaluationRequest(
        audio_path="user_free_speech.wav",
        mode="asr_confirmed_weak_reference",
        user_confirmed_text="ラーメンをください",
    )
)
```

Rules:

- ASR raw text must not become a scoring reference directly.
- `user_confirmed_text` is required for weak-reference scoring.
- Weak-reference feedback is practice support, not strict pronunciation correctness.

## ASR + Kanade flow

Kanade can be used as personalized reference playback, but it must not be treated as correctness scoring.

```python
response = client.evaluate(
    EvaluationRequest(
        audio_path="user_free_speech.wav",
        mode="kanade_asr_voice_reference",
        user_confirmed_text="ラーメンをください",
    )
)
```

Policy:

- Kanade is playback/reference experience only.
- Kanade output is excluded from pronunciation correctness.
- Similarity to Kanade audio should not be shown as a score.

## Public response contract

```python
{
    "ok": True,
    "mode": "reference",
    "user_facing": {
        "status": "pass",
        "practice_score": {"value": 93, "label": "良好"},
        "summary_text": "全体としてよくできています。",
        "primary_suggestion_text": None,
        "mode_notice": "..."
    },
    "raw_result": {...}
}
```

## Current limitations

- `practice_score` is demo guidance, not validated pronunciation ability.
- `total_score` and `prosody_score` are proxy/debug metrics.
- Fixed-reference is currently the most reliable path.
- ASR-generated reference requires user confirmation.
- Kanade is playback reference only, not scoring ground truth.
