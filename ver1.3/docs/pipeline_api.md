# Pipeline API

This package exposes a small SDK-style interface for external projects that
want to call the Japanese speech evaluation system without depending on debug
UI internals.

## Install

From this repository:

```bash
pip install -e ver1.3
```

Or, if you are already inside `ver1.3`:

```bash
pip install -e .
```

## Fixed-reference reading

Use this for the serious fixed sentence practice mode.

```python
from jp_speech_eval import EvaluationRequest, SpeechEvalConfig, SpeechEvaluationClient

client = SpeechEvaluationClient(
    SpeechEvalConfig(cache_path="cache/ramen_kudasai")
)

response = client.evaluate(
    EvaluationRequest(
        audio_path="path/to/user.wav",
        mode="reference",
    )
)

data = response.to_dict()
print(data["ok"])
print(data["user_facing"])
print(data["raw_result"]["details"])
```

Recommended product output:

- `response.user_facing["display_score"]`
- `response.user_facing["practice_check_result"]`
- `response.user_facing["user_messages"]`
- `response.user_facing["focus_feedback"]`

Debug/research output:

- `response.raw_result`
- `response.user_facing["debug"]`

## ASR-confirmed weak reference

ASR-generated reference must be two-step. Do not score raw ASR text directly.

```python
from jp_speech_eval import EvaluationRequest, SpeechEvaluationClient

client = SpeechEvaluationClient()

prompt = client.build_asr_confirmation("path/to/user.wav").to_dict()
print(prompt["prompt"]["asr_candidates"])
print(prompt["prompt"]["editable_text"])

# Show candidates to the user, then submit the selected or edited text.
response = client.evaluate(
    EvaluationRequest(
        audio_path="path/to/user.wav",
        mode="asr_confirmed_weak_reference",
        user_confirmed_text="ラーメンをください",
    )
)
```

This mode returns `weak_reference=true` in debug details. It is suitable for
practice feedback, not strict pronunciation correctness.

## ASR + Kanade playback mode

This is the same confirmed-ASR weak scoring path, with Kanade voice-conditioned
reference generated only for playback.

```python
response = client.evaluate(
    EvaluationRequest(
        audio_path="path/to/user.wav",
        mode="kanade_asr_confirmed_voice_reference",
        user_confirmed_text="ラーメンをください",
    )
)
```

Kanade output is marked:

- `demo_only=true`
- `exclude_from_pronunciation_score=true`

It should not be used as pronunciation correctness evidence.

## One-shot helper

```python
from jp_speech_eval import EvaluationRequest, evaluate_speech

data = evaluate_speech(
    EvaluationRequest(
        audio_path="path/to/user.wav",
        mode="reference",
        cache_path="cache/ramen_kudasai",
    )
)
```

## Supported public modes

- `reference`
- `asr_confirmed_weak_reference`
- `kanade_asr_confirmed_voice_reference`
- `acoustic`
- `transcript_assisted_light`

Legacy raw-ASR modes are intentionally guarded and may raise an error unless
the user has confirmed or edited the transcript.
