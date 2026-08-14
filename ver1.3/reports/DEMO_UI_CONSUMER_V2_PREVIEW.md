# Consumer Demo UI v2 Preview

## Status

- Branch: `free-assessment-integration-v1`
- Preview file: `debug_ui/consumer_v2.html`
- Existing `debug_ui/index.html`: unchanged
- Scoring/backend/API behavior: unchanged
- Hugging Face default page: unchanged until the preview is explicitly promoted

## Why this preview exists

The existing demo exposes useful research diagnostics, but the normal learner journey competes with reliability, endpointing, realtime replay, pitch details, and debug output. The preview reorganizes the same backend around a consumer-first path:

1. choose read-aloud or free-speaking practice;
2. see/listen to the target when one exists;
3. record or upload audio;
4. confirm ASR text in free-speaking mode;
5. see the product display score, a small set of available dimensions, and learner-facing feedback first;
6. expand technical analysis only when desired.

## Safety constraints

The preview deliberately does **not**:

- change `ProductScore v2`;
- enable ProductScore v3;
- invent a rhythm, pitch, or pronunciation score when the API marks it unavailable;
- average research/vendor evidence;
- hide ASR confirmation in free-speaking mode;
- remove the existing research UI;
- introduce a paid API.

## Visual/interaction changes

- reduced default card density;
- two primary learner tasks instead of exposing experimental mode taxonomy;
- one dominant recording action;
- product display score is visually primary;
- learner feedback appears before technical diagnostics;
- confidence is shown as supporting context rather than a separate research dashboard;
- detailed analysis is collapsed by default;
- raw JSON is nested one level deeper under detailed analysis;
- mobile layout is intentionally single-column and touch-friendly;
- multilingual shell remains available (`zh-CN`, `zh-TW`, `ja`, `en`).

## Preview locally

Start the existing debug server and open:

```text
http://127.0.0.1:8765/consumer_v2.html
```

The current public/default UI remains:

```text
http://127.0.0.1:8765/
```

## Promotion gate

Do not replace `debug_ui/index.html` or the Hugging Face Space default page until the preview has been manually checked for:

- microphone permission and recording;
- fixed-reading evaluation;
- free-speaking ASR confirmation;
- upload flow;
- missing-dimension rendering;
- mobile layout;
- all four language shells;
- public-space cold-start/failure behavior.

If the preview is worse, delete the preview commit/file; no production UI rollback is required because the old page was not replaced.
