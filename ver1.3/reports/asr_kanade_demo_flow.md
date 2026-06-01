# ASR + Kanade demo flow

ASR + Kanade is the product-facing highlight mode: the user can hear an ideal reference in a voice color close to their own. It is not a correctness scorer.

## Flow

```text
user recording
-> ASR
-> user confirms / edits text
-> TTS pseudo-reference
-> Kanade voice conversion
-> play personalized reference audio
-> evaluation uses non-Kanade reference/features only
-> render UserFacingResult
```

## What Kanade does

- Converts the pseudo-reference playback toward the user's voice color.
- Helps the user compare “my current recording” and “a suggested ideal version closer to my voice.”
- Improves demo experience because the reference feels less like a generic machine voice.

## What Kanade does not do

- It does not define pronunciation correctness.
- It does not contribute to `practice_score`.
- It does not produce a `similarity_to_kanade` score.
- It is not ground truth.

## User-facing copy

Use:

- これはあなたの声に近い参考音です。声の似ている度合いは採点していません。
- 認識された文をもとにした参考判定です。厳密な発音評価ではありません。

Avoid:

- Kanade 音声に近いほど発音が正しい
- 声質の一致を発音点として採点
- ASR raw text を正解として採点

## Demo state

In `response.user_facing`, Kanade mode should normally be `debug_only` or clearly marked as practice reference. The UI may still show playback controls and simple guidance, but correctness scoring must remain based on non-Kanade evidence.
