# ASR + Kanade product flow

ASR + Kanade is a playback-centered practice mode. It is not a stricter scoring mode.

## Flow

1. User speaks freely.
2. ASR produces candidate text.
3. User confirms or edits the text.
4. The confirmed text becomes a weak target.
5. TTS/reference audio is generated as a pseudo-reference.
6. Kanade converts the reference toward the user's voice color for playback.
7. The app plays the user-voice-style ideal reference.
8. Scoring uses non-Kanade reference features, content, rhythm, fluency, and reliable evidence only.

## Product policy

- Kanade is only for playback and auditory feedback.
- Kanade output is excluded from pronunciation correctness scoring.
- Similarity to the Kanade voice is not evaluated.
- ASR raw result must not become a scoring reference until the user confirms or edits it.
- ASR-generated reference is weak-reference practice; it should not claim strict pronunciation correctness.
- Special mora feedback is conservative and hidden by default.

## User-facing wording

Use:

- 自分の声に近い参考音で，理想の言い方を聞いてみましょう。
- 確認した文をもとにした練習用フィードバックです。

Avoid:

- Kanade 音声に似ているほど発音が正しい
- ASR が出した文をそのまま正解として採点しました
