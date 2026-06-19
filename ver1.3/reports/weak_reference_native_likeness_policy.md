# Weak-reference native-likeness policy

## Why arbitrary sentences are not strict fixed-reference scoring

For fixed-reference pronunciation assessment, the system can compare the learner
against a verified target sentence, known mora timing, and a trusted reference
audio contour. That is a text-dependent, reference-based setting.

For arbitrary speech after ASR/user confirmation, there is usually no verified
human/native reference audio. In that setting, forcing the utterance to match a
TTS or OpenJTalk contour overclaims the evidence. The result should therefore be
a practice/native-likeness score, not teacher-grade pitch-accent correctness.

## What the weak score means

`weak_reference_native_likeness` estimates whether the recording sounds broadly
natural enough for practice:

- enough voiced F0 evidence,
- non-flat but not wildly unstable pitch movement,
- reasonable local pitch transitions,
- usable phrase-final intonation evidence,
- rhythm and fluency proxies from existing timing features.

It intentionally does not say that a specific accent nucleus is correct or
incorrect.

## Why native speech should score high here

Native speech should not be penalized just because OpenJTalk predicts a different
H/L contour. In weak-reference mode, JVS native audio is used as the baseline:
normal native pitch movement should receive high weak prosody naturalness, while
flat or shuffled pitch controls should be lower.

## JVS and JANON roles

- JVS: native baseline and counterfactual controls.
- JANON: learner trend sanity check only. It is useful for checking whether the
  score behaves plausibly across learner groups, but it is not calibration by
  itself.

## OpenJTalk role

OpenJTalk may provide kana, mora segmentation, and weak accent hints. In
arbitrary-sentence mode it is not a reliable pitch reference and must not be used
as strict pitch-accent ground truth.

## User-facing pitch feedback

Weak-reference pitch feedback may say:

- 音高变化可能偏平。
- 句末语调可能不够清楚。
- 音高变化仅供参考。

It should not say:

- 高低重音错了。
- 第 N 拍的アクセントが違う。
- Teacher-grade pitch accent correctness.

## Current limitation

Wrong accent-drop controls are still weakly separated by the broad naturalness
score. That is acceptable for this mode because it is not strict accent
correctness. If product needs accent-drop correctness, it must use verified
fixed-reference audio or a separately validated pitch-accent model.

## What this does not claim

This policy does not claim full pronunciation scoring, strict pitch-accent
correctness, teacher-level feedback, or calibrated learner proficiency. It only
creates a safer practice score for arbitrary confirmed Japanese text.
