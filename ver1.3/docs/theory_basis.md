# Theory basis and current limitations

This document is for project discussion. It separates reasonably supported design choices from engineering defaults that still need calibration.

## 1. Mora-level prosody is the core Japanese-specific unit

For Japanese pitch accent, the useful analysis unit is not the English-like syllable, but the mora. For example, `ラーメン` is analyzed as `ラ・ー・メ・ン`. In this project, prosody is therefore represented as:

- mora-level F0 representative value
- adjacent mora F0 direction: up / down / flat
- sentence-final intonation trend

This follows the direction of Minematsu / Hirose style Japanese accent training work, where learner accent is analyzed using F0 movement around mora units and feedback is given visually and acoustically.

## 2. F0, duration, and power are practical acoustic feedback signals

The current acoustic module uses:

- F0: pitch accent / intonation
- duration: mora rhythm, long vowel, nasal `ン`, sokuon `ッ`
- endpointed speech duration: speech rate and duration-normalized feedback
- pause: fluency and hesitation inside the detected speech region
- energy/power: volume and expression/style proxy

These are not arbitrary: they correspond to standard speech-signal correlates of prosody and pronunciation training feedback. However, the exact thresholds in `configs/scoring_config.json` are engineering defaults.

Pronunciation and prosody are the core pronunciation-related dimensions in this prototype. Fluency is delivery/style, not pronunciation correctness. Tone/emotion is expression/style, not pronunciation correctness.

## 3. MFCC-DTW is alignment support, not final pronunciation correctness

MFCC-DTW is used to approximately align a user utterance with a precomputed TTS reference. It is useful for obtaining approximate mora boundaries. It should not be treated as the final pronunciation score, because MFCC contains speaker/timbre differences and may not isolate phonemic contrasts.

Future pronunciation scoring should add at least one of:

- ASR + kana edit distance
- candidate sentence matching
- CTC posterior alignment
- GOP-like phoneme/mora posterior score
- posteriorgram-based fast DTW

## 4. Realtime design: two-layer evaluation

Product target: karaoke-like feedback while the user is speaking.

The architecture is split into two layers:

1. Realtime coarse feedback
   - frame-level RMS / F0 / endpoint state
   - approximate mora progress from cached reference duration
   - fast UI feedback: waiting_for_speech, speaking, maybe_ending, ended, plus volume and pitch only after speech starts

2. Sentence-final detailed feedback
   - endpointing / VAD trimming of fixed-duration recordings
   - cached DTW mora alignment
   - pyworld/librosa F0 extraction
   - mora-level prosody comparison
   - fluency / expression-style proxy / feedback text

The realtime layer should be low latency. The sentence-final layer can be delayed by a few hundred milliseconds.

## 5. Current thresholds are not research claims

The following values are currently engineering defaults:

- speech rate range
- endpointing RMS / voicing threshold
- long pause threshold
- F0 direction threshold
- pitch range threshold for flat/exaggerated tone
- energy threshold for low volume

Before research presentation, these should be calibrated with:

- native speaker recordings
- learner recordings
- teacher labels
- listener naturalness ratings
- ablation experiments

## 6. Suggested research ideas generated from product constraints

- Low-cost mora-level prosody assessment on mobile CPU
- Robust alignment under disfluency, pause, and learner pronunciation
- Speaker-normalized F0 comparison across gender/age/voice type
- Separating pitch accent errors from emotional intonation
- Converting acoustic features into actionable feedback for learners
