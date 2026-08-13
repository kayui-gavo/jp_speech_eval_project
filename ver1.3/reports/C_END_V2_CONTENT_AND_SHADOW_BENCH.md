# C-end v2 Content and Shadow Bench

## Content verification

|system|correct false mismatch|wrong false verified|cold sec|warm p50|warm p90|
|---|---:|---:|---:|---:|---:|
|acoustic_only|0.0|0.98||0.0|0.0|
|faster_whisper_base|0.1667|0.0|1.2289|0.6577|0.7474|
|faster_whisper_small|0.0667|0.0|2.492241|1.6544|1.8773|
|faster_whisper_tiny|0.2667|0.0|0.957955|0.4543|0.5022|

MFCC acoustic-only uses `acoustic_likely_match` only; it never sets `content_verified=true`. The recommended production policy is selected from the lowest false-verified ASR option subject to observed latency, rather than from a new DTW threshold.

- Manifest: 30 correct pairs and 50 same/cross-speaker duration-matched wrong-target pairs (`0.90–1.10` for every hard negative).
- Decision: **always faster-whisper small**. It had the lowest correct-target false-mismatch rate (6.67%) while every ASR model had 0/50 wrong-target false verification. Tiny/base remain useful latency baselines, not the default verification policy.
- New semantic contract: `content_verified=true` is emitted only at `verification_level=asr_verified`; MFCC-only evidence is `acoustic_likely_match=true`, never a verification claim.
- Final rerun on this branch (`small`, 30 correct / 50 hard wrong): **2/30** correct-target mismatches (6.67%) and **0/50** wrong-target false verifications. Cold latency was 6.375 s and warm p50/p90 were 2.806/3.319 s on this machine.

## Engineering negative controls

- Generated and labelled `synthetic_engineering_control`: silence, white/pink noise, tone, burst noise, English system TTS, Mandarin system TTS.
- Target verification and broad-fallback eligibility are deliberately separate. Target verification remains Japanese-biased; the fallback additionally requires speech/VAD evidence, transcript sanity, independently requested ASR language evidence, and conservative lexical coherence. It does **not** use a `has_hiragana` rule.
- On this machine, unforced faster-whisper language ID is not reliable by itself for short system English/Mandarin controls: it can report `ja`. The English transcript is rejected by transcript sanity; the Mandarin hallucination (`全地化普通クコ`) is rejected as an ambiguous multi-token content-fragment sequence. This is a safety eligibility decision, not a pronunciation judgment.
- The exact short-form matrix is covered by regression tests and a system-TTS evidence replay: `はい`, `いいえ`, `寿司`, `東京`, `ラーメン`, `コーヒー`, and `ありがとうございます` all remain eligible. The system voice caused low-confidence `en` labels for `いいえ` and `ラーメン`; those do not veto Japanese eligibility, while a confident non-Japanese label does.
- After the fallback eligibility repair, silence, white noise, pink-like noise, tone, burst noise, English TTS, and Mandarin TTS produce no normal Japanese practice score. The synthetic suite is intentionally retained as an engineering control, not represented as human negative-speech data.
- This is only an engineering control suite. It is not presented as human English/Mandarin/noise data; real negative controls remain a coverage gap.

## Product calibration ladder

- Conditions measured: 13. All raw evidence is in `outputs/content_shadow_v2_final/product_calibration_ladder.csv`; no score mapping was changed.
- Current score ceiling remains an upstream saturation finding; this bench records its components rather than compressing the displayed curve.
- The ladder has 13 conditions: ±10/20/30% speed, pause insertion, and 25/15/8 dB noise. It demonstrates desired degradation for a 0.45 s inserted pause (89 → 80), but non-monotonic values for speed and noise (8 dB noise reached 90). No display curve or ProductScore weight was changed; this requires a later calibration patch, not cosmetic clamping.
- `details.calibration_features` now preserves continuous `mora_duration_cv`, speech-rate distance, pause excess/density, and special-mora penalty before product rounding.

## WavLM multi-reference

|layer|native median|native SD|wrong median|separation|
|---:|---:|---:|---:|---:|
|6|0.28157|0.0242|0.54368|0.26211|
|12|0.22045|0.01654|0.49966|0.27921|
|18|0.17124|0.01792|0.43101|0.25977|
|24|0.22569|0.02545|0.67984|0.45415|

- Layer 24 has the largest median wrong-target separation (0.45415); layer 12 has the smallest native leave-one-speaker-out SD (0.01654). No /100 mapping is proposed. Median aggregation is the conservative default experiment; nearest reference is reported separately because it can be overly permissive.

## Special mora v3, phrase intonation v2, and accent nucleus v1

- Real JANON `ばっちり` / `うっとうしい` and JVS panel runs completed as shadow-only. Sokuon now exposes neighbor-relative `minimum_energy_to_neighbor_ratio`, longest low-energy run, low-energy fraction, voicing interruption, and following context. Long vowels expose a combined vowel-nucleus duration/continuity representation; moraic nasals expose context class only.
- Many real panel ROIs used equal-boundary fallback and therefore correctly report `unavailable_alignment_or_roi_unreliable`, not phone correctness. A closure-shortening counterfactual is still required before any sokuon decision can be claimed. `user_facing=false` remains enforced.
- Phrase intonation now emits raw reference-relative contour correlation/RMSE, adjacent transition agreement, and final-movement difference; it has no arbitrary candidate score. The adjacent-F0-gap bug is covered by a unit test.
- Accent output is now per accent phrase. OpenJTalk-derived targets remain `weak_target=true`, so no correct/incorrect claim is emitted. The audited target set did not include human/OJAD-reviewed phrase labels; that is a coverage gap, not a negative correctness result.

## Shadow guarantees

- WavLM, special-mora v3, phrase intonation v2, and accent-nucleus v1 remain default-off, exception-isolated, and outside ProductScore/practice feedback.
- Sokuon uses neighbor-relative low-energy/closure evidence; long vowels use a combined vowel nucleus; moraic nasals report context classes only.
- Phrase intonation no longer emits an arbitrary /100 mapping and no longer bridges missing-F0 morae. Accent analysis is per accent phrase and only computes target correctness for strong target provenance.

## Merge and release gates

### MERGE GATE — PASS WITH ISSUES

The branch is safe to merge as a content-gate/architecture repair: default configuration now agrees on always-ASR, content verification has the always-ASR small benchmark evidence, engineering negative controls are rejected by the complete fallback path, and shadows remain default-off and outside ProductScore. The test suite is the merge verification authority.

### RELEASE / SCORE-CALIBRATION GATE — BLOCKED

This is intentionally a separate later release blocker, not a reason to retain safe content fixes outside `main`: display-score ceiling remains, `fallback_equal` can create artificial CV≈0, rhythm/fluency inputs remain saturated in parts of the ladder, and pronunciation is still partly a timing proxy. No score mapping, ProductScore weight, special-mora threshold, or shadow-to-user-score connection was changed in this branch.
