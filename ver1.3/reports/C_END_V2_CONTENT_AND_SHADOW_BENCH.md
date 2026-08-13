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

## Engineering negative controls

- Generated and labelled `synthetic_engineering_control`: silence, white/pink noise, tone, burst noise, English system TTS, Mandarin system TTS.
- After the fallback eligibility repair, silence, white noise, tone, burst noise, English TTS, and Mandarin TTS produce no normal Japanese practice score. Pink-like noise is also rejected after the final `f0_coverage < 0.10` fallback guard; its first run exposed the ASR-hallucination bug.
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

## Merge gate

**PASS WITH ISSUES for the content-gate repair, but BLOCK MERGE for score-calibration closeout.** The hard mismatch semantic regression is repaired by always-ASR small and engineering negatives are rejected; production score ceiling/non-monotonic channel behavior remains unresolved and shadows remain non-production.
