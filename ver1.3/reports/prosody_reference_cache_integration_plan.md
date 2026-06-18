# Prosody Reference Cache Integration Plan

- generated_at: 2026-06-19 JST
- scope: reference target provenance and cache selection only
- non-goals: calibration, aggregate weights, UI changes, ASR/TTS provider integration

## Existing Repo Structure

The current fixed-reference path already has a sentence cache:

- `cache/<target>.json`: text, kana, mora sequence, OpenJTalk-derived pitch labels, reference provenance, and reference mora boundaries.
- `cache/<target>.npz`: reference waveform features including `ref_f0_times` and `ref_f0`.
- optional `cache/<target>.ref.wav`: saved reference waveform.

`evaluator.py` previously computed mora-level reference F0 from `SentenceCache.ref_f0_times/ref_f0/ref_mora_boundaries` and passed it to `score_prosody`. The missing piece was provenance: any reference F0 was surfaced as `tts_reference`, so reliable human reference audio and weak TTS pseudo-reference were not clearly separated.

## Target Source Priority

1. `reference_audio_f0_cache`
   - Sidecar cache exists at `<cache>.prosody_ref.json`.
   - Cache is marked reliable.
   - Mora count matches the target.
   - Reference F0 coverage is sufficient.
   - Reference provenance is trusted or explicitly verified.

2. `reference_audio_f0_runtime`
   - No sidecar cache exists.
   - Sentence cache has trusted human/native/teacher/verified reference provenance.
   - Runtime mora-level reference F0 has sufficient coverage.
   - This is accepted as a temporary runtime path, but the long-term path should generate a sidecar cache.

3. `tts_reference_weak` / `reference_audio_f0_runtime_weak`
   - Reference F0 exists but provenance or timing is not reliable enough for strong pitch correctness.
   - It can still support raw diagnostics and practice proxies.
   - It must not be treated as a formal pitch target.

4. `openjtalk_accent_phrase_chain`
   - Text-front-end pitch labels only.
   - Kept as weak/debug/fallback target.
   - Not promoted to a strong user-facing pitch correctness reference.

## Sidecar Cache Fields

The new sidecar cache schema is `prosody_reference_cache_v1` and includes:

- `target_id`
- `target_text`
- `target_kana`
- `target_mora_sequence`
- `reference_audio_path`
- `reference_source`
- `reference_f0_mora_values`
- `reference_f0_smoothed_values`
- `voiced_mora_mask`
- `mora_timing_source`
- `pitch_target_source`
- `pitch_target_reliability`
- `f0_coverage`
- `cache_version`
- `created_at`
- `quality_flags`
- `reliable`

## Quality Gates

A sidecar cache is reliable only when:

- F0 coverage is at least the configured minimum, currently `0.50`.
- F0 value count matches the target mora count.
- Reference timing is not marked as fallback.
- Reference source is trusted, or the builder was called with `--verified-reference`.

Warning-only flags may still be recorded, for example `equal_mora_timing_approx`, because equal timing is useful for diagnostics but should remain visible in provenance.

## User-Facing Pitch Conditions

Pitch/prosody can be strongly user-facing only when all of these hold:

- pitch target source is `reference_audio_f0_cache` or trusted `reference_audio_f0_runtime`
- `pitch_target_reliability = reliable`
- content gate passes
- alignment is not fallback
- F0 coverage is sufficient
- recording/reliability gates pass

If target source is OpenJTalk-only, TTS pseudo-reference, unreliable sidecar, low-F0, content mismatch, or fallback alignment, pitch details remain debug/practice only or unavailable according to the existing UI contract.

## Why OpenJTalk Was Downgraded

The previous diagnostics showed:

- self-oracle reference contour: near-perfect native score
- smoothed self target: high native score
- OpenJTalk target: much lower native score
- flat/shuffled counterfactuals: lower than native when reference contour is correct

So the primary issue is not that `score_prosody` cannot rank contours at all. The weak point is using a tool-generated accent phrase chain as if it were the real F0 target for native speech. OpenJTalk remains useful for text structure, H/L labels, and fallback diagnostics, but not as a strong pitch correctness ground truth.

## Still Not Claimed

This integration does not prove pitch/prosody is ready for stable formal scoring. The remaining work is:

- build verified reference caches for real fixed-reference targets
- validate reference timing quality, ideally with lab or forced alignment
- run cross-speaker same-sentence sanity checks
- run audio-level flat/random/wrong-accent counterfactuals
- only then consider calibration

