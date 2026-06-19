# Verified Reference Audio Asset Specification

- generated_at: 2026-06-19 JST
- scope: packaged fixed-reference demo targets only
- purpose: define when a target may be treated as a strong pitch/prosody reference

## Reference Audio Requirements

A fixed-reference target may use strong pitch/prosody feedback only when the reference audio is a verified human/native/teacher recording.

Required audio properties:

- Voice source: human native speaker or teacher-approved speaker.
- Not allowed: TTS, OpenJTalk, pseudo reference, synthetic voice, unverified ASR/TTS reconstruction.
- Text match: the spoken sentence must exactly match `target_text`.
- Reading match: the expected kana/mora sequence must match `target_kana`.
- Format: mono WAV, 16-bit PCM preferred, 24 kHz preferred for newly recorded assets; 16 kHz is accepted by the current repo pipeline.
- Recording quality: quiet room, no clipping, no music/noise bed, no long leading/trailing silence, natural single-sentence delivery.
- Takes: multiple takes are allowed, but one take must be selected and recorded as the active reference.
- Provenance must include `speaker_id`, `take_id`, `recorded_at` when available, `verified_by`, and source/license or usage note.

## Required Manifest Fields

Each packaged strong reference target should provide:

- `target_id`
- `target_text`
- `target_kana`
- `reference_audio_path`
- `reference_source`
- `reference_speaker_id`
- `reference_take_id`
- `verified_by`
- `verification_status`
- `timing_source`
- `license_or_usage_note`
- `quality_flags`

The legacy `reference_audio` key may remain for compatibility, but `reference_audio_path` should be populated for new verified assets.

## Strong Pitch Reference Conditions

All of these must be true:

- Source is verified human/native/teacher audio.
- Reference path exists in the packaged repo.
- Audio is readable.
- Audio is mono WAV and duration is plausible for the target sentence.
- Target text/kana/mora count matches the cache and sidecar.
- F0 coverage is sufficient.
- Mora timing is not fallback and not equal-mora approximation.
- A `<cache>.prosody_ref.json` sidecar exists.
- Sidecar is reliable.
- No provenance inconsistency is present.

## Explicitly Forbidden

- TTS/OpenJTalk/pseudo references must not become reliable automatically.
- A missing reference path must not coexist with `human_checked` or `verified` status.
- Equal-mora timing must not be silently treated as strong alignment.
- Debug or test-only JVS fixtures must not be mixed into the packaged demo manifest.
- A reliable sidecar must not be created only to make the UI look better.

## Current Product State

The packaged demo currently has no strong pitch reference target. `ramen_kudasai` has a TTS/OpenJTalk pseudo reference and must remain weak until a verified human/native/teacher reference recording and reliable timing sidecar are added.

