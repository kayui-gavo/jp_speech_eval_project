# Verified Reference Asset Validation

- generated_at: 2026-06-19T02:56:06+00:00
- targets: 5
- strong_pitch_reference_targets: 0

## Validation Table

| target_id | audio | source | status | sidecar | timing | f0_coverage | strong | blocking_reasons |
|---|---|---|---|---|---|---:|---|---|
| coffee_kudasai | no/no | auto_pyopenjtalk | auto_pyopenjtalk | no/no |  |  | no | missing_reference_audio_path;missing_sentence_cache;tts_or_pseudo_reference;untrusted_reference_source;verification_status_not_verified;missing_verified_by |
| eki_made_onegaishimasu | no/no | auto_pyopenjtalk | auto_pyopenjtalk | no/no |  |  | no | missing_reference_audio_path;missing_sentence_cache;tts_or_pseudo_reference;untrusted_reference_source;verification_status_not_verified;missing_verified_by |
| mou_ichido_onegaishimasu | no/no | auto_pyopenjtalk | auto_pyopenjtalk | no/no |  |  | no | missing_reference_audio_path;missing_sentence_cache;tts_or_pseudo_reference;untrusted_reference_source;verification_status_not_verified;missing_verified_by |
| ramen_kudasai | yes/yes | pyopenjtalk_tts_pseudo_reference | auto_pyopenjtalk | no/no | equal_mora | 1.0 | no | missing_prosody_sidecar;sidecar_unreliable;weak_or_approximate_mora_timing;tts_or_pseudo_reference;untrusted_reference_source;verification_status_not_verified;missing_verified_by |
| sumimasen | no/no | auto_pyopenjtalk | auto_pyopenjtalk | no/no |  |  | no | missing_reference_audio_path;missing_sentence_cache;tts_or_pseudo_reference;untrusted_reference_source;verification_status_not_verified;missing_verified_by |

## Interpretation

- No packaged target currently satisfies the strong pitch reference asset requirements.
- TTS/OpenJTalk/pseudo references are blockers, even if a manifest claims verification.
- Equal-mora or fallback timing is not accepted as a strong pitch reference alignment source.
- Test-only JVS fixtures are intentionally outside the packaged demo manifest.
