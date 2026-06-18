# Fixed-reference prosody target inventory

- generated_at: 2026-06-18T16:57:03+00:00
- targets: 5
- strong_pitch_reference_targets: 0

## Inventory

| target_id | reference_source | ref_wav | sidecar | sidecar_reliable | timing | f0_coverage | strong_pitch | reason_if_not |
|---|---|---|---|---|---|---:|---|---|
| coffee_kudasai |  | no | no | no |  |  | no | missing_sentence_cache |
| eki_made_onegaishimasu |  | no | no | no |  |  | no | missing_sentence_cache |
| mou_ichido_onegaishimasu |  | no | no | no |  |  | no | missing_sentence_cache |
| ramen_kudasai | pyopenjtalk_tts_pseudo_reference | yes | no | no | equal_mora | 1.0 | no | manifest_reference_audio_missing;untrusted_reference_source;manifest_claims_human_checked_but_cache_not_verified |
| sumimasen |  | no | no | no |  |  | no | missing_sentence_cache |

## Interpretation

- No packaged fixed-reference target currently has a verified reliable human/native reference F0 sidecar.
- Targets without reliable sidecars should remain weak/practice/debug for pitch correctness.
- Do not use `--verified-reference` on TTS/OpenJTalk pseudo references; verified means provenance has been checked as human/native/teacher audio.
