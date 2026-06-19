# Fixed-reference prosody target inventory

- generated_at: 2026-06-19T02:56:06+00:00
- targets: 5
- strong_pitch_reference_targets: 0

## Inventory

| target_id | manifest_level | manifest_pitch_status | reference_source | ref_wav | sidecar | sidecar_reliable | timing | f0_coverage | strong_pitch | reason_if_not |
|---|---|---|---|---|---|---|---|---:|---|---|
| coffee_kudasai | auto_pyopenjtalk |  |  | no | no | no |  |  | no | missing_sentence_cache |
| eki_made_onegaishimasu | auto_pyopenjtalk |  |  | no | no | no |  |  | no | missing_sentence_cache |
| mou_ichido_onegaishimasu | auto_pyopenjtalk |  |  | no | no | no |  |  | no | missing_sentence_cache |
| ramen_kudasai | auto_pyopenjtalk | weak_tts_pseudo_reference | pyopenjtalk_tts_pseudo_reference | yes | no | no | equal_mora | 1.0 | no | untrusted_reference_source;weak_tts_pseudo_reference |
| sumimasen | auto_pyopenjtalk |  |  | no | no | no |  |  | no | missing_sentence_cache |

## Interpretation

- No packaged fixed-reference target currently has a verified reliable human/native reference F0 sidecar.
- Targets without reliable sidecars should remain weak/practice/debug for pitch correctness.
- Do not use `--verified-reference` on TTS/OpenJTalk pseudo references; verified means provenance has been checked as human/native/teacher audio.
