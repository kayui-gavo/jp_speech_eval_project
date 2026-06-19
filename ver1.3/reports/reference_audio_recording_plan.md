# Reference Audio Recording Plan

- generated_at: 2026-06-19 JST
- scope: current packaged fixed-reference targets

## Plan

| target_id | target_text | target_kana | speakers | takes | current_status | action_required |
|---|---|---|---:|---:|---|---|
| ramen_kudasai | ラーメンをください | ラーメンヲクダサイ | 2 | 3 | weak_tts_pseudo_reference | Record human/native/teacher reference and add non-fallback mora timing sidecar. |
| coffee_kudasai | コーヒーをください | コーヒーヲクダサイ | 2 | 3 | no_reference_audio | Record human/native/teacher reference and build cache/sidecar. |
| sumimasen | すみません | スミマセン | 2 | 3 | no_reference_audio | Record human/native/teacher reference and build cache/sidecar. |
| mou_ichido_onegaishimasu | もう一度お願いします | モーイチドオネガイシマス | 2 | 3 | no_reference_audio | Record human/native/teacher reference and build cache/sidecar. |
| eki_made_onegaishimasu | 駅までお願いします | エキマデオネガイシマス | 2 | 3 | no_reference_audio | Record human/native/teacher reference and build cache/sidecar. |

## Required Capture Format

- Mono WAV.
- 16-bit PCM.
- 24 kHz preferred for new recordings; 16 kHz accepted by the current pipeline.
- Quiet room, no clipping, no background music/noise.
- One natural sentence per file.

## Verification

Each selected take needs `speaker_id`, `take_id`, `verified_by`, verification date, and a license/usage note before it can be considered for a strong pitch reference sidecar.

