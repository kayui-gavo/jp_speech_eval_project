# ASR/TTS Mode-Specific Path Audit

This audit describes the current code path at `5bed10b`. It does not change runtime behavior.

## Mode Matrix

| mode | ASR role | TTS role | impacts scoring | impacts feedback | impacts UX | notes |
|---|---|---|---|---|---|---|
| arbitrary speech / ASR-confirmed weak-reference | Produces candidate text and language evidence; the user-confirmed text determines kana/mora content and score eligibility | Builds a pseudo-reference cache used by cached alignment; it is held fixed and is not a strict pitch target | ASR: yes, through eligibility and text/mora representation. TTS: indirectly, through alignment/timing-derived proxies; not as strict pitch correctness | ASR: yes, through content and localization evidence. TTS: potentially through alignment-derived timing evidence | ASR reduces correction burden; TTS supplies playback/shadowing audio | Weak prosody naturalness uses user F0 coverage/range/movement, not strict OpenJTalk/TTS contour matching |
| fixed-reference / TTS pseudo-reference | Optional content confirmation only; not the principal comparison variable | Generates reference audio, reference F0, and approximate timing cache | Yes. Reference contour/timing quality can constrain reference-based prosody and alignment | Yes, because reference-based evidence can alter which detailed advice is supportable | Yes, as the sound the learner imitates | Synthetic TTS provenance remains weak and cannot be promoted to human/native ground truth |
| verified fixed-reference human/native | Optional content confirmation; fixed text is known | Should be absent from the strict baseline; may remain as separate playback UX only | Human/native F0 cache and verified timing are the strict reference | Strong pitch feedback is allowed only when reliability and other gates pass | Optional TTS playback must be clearly separated from the verified scoring reference | `reference_audio_f0_cache` with verified provenance is the strict pitch baseline |

## Arbitrary-Speech Weak Reference

1. ASR enters transcript candidacy, language/content evidence, kana/mora construction after confirmation, no-score decisions, and potential timestamp anchors.
2. TTS currently creates the cache consumed by `cached_dtw`, so it is not literally absent from all score inputs. It may influence alignment and timing-related proxies.
3. TTS is not the formal pitch judge in the weak native-likeness path. Weak pitch uses recording-side F0 coverage, range, local movement, smoothness, and final behavior.
4. The mode-specific ASR benchmark therefore fixes TTS and varies only transcript/timestamp evidence.

## Fixed Pseudo Reference

1. TTS can generate the reference WAV and cache.
2. Reference F0 and timing may originate from that synthetic cache; poor prosody or equal-mora timing can directly constrain reference-based scores.
3. A better TTS could improve imitation UX and pseudo-reference consistency, but quality alone cannot change its provenance from `synthetic_tts` / weak to verified human/native.

## Verified Fixed Reference

1. Verified human/native audio and non-fallback timing should supply the strict F0 baseline.
2. TTS should be excluded from strict scoring. A TTS playback control may coexist only if the UI and metadata distinguish it from the scoring reference.
3. The local JVS test-only fixtures demonstrate this path's upper bound; they do not supply verified references for packaged demo sentences.

## Boundary

Provider upgrades alone do not establish strict pitch-accent correctness. That also requires verified sentence references, trustworthy mora timing, real negative controls (especially wrong accent drops), calibration, and human validation.
