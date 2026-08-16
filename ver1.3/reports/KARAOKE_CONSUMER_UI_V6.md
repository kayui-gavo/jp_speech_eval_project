# Karaoke Consumer UI v6

Status: engineering-complete consumer playback surface; no ProductScore change

Branch: `karaoke-consumer-ui-v6`

Base: `free-speech-promotion-gate-v5`

## 1. Why this round exists

The scoring and validation layers had become substantially more careful than the visible consumer UI.

The previous consumer page still exposed legacy wording such as `発音` and `ピッチ`, and its visible free-speaking path was built around an ASR pseudo-reference workflow. That presentation no longer matched the product contract:

1. 流暢さ
2. 明瞭さ
3. リズム
4. 抑揚

Nor did it match the direct target-independent free-speech evaluator introduced in the preceding rounds.

v6 therefore focuses on the C-end interaction loop:

**話す → 見える → 聞き返す → 一つ直す → もう一度話す**

The goal is not to make the debug page more decorative. The goal is to make already available evidence understandable and replayable without visually claiming more precision than the backend actually supports.

## 2. ProductScore is deliberately unchanged

v6 does not change:

- `consumer_four_score_v2`;
- component weights;
- the display transform;
- fixed-reference scoring equations;
- free-speech score mappings;
- reliability caps;
- language eligibility logic;
- promotion status of any v4/v5 shadow candidate.

The new timeline has:

- `score_role = visualization_only`;
- `product_score_changed = false`.

The API addition is additive. Existing raw and user-facing result objects remain available.

## 3. New consumer surface

Added:

- `debug_ui/consumer_v3.html`

The consumer page uses the existing restrained ivory / navy / rose / gold visual language, but its information hierarchy is redesigned around practice rather than diagnostics.

The primary result flow is:

1. record or upload;
2. see the practice score;
3. replay the utterance with karaoke-style synchronization;
4. inspect the four dimensions;
5. act on one focused suggestion.

The page is responsive for desktop and narrow mobile layouts.

### Consumer-visible modes

The page exposes only:

- `お手本練習` → fixed-reference evaluation;
- `自由に話す` → direct `transcript_assisted_light` broad-Japanese evaluation.

Legacy pseudo-reference modes remain available for internal compatibility/debugging, but are not exposed as the consumer definition of free speaking.

## 4. One playback timebase

A subtle but important synchronization bug was identified before the karaoke UI was implemented.

Free-speech ASR, pause detection, and F0 extraction run on the VAD-trimmed speech waveform, while the user listens back to the original uploaded/recorded waveform. Using raw ASR timestamps directly would therefore make subtitles early whenever the original recording begins with silence.

Added:

- `src/jp_speech_eval/app_core/karaoke_timeline.py`

The consumer timeline converts trim-relative evidence to:

`original_recording_playback_seconds`

using the detected speech-region offset.

Words, moras, pauses, pitch points, the scrubber, and the playback cursor therefore share one coordinate system.

## 5. Free-speech karaoke synchronization

When Faster-Whisper provides real word timestamps, the consumer UI:

- highlights the current recognized word during playback;
- dims completed words;
- allows a word to be clicked to seek to its actual timestamp;
- preserves ASR probability only as a subtle confidence cue.

Low ASR confidence is not rendered as a pronunciation error.

The timeline explicitly states:

`playback_alignment_not_phone_correctness`

When word timestamps are unavailable, the page does **not** divide utterance duration by character count or invent token timing. It falls back to static sentence display.

## 6. Fixed-practice mora karaoke

The fixed-reference evaluator already retains user-audio mora boundaries in `mora_table`.

v6 reuses these boundaries for playback synchronization.

If alignment evidence is sufficiently reliable, the UI shows:

- `拍同期 ON`

If the alignment used equal fallback, is explicitly unavailable, has missing confidence, or has confidence below the conservative visualization threshold, it shows:

- `拍同期・概算`

This distinction is visual honesty, not a new scoring threshold.

The mora row is clickable and seeks to the corresponding position in the user's own recording.

No mora is painted red/green as a correctness judgement.

The contract explicitly states:

`mora_playback_alignment_not_phone_correctness`

## 7. Karaoke-style voice movement visualization

### Free speech

The free-speech evaluator already computes a full F0 trajectory. Previously, that trajectory was reduced to statistics and discarded before the UI could use it.

v6 reuses the **same F0 pass** and exports a compressed visualization track. No second F0 extraction or additional speech model call is introduced.

The vertical axis is:

`semitone_relative_to_speaker_median`

This makes the curve describe movement rather than rewarding a higher absolute F0, and avoids turning voice-range/sex differences into an apparent quality judgement.

Unvoiced regions are retained as gaps rather than interpolated into invented pitch.

### Fixed practice

When per-mora user and reference F0 are available, v6 creates two independently normalized curves:

- solid rose: `あなた`;
- dashed gold: `お手本`.

Both are normalized around each speaker/reference median before comparison. The reference shape is mapped to the user's mora midpoints so the two contours can be inspected on the user's playback timebase.

The visualization is explicitly:

`mora_aligned_voice_movement_shape_not_strict_pitch_accent_correctness`

It is useful for seeing whether a phrase rises, falls, or stays relatively flat in a similar region, but it is not presented as lexical pitch-accent correctness or accent-nucleus detection.

## 8. Pause visualization

Detected long silent pauses are rendered as lightly shaded time regions.

They are independently switchable from pitch movement.

The contract states:

`acoustic_pause_not_automatically_a_fluency_error`

A pause can be normal discourse planning, phrasing, or hesitation. v6 therefore visualizes it without automatically colouring it as a mistake.

## 9. Evidence state is visible without exposing research jargon

The backend already distinguishes evidence states, but the old consumer UI hid this distinction.

v6 maps those states to compact consumer copy:

- `measured_proxy` → `音声から確認`;
- `broad_proxy` → `大まかな目安`;
- `neutral_prior` → `参考値`;
- unavailable → `未測定`.

This matters especially for a neutral score such as 70. A neutral placeholder can still be shown for a friendly C-end experience, but the UI no longer makes it look identical to a directly measured score.

Recording analyzability is displayed separately and explicitly marked as not being a scored speaking dimension.

## 10. Public demo behaviour

The debug server now provides a stable consumer route:

- `/consumer`

When launched with `--public-demo`, the root `/` also opens the consumer practice surface.

Normal local-development mode keeps the original debug root so internal diagnostics remain easy to access.

Public-demo safety defaults from the existing server remain in force, including no retained uploads and no evaluation JSONL logging.

## 11. Browser-side interaction

The page implements:

- microphone capture;
- mono WAV creation/downsampling to 16 kHz;
- upload fallback;
- live input-level meter;
- replay/pause;
- scrubber;
- playback cursor;
- timestamp-driven word/mora highlighting;
- click-to-seek lyrics;
- pause layer toggle;
- voice-movement layer toggle;
- mobile-responsive layout;
- reduced-motion handling.

The inline JavaScript is syntax-checked by Node in CI when Node is available.

The DOM is also checked for duplicate ids and several semantic regressions.

## 12. Guardrails frozen in code and tests

The timeline explicitly records that:

- word timestamps are not phone correctness;
- mora alignment is not phone correctness;
- the pitch curve is not lexical pitch-accent correctness;
- a pause is not automatically an error;
- recording quality is not a speaking dimension;
- missing evidence must not be rendered as bad performance.

Static UI tests additionally prohibit regression back to the old top-level `発音 / ピッチ` wording and prohibit consumer exposure of pseudo-reference free speaking.

## 13. Validation

The v6 lightweight regression suite has passed on the karaoke implementation with:

- 148 tests passed;
- 3 dependency deprecation warnings from the audio compatibility stack;
- the existing historical reliability-cap audit unchanged.

The regression coverage includes:

- original-recording timebase conversion;
- real ASR timestamp synchronization;
- no fabricated character timing;
- fixed-reference mora synchronization;
- approximate alignment labelling;
- independently normalized user/reference F0 contours;
- exact four-dimensional consumer semantics;
- evidence-state presentation;
- browser JavaScript parsing;
- DOM id uniqueness;
- upload-control markup;
- stable consumer route;
- public-demo consumer landing.

## 14. What has not been claimed

GitHub CI does not replace a physical-device usability pass.

This round has not yet demonstrated:

- microphone capture behaviour on real iPhone Safari;
- microphone capture behaviour on real Android Chrome;
- visual smoothness on low-end mobile hardware;
- real-world perceived subtitle synchronization across a diverse set of recordings;
- end-to-end latency under the deployed ASR/model hardware;
- accessibility with screen readers;
- that the current score candidates are criterion-valid simply because the UI can display them well.

These remain separate acceptance tasks.

## 15. Next high-value work

The next useful work should combine product and validation rather than add decorative layers:

1. run real-device smoke tests for recording/replay and responsive layout;
2. benchmark end-to-end C-end latency, especially the word-timestamp cost;
3. use real Japanese recordings to inspect perceived word/mora synchronization drift;
4. connect session history so a learner can replay `前回 → 今回` on the same visual language;
5. only add finer-grained error colouring when a validated construct-specific detector actually supports it.

The UI should remain capable of becoming more game-like, but visual certainty must never outrun evidence certainty.
