# HF Space Karaoke Integration v7

Status: engineering-complete integration into the established Hugging Face Space UI; not deployed or merged

Branch: `hf-space-karaoke-integration-v7`

Base: `karaoke-consumer-ui-v6`

## 1. Baseline correction

The established public demo is the Hugging Face Space published by `deploy/publish_hf_space.sh` and started by `deploy/start_full_demo.sh`.

Its primary UI is:

- `ver1.3/debug_ui/index.html`

It is **not** `consumer_v2.html` or `consumer_v3.html`.

v7 therefore does not replace the public UI with the experimental `consumer_v3` surface. The existing Space layout, four-language switch, recording flow, reference playback, feedback, and local research controls remain the base.

`/consumer` may continue to expose `consumer_v3.html` as an experimental design sandbox, but it is not the public product root.

## 2. ProductScore is unchanged

v7 changes no scoring equation or ProductScore policy.

It does not alter:

- `consumer_four_score_v2`;
- component weights;
- score display transform;
- free-speech score mapping;
- fixed-reference score mapping;
- reliability caps;
- language eligibility;
- v4/v5 candidate promotion status.

The karaoke payload introduced in v6 remains `visualization_only`.

## 3. Public practice modes are now semantically aligned

The public Hugging Face deployment previously started with:

- `reference`;
- `asr_pseudo_reference`;
- `kanade_asr_voice_reference`.

That no longer matched the product contract for free speech.

v7 aligns both the UI and `deploy/start_full_demo.sh` to:

- `reference` → fixed/model practice;
- `transcript_assisted_light` → direct broad-Japanese free speaking.

Pseudo-reference and Kanade experimental modes remain available to local/debug deployments when configured, but are not presented as the consumer definition of free speaking.

## 4. Karaoke replay is integrated into the existing Space

A new replay panel is inserted directly after the existing score panel in `index.html`.

It consumes the additive `karaoke_timeline` payload already produced by v6.

The panel includes:

- timestamp-driven transcript/mora highlighting;
- click-to-seek tokens;
- user recording playback;
- scrubber and playback cursor;
- pause shading;
- user voice-movement curve;
- fixed-practice reference voice-movement overlay when available;
- independent layer toggles.

No token timing is fabricated from text length or utterance duration.

When word timestamps are unavailable, the UI falls back to full-text display rather than invented synchronization.

## 5. One playback surface for recording, upload, and sample

The karaoke player now points to the same audio actually being reviewed.

- microphone recording → generated WAV blob;
- uploaded WAV → uploaded-file object URL;
- built-in sample evaluation → sample recording URL.

This fixes a C-end mismatch in which uploaded audio could be scored but not correctly replayed through the new visualization.

## 6. Microphone capture consistency

The browser microphone request now asks for:

- mono input;
- `echoCancellation: false`;
- `noiseSuppression: false`;
- `autoGainControl: false`.

Browsers may still apply platform-specific behavior, but the product no longer explicitly requests processing that can alter timing, level, or pitch evidence before evaluation.

Physical-device verification remains required.

## 7. Frozen four-dimensional semantics are visible

The public score surface uses the canonical dimensions:

1. 流暢さ
2. 明瞭さ
3. リズム
4. 抑揚

Equivalent localized labels are included for Simplified Chinese, Traditional Chinese, Japanese, and English.

The UI consumes `karaoke_timeline.dimensions` when present and otherwise falls back to the canonical `user_facing.score_dimensions` contract.

It does not use the old public five-score labels as the consumer dimension model.

## 8. Evidence certainty is visible

The score cards distinguish:

- `measured_proxy`;
- `broad_proxy`;
- `neutral_prior`;
- unavailable evidence.

Consumer copy is localized and intentionally avoids research terminology where possible.

This prevents a neutral placeholder such as 70 from visually masquerading as an equally strong direct measurement.

## 9. Pitch visualization no longer overclaims correctness in public

The legacy Space pitch plot contained per-mora target/observed H/L labels and green/red matching.

That diagnostic remains useful for local engineering inspection, but current evidence does not justify presenting it to a learner as strict lexical pitch-accent correctness.

v7 therefore:

- gates H/L target/observed labels behind local/debug mode;
- gates green/red H/L matching behind local/debug mode;
- hides the legacy pitch panel in public mode;
- uses the new speaker-relative karaoke voice-movement visualization as the public pitch experience.

The same principle applies to the legacy speech-region and realtime engineering panels: they remain locally available but are hidden from the public consumer surface.

## 10. Public information hierarchy

The public Space is now centered on:

1. practice prompt / model audio;
2. recording or upload;
3. practice score and the four dimensions;
4. karaoke replay;
5. result reliability;
6. learner feedback.

Engineering-only realtime replay, speech-region diagnostics, legacy pitch diagnostics, and research JSON/CSV controls are retained in the codebase for local development but hidden in `public-mode`.

This is an incremental productization of the existing Space, not a parallel redesign.

## 11. Guard tests

Added:

- `tests/test_hf_space_karaoke_v7.py`

The test suite guards:

- preservation of the established `index.html` Space surface;
- fixed + direct free-speech public modes;
- deployment-script mode consistency;
- canonical four-dimension labels;
- evidence-state rendering;
- real word/mora timing only;
- recording-clock replay synchronization;
- pause/F0 layer availability;
- uploaded/sample audio replay wiring;
- raw-ish microphone constraints;
- no public H/L red/green correctness presentation;
- public hiding of legacy engineering panels;
- duplicate DOM ids;
- browser JavaScript parsing through Node when available.

The lightweight regression workflow now includes the v7 branch and this test file.

## 12. Validation

The latest completed full functional regression before final documentation cleanup reports:

- 159 tests passed;
- 3 existing dependency deprecation warnings from the audio compatibility stack;
- historical reliability-cap audit unchanged:
  - 37 applicable;
  - 6 compatible historical replays;
  - 31 scorer/config drift;
  - 4 censored historical rows;
  - 2 trustworthy counterfactuals;
  - decision remains `none`.

No score-policy drift was introduced by this UI/deployment round.

## 13. Not yet claimed

v7 is not yet published to the Hugging Face Space and is not merged.

This round also does not claim:

- physical iPhone Safari microphone validation;
- physical Android Chrome microphone validation;
- deployed end-to-end latency measurement;
- subjective synchronization testing with diverse real recordings;
- screen-reader accessibility acceptance;
- criterion validity for any score candidate merely because it is now visualized well.

## 14. Next high-value product work

After real-device smoke testing, the next consumer-facing addition should be comparable session replay:

- previous attempt → current attempt;
- same target / same score contract only for numeric deltas;
- side-by-side or toggleable karaoke replay;
- four-dimension change only when comparison context is compatible;
- no fake improvement claim across different targets or score contracts.

That should reuse the established Space UI and the same karaoke timeline rather than create another consumer frontend.
