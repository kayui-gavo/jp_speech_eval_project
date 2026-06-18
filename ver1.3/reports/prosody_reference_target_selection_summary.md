# Prosody Reference Target Selection Summary

- generated_at: 2026-06-19 JST
- branch context: `rollback/stable-packaged-demo-6-2`
- implementation scope: target source selection, sidecar cache builder, metadata/gate tests

## Implemented Files

- `src/jp_speech_eval/prosody_reference_cache.py`
  - Builds and reads `<cache>.prosody_ref.json`.
  - Selects the best available pitch/prosody reference target.
  - Marks target reliability as `reliable`, `weak`, `heuristic`, or `unreliable`.

- `scripts/build_prosody_reference_cache.py`
  - CLI builder for sidecar reference F0 cache.
  - Supports `--verified-reference` for explicitly checked human/native/teacher reference audio.
  - Supports `--dry-run` for provenance inspection without writing files.

- `src/jp_speech_eval/evaluator.py`
  - Uses the target selector before calling `score_prosody`.
  - Passes explicit `reference_f0_target_source` when a reference contour is used.
  - Adds `pitch_target_source`, `pitch_target_reliability`, and `prosody_reference_target` to details.

- `src/jp_speech_eval/scoring.py`
  - Adds an optional source label for reference-F0 targets.
  - Does not change the scoring formula.

- `src/jp_speech_eval/scoring_policy.py`
  - Blocks strong pitch feedback when target reliability is `weak`, `heuristic`, `unreliable`, or `invalid`.
  - Preserves the existing `asr_confirmed_weak_reference` practice-mode exception.

- `src/jp_speech_eval/target_specs.py`
  - Recognizes `reference_audio_f0_cache` and trusted runtime reference as human-checked target source classes.

- `src/jp_speech_eval/feedback_renderer.py`
  - Adds pitch target reliability to debug payload.
  - Does not expose raw/debug pitch as formal UI score when gates block it.

## Verified Behaviors

| case | expected behavior | result |
|---|---|---|
| reliable sidecar cache exists | use `reference_audio_f0_cache` | covered by unit + evaluator smoke |
| reliable sidecar exists over TTS runtime fallback | sidecar wins | covered |
| missing sidecar but trusted native reference | use `reference_audio_f0_runtime` | covered |
| TTS reference without verified sidecar | mark `tts_reference_weak` | covered |
| OpenJTalk-only target | pitch feedback blocked as heuristic | covered |
| low reference F0 coverage | sidecar unreliable | covered |
| content mismatch despite reliable reference | formal score/pitch hidden | covered |
| fallback alignment despite reliable reference | formal score/pitch hidden | covered |
| tone score | not restored to core four dimensions | covered by existing tests |

## Cache Builder Example

Inspect an existing cache without writing:

```bash
cd ver1.3
../.venv/bin/python scripts/build_prosody_reference_cache.py --cache cache/ramen_kudasai --dry-run
```

Build a sidecar for a verified human/native reference:

```bash
cd ver1.3
../.venv/bin/python scripts/build_prosody_reference_cache.py \
  --cache cache/ramen_kudasai \
  --verified-reference
```

The current `cache/ramen_kudasai` dry-run is not reliable by default because its provenance is `pyopenjtalk_tts_pseudo_reference` and timing is equal-mora approximate. This is intentional.

## Cross-Speaker Sanity

Not performed in this implementation step. JVS is present locally, but this change only establishes the cache/selection path and target provenance gates. Cross-speaker evaluation should be a separate audit using same-sentence native reference caches so that alignment/reference quality can be inspected without mixing it into this code change.

## Calibration Readiness

Still not ready. This step fixes target provenance and gives the evaluator a reliable-reference path. It does not calibrate scores, prove cross-speaker stability, or validate wrong-accent audio counterfactuals at waveform level.

