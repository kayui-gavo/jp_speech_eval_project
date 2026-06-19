# Pitch Negative Control Protocol

- generated_at: 2026-06-19 JST
- scope: diagnostic protocol only
- calibration_status: inactive

## Current Feasibility

The repo can run repeatable semi-audio pitch controls today: real JVS audio and lab timing provide mora-level F0, then counterfactuals replace the F0 contour while keeping content, mora count, and timing fixed.

The repo does not currently have a production-ready waveform-level F0 manipulation pipeline. A waveform-level protocol would need either a vocoder/resynthesis path or an external pitch-shifting tool that preserves segment timing and content without introducing artifacts.

## Control Construction

| control | construction | expected result | current feasibility |
|---|---|---|---|
| native reference | human/native reference F0 from matched sentence | highest among controlled contours | available in semi-audio JVS audit |
| flat pitch | replace valid F0 with per-sentence median | clearly lower than native | available |
| random/shuffled pitch | shuffle valid mora F0 values with fixed seed | clearly lower than native | available |
| low F0 coverage | retain too few voiced mora values | unavailable/insufficient evidence | available |
| wrong accent drop | alter drop transition around accent nucleus | lower than native, especially on drop-specific component | partially available; separation weak |

## Wrong-Drop Diagnosis

The current wrong-drop counterfactual changes local F0 around accent-drop candidates, but score separation is weak because accent-drop agreement is logged and used for feedback, not an independent weighted score component. The aggregate mainly sees contour correlation and transition agreement; changing one drop often leaves the overall contour close to native.

## Suitable Wrong-Drop Items

Good candidates:

- Sentences with at least one clear accent phrase drop.
- Sentences where OpenJTalk/JVS lab timing gives enough voiced mora around the drop.
- Sentences where changing the drop window does not also change content, duration, or alignment.

Bad candidates:

- Heiban-like sentences without a clear drop.
- Very short sentences with too few voiced mora.
- Sentences where the target drop source is only heuristic and not checked.
- Cases where the altered F0 does not overlap the intended drop mora.

## Next Technical Step

Do not change weights yet. First improve target extraction and drop-window diagnostics:

1. Identify phrase-level accent-drop windows from verified targets.
2. Confirm the wrong-drop manipulation actually changes those windows.
3. Add a separate diagnostic accent-drop subscore.
4. Validate with real or waveform-level wrong-accent audio.

Only after those checks should calibration or weighting be considered.

