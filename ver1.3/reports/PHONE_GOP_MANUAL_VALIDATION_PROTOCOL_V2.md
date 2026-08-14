# Japanese phone-GOP manual validation protocol v2

Date: 2026-08-15
Status: current operational version; supersedes `PHONE_GOP_MANUAL_VALIDATION_PROTOCOL_V1.md`

## Current assets

- Recording manifest: `data/phone_gop_manual_validation_manifest_v2.csv`
- Human inspection template: `data/phone_gop_manual_inspection_template_v1.csv`
- Practical recording sheet: `reports/PHONE_GOP_RECORDING_SHEET_V1.md`
- Technical GOP survey: `reports/JAPANESE_GOP_TECH_SURVEY_V1.md`

V2 preserves the scientific design of V1 and corrects the operational manifest identifiers. Use V2 assets for all new recordings.

## Frozen principles

1. The target text always remains the **canonical Japanese target**, even when the speaker deliberately realizes a different phone/sequence. Do not feed the intentional error text to GOP as the target.
2. Every core item has two natural controls (`N1`, `N2`) before the intentional-error take. This gives a within-speaker stability reference.
3. `intended_error` is not ground truth. A listener must confirm whether the error was actually realized.
4. Human inspection is two-pass: blind listening first, intended manipulation revealed second.
5. Unanalyzable audio is null, never severity 0.
6. Initial evaluation uses ordering/localization, not invented numeric thresholds.
7. Special morae require timing evidence in addition to phone evidence.
8. Artificial errors are an engineering stress test; they do not replace naturally occurring learner errors.
9. The four C-end dimensions must be tested for leakage: pause→fluency, timing→rhythm, phone errors→clarity, F0 manipulations→intonation.
10. Phone-GOP stays shadow-only until real learner validation.

## Quick pilot

Record the 38 items in `PHONE_GOP_RECORDING_SHEET_V1.md` in one controlled session. Do not randomize the *recording* order: normal takes should precede intentional-error instructions so the speaker is not primed to distort the clean controls.

After recording, randomize the files for the *listening* pass.

For each clip label:

- analyzable yes/no;
- what was actually heard;
- target recovered yes/no;
- audible error type;
- error location;
- severity 0–3;
- listener confidence;
- after reveal: realized-as-intended yes/no/uncertain.

Then compare the label with phone-GOP output:

- target mean/max logit margin;
- posterior GOP margin;
- best competitor;
- entropy;
- target rank within word;
- support-frame count/duration;
- adjacent-phone damage.

## Initial qualitative gates

Do not set fixed thresholds before pilot distributions exist. Look for:

- N1 and N2 closer to each other than to a successful strong error;
- intended error phone becoming clearly weaker in the E take;
- categorical substitution competitor matching the intended substitute or a plausible phonetic neighbor;
- limited neighbor leakage;
- pause-only and pitch-only controls not causing a phone-level clarity collapse;
- natural high-vowel devoicing not producing a severe false alarm;
- channel perturbations producing less target-phone damage than a successful categorical mispronunciation.

Only after this should numeric deltas be defined from clean-repeat variance.

## Next stages

- Stage A: one-speaker controlled 38-clip quick battery.
- Stage B: reduced battery on 3–5 speakers, preferably including at least one native Japanese speaker.
- Stage C: naturally occurring learner errors with manual phone-level labels.
- Stage D: fuse validated phone evidence with WavLM and ASR intelligibility into the C-end clarity score.

The detailed rationale and literature background remain in V1 and `JAPANESE_GOP_TECH_SURVEY_V1.md`; V2 is the operational source of truth for recording/inspection.
