# Baseline Evolution v1

## Status

Branch: `baseline-evolution-v1`

This patch is a baseline hardening step, not a ProductScore remapping. It fixes two measurement-chain problems, adds explicit reference-boundary provenance, and expands Japanese phone-CTC shadow diagnostics. It deliberately does **not** promote GOP/SSL evidence to a user-facing `/100` score.

The product policy remains:

- ordinary, plausible Japanese should receive useful practice feedback whenever the recording is analyzable;
- recording/channel quality is a reliability input, not pronunciation correctness;
- content verification, pronunciation clarity, Japanese timing, fluency, and intonation are separate constructs;
- weak evidence should reduce confidence/detail rather than invent a precise local error;
- research evidence is promoted only after Japanese-L2 criterion validation;
- `/100` remains a C-end practice heuristic until human calibration supports a stronger claim.

## 1. P0 repair: recording-domain audio is no longer destroyed by analysis normalization

### Previous problem

`load_audio()` peak-normalized the waveform before `assess_recording_quality()` measured absolute speech level and clipping. Peak normalization is useful for analysis robustness, but it destroys the original amplitude scale needed by those channel diagnostics.

### New contract

`load_audio_views()` now exposes two domains:

1. amplitude-preserved native-rate mono decode for recording/channel diagnostics;
2. resampled, peak-normalized analysis audio for VAD, ASR, alignment, F0, SSL, and phone-CTC processing.

The legacy `load_audio()` API remains compatible. Its normalized ndarray carries the raw decode as non-scoring provenance, allowing existing evaluator code to recover the original recording domain without a broad API rewrite.

No recording-quality threshold was retuned in this patch. Separating the signal domains is a measurement repair; threshold calibration is a later empirical task.

Regression guards are in `tests/test_audio_input_domains.py`.

## 2. P0 repair: reference boundaries now have explicit provenance

### Previous problem

Reference mora boundaries were commonly initialized by equal division of reference duration. Cached MFCC-DTW could then map learner timing from those pseudo-boundaries. This is acceptable as coarse fallback evidence but not as a precise phone/mora alignment source for local pronunciation claims.

### New cache metadata

`SentenceMeta` now records:

- `ref_boundary_method`
- `ref_boundary_confidence`
- `ref_boundary_tier`
- `ref_boundary_source`

Old caches remain readable. Equal-mora caches are explicitly classified as low-precision `equal_fallback` instead of being silently treated as verified timing.

### Offline verified-alignment import

`prepare_cache.py` now supports an exact reference WAV plus a matching phone alignment:

```bash
python scripts/prepare_cache.py \
  --text "ラーメンをください" \
  --reference-wav path/to/reference.wav \
  --reference-alignment path/to/reference.TextGrid \
  --reference-alignment-method mfa \
  --reference-source human_reference \
  --out cache/ramen_verified
```

Simple `start end phone` `.lab` files are also supported. The imported phone sequence is conservatively mapped to the target mora sequence. Mapping must succeed one-to-one; otherwise cache preparation fails closed rather than fabricating precise boundaries.

MFA/Narabas remain offline research/cache tools. They are not silent runtime dependencies of the C-end product.

Regression guards are in `tests/test_reference_boundary_provenance.py`.

## 3. Survey-driven phone-CTC shadow evolution

Recent pronunciation-assessment work reports that raw-logit competition can retain useful discriminative information that softmax posterior GOP may lose through saturation. The current project already has Japanese-specific competitor restrictions, CTC sequence/deletion evidence, posterior margin, entropy, and model-peakiness diagnostics. Therefore the appropriate evolution is incremental rather than replacing the backbone.

`ctc_posterior_diagnostics_v2` adds descriptive, shift-invariant logit evidence:

- full-vocabulary top-1 logit margin mean/p05;
- phone-only top-1 logit margin mean/p05;
- blank-minus-best-phone logit margin mean.

These values remain:

- `product_calibrated=false`
- `score_mapped=false`
- shadow/research evidence only.

They must be compared against Japanese-L2 human criterion labels before any score mapping or user-facing diagnosis.

Primary references:

- Parikh et al., Interspeech 2025, *Evaluating Logit-Based GOP Scores for Mispronunciation Detection*, DOI 10.21437/Interspeech.2025-1012.
- Parikh et al., Interspeech 2025, *Enhancing GOP in CTC-Based Mispronunciation Detection with Phonological Knowledge*, DOI 10.21437/Interspeech.2025-829.
- Dong et al., SLaTE 2025, *Automatic Pronunciation Assessment for L2 English by Incorporating Suprasegmental Features and Weighted Loss Function*, DOI 10.21437/SLaTE.2025-5.

The third result reinforces the existing project architecture: fluency/prosody should retain suprasegmental evidence instead of being collapsed into a single GOP construct.

## 4. Japanese-specific recognizer research candidate

Kubo, Sproat, Taguchi, and Jones, *Building Tailored Speech Recognizers for Japanese Speaking Assessment* (arXiv:2509.20655; listed for Interspeech 2026), build Japanese assessment-oriented recognizers that output phonemic labels with accent markers. Their reported CSJ-core average mora-label error improves from 12.3% to 7.1% with their proposed approach.

This is highly relevant to the future clarity/pitch research path because it is more task-specific than a generic multilingual recognizer. No directly deployable public model checkpoint was identified during this survey, so it is recorded as a reproduction/comparison candidate rather than a runtime dependency.

## 5. Forced-alignment survey conclusion

MFA 3.x remains a strong offline benchmark/cache candidate. The 2026 MFA alignment paper reports state-of-the-art or near-state-of-the-art performance across evaluated English/Japanese/Korean benchmarks, with mean boundary errors below 15 ms. MFA documentation also explicitly distinguishes comparison against hand-corrected reference alignments from mere agreement between two automatic aligners.

Therefore the project should use a hierarchy of evidence rather than treating one aligner as ground truth:

1. human/hand-checked phone or mora boundary;
2. validated offline forced alignment tied to the exact reference WAV;
3. engine-provided duration/timing metadata after validation;
4. coarse acoustic DTW support;
5. equal-mora fallback.

Local special-mora or phone-correctness feedback must not be promoted solely from levels 4-5.

Primary reference:

- McAuliffe et al., 2026, *Montreal Forced Aligner and the state of speech-to-text alignment in 2026*, arXiv:2606.18466.

## 6. What is intentionally unchanged

This patch does not change:

- the four C-end component weights;
- ProductScore centering;
- content-match ASR thresholds or selected `faster-whisper-small` policy;
- mora-timing thresholds;
- special-mora rollout thresholds;
- F0/pitch thresholds;
- WavLM-to-score mapping (there is still none);
- phone-CTC/GOP-to-score mapping (there is still none).

The existing content benchmark already found `always faster-whisper small` to have 0/50 hard wrong-target false verifications and 2/30 correct-target false mismatches on its fixed panel. That result should be preserved until a new held-out acceptance set justifies a change.

## 7. Remaining product-semantic issue

The legacy evaluator still contains raw-score caps triggered by alignment/F0 evidence quality. These raw fields predate the semantic four-component C-end layer. Reliability failure should generally reduce confidence and local-detail eligibility rather than be interpreted as pronunciation incorrectness.

Do **not** delete those caps blindly in this patch: they interact with legacy reports/tests, and old equal-boundary caches are still widespread. The safe order is:

1. migrate the reference bank to explicit boundary provenance;
2. rerun fixed/broad/negative-control acceptance;
3. compare C-end four-component scores before/after removing reliability-driven raw construct caps;
4. keep the change only if ordinary Japanese score availability and useful score differentiation are preserved.

## 8. Next release gates

### Gate A — reference-bank migration

For every production/demo fixed target, record the boundary tier. Prefer exact human/native reference WAVs plus aligned phone timing. Quantify how many targets remain on equal fallback.

### Gate B — acceptance closure

Rerun the same candidate without threshold tuning on:

- native Japanese;
- learner Japanese;
- same-duration wrong Japanese;
- real English/Mandarin speech where available;
- silence/noise/unusable recordings;
- short but valid Japanese;
- channel/noise variants.

Required product behavior: valid/plausible Japanese retains a score when analyzable; invalid/non-Japanese/unusable input does not receive a normal Japanese practice score; weak alignment removes local certainty rather than fabricating a phone error.

### Gate C — human criterion pilot

Execute the already frozen `HUMAN_PRONUNCIATION_CALIBRATION_PROTOCOL.md`. The current 14 learner clips from two speakers remain a pilot only; recruit the planned additional learner speakers before a promotion-grade mapping study.

For the same blinded clips, export and compare:

- WavLM same-target multi-reference evidence;
- Beatrice phone-CTC posterior evidence;
- Beatrice logit-margin evidence;
- dual-CTC equivalents where practical;
- restricted/alignment-free CTC sequence evidence;
- ASR recoverability/content evidence;
- existing timing/fluency/prosody evidence.

Model/fusion selection must use development data and be evaluated with held-out target and held-out speaker splits.

### Gate D — first production promotion

Only after Gate C should one of CTC, WavLM, a hybrid, or a reproduced Japanese assessment-specific recognizer replace the low-confidence clarity prior or receive a `/100` mapping.

## 9. CI

The lightweight baseline-evolution workflow covers the new audio-domain, reference-provenance, CTC-diagnostic, alignment, content, and user-score guards on Python 3.11. Heavy model downloads remain excluded from this fast gate; they keep their separate research/preflight workflows.
