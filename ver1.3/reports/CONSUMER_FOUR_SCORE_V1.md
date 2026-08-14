# Consumer Four-Score Policy v1

## Product decision

For the C-end demo, once an utterance has passed the product's basic Japanese/recording eligibility checks, the learner should always receive four numeric practice scores:

1. `流暢さ` — fluency
2. `明瞭さ` — clarity
3. `リズム` — rhythm
4. `抑揚` — intonation

A missing high-quality signal should degrade the evidence tier and confidence, not silently turn into `0` and not make a dimension disappear. True no-score remains appropriate for clearly non-Japanese, nonspeech, or unusable recordings.

These are **consumer practice scores**, not psychometric or certified language-test scores. Every dimension emitted by `consumer_dimension_policy.py` carries provenance, `confidence`, `evidence_tier`, and `product_calibrated=false` so that the product can remain continuous while research validation proceeds.

## Why these four dimensions

### 1. Fluency / 流暢さ

L2 utterance-fluency research commonly separates speed, breakdown, and repair fluency. The current runtime has direct evidence for speed and breakdown (pausing), but not yet a strong repair detector. Therefore the consumer score combines speaking-rate and pause evidence rather than using pause score alone.

Current consumer mapping:

- 48% `rate_score`
- 52% `pause_score`

This is a product mapping, not a literature-derived psychometric weighting. The scientifically motivated part is the inclusion of both speed and breakdown evidence. Repair fluency remains a future task.

### 2. Clarity / 明瞭さ

`明瞭さ` is intentionally broader than strict segmental pronunciation accuracy. It must not be filled by recording quality, and the legacy `pronunciation_score` is not used because that score is mainly a mora-timing/special-mora-duration proxy.

Evidence hierarchy:

1. explicitly mapped SSL/pronunciation evidence, when available;
2. ASR kana agreement with the known target plus acoustic support;
3. reference-relative acoustic content similarity;
4. MFCC-DTW reference similarity;
5. a neutral low-confidence product prior only when no independent clarity evidence is available.

ASR agreement is interpreted as **machine-intelligibility evidence**, not human intelligibility, comprehensibility, accentedness, or phone correctness. MFCC-DTW is an acoustic/reference similarity cue, not a phone-correctness measure.

The desired future backbone remains multi-reference SSL/WavLM. Existing v3.2 experiments show useful native < learner < wrong-target ordering overall, but also substantial speaker/channel sensitivity and a target with native/learner overlap, so raw WavLM distance is not yet treated as a calibrated `/100` pronunciation score.

### 3. Rhythm / リズム

Japanese rhythm is treated as timing structure, not as a requirement that all morae have identical duration. The primary tier combines:

- local mora/special-mora timing evidence; and
- a broad reference-relative duration/tempo term.

When local alignment fails, the product falls back to broad speaking-rate plus global duration evidence rather than removing the rhythm score. This is deliberately less specific and receives lower confidence.

Current primary product blend:

- 78% local mora/special-mora timing proxy
- 22% global duration match

Fallback blend:

- 62% speaking-rate score
- 38% global duration match

These coefficients are product heuristics. They should later be validated against human rhythm ratings and/or the DTW-warp rhythm methods in McIntosh et al. (2026).

### 4. Intonation / 抑揚

`抑揚` means phrase/sentence F0 movement. It is deliberately separate from strict lexical pitch-accent correctness.

Evidence hierarchy:

1. existing speaker-normalized, reference-relative mora F0 contour score;
2. partial paired F0 contour comparison when only part of the mora contour is available;
3. broad pitch-movement/range naturalness when reference-relative contour extraction is too sparse;
4. a neutral low-confidence product prior only when F0 is genuinely unavailable.

For confirmed free speech, a generated TTS reference can support a broad contour comparison, but its extremes are shrunk toward a neutral practice anchor and confidence is reduced. TTS prosody is not treated as lexical-accent ground truth.

## Scientific anchors

### Phone / rhythm / intonation comparison

McIntosh, Smit, Saito, Minematsu, and Kamper (2026), *Self-supervised Speech Comparison for L2 Phone, Rhythm, and Intonation Scoring*, arXiv:2607.13721.

The paper evaluates English and Japanese L2 speech and supports the general architecture of reference comparison with WavLM + DTW: phonetic comparison is strong, DTW-path warping can represent rhythm, and intonation can be modeled with prosodic residual/F0/intensity evidence. Intonation is less robust than phone/rhythm scoring, which supports confidence degradation rather than overclaiming.

### Japanese rhythm is not equal-mora timing

Warner, N., & Arai, T. (2001). *The role of the mora in the timing of spontaneous Japanese speech*. Journal of the Acoustical Society of America, 109(3), 1144–1156. DOI: 10.1121/1.1344156.

Their spontaneous-speech results show that mora count alone is a weaker predictor of timing than careful-speech descriptions suggest; syllable structure and final lengthening matter. Therefore the consumer rhythm score should not reward mechanical equal-mora timing as the sole target.

### L2 fluency

The established utterance-fluency framework separates speed, breakdown, and repair fluency. The present implementation covers speed and breakdown and explicitly records the repair gap. A future spontaneous-speech version should add filled pauses, repetitions, false starts, and self-repair before claiming full fluency coverage.

### Clarity and comprehensibility

Automated L2 comprehensibility work shows that temporal, phonological, and prosodic measures can predict listener judgments, but an ASR transcript is not itself a human comprehensibility rating. The current clarity score is therefore deliberately named a broad C-end practice proxy in metadata and remains `product_calibrated=false` until listener calibration is available.

## Product semantics

### Always-score rule

After Japanese eligibility passes:

```text
high-quality evidence -> numeric score + medium/high confidence
partial evidence      -> numeric score + low/medium confidence
weak evidence         -> numeric score + low confidence / fallback tier
```

Before Japanese eligibility passes:

```text
non-Japanese / nonspeech / unusable recording -> no score
```

A score of `0` must mean a genuine numeric result near zero; it must never mean missing evidence.

## Consumer total

The Consumer preview currently uses the equal-weight mean of the four displayed practice dimensions so the headline score and the four cards describe one coherent surface. The previous three-factor display score is preserved as `legacy_display_score` for audit.

This equal weighting is a **product choice**, not a validated educational-measurement weighting. It should be revisited after dimension-level human calibration and distribution analysis.

## Next validation priorities

1. Reduce use of low-confidence priors by adding a global/frame-level F0 fallback independent of mora alignment.
2. Add ASR-confirmation agreement evidence to free-speech clarity so that user-confirmed text can be compared with the original unconstrained Japanese ASR hypothesis.
3. Continue WavLM multi-reference research with channel normalization; do not promote raw SSL distance to a dominant `/100` score yet.
4. Add repair-fluency detection for spontaneous speech.
5. Evaluate four-score distributions on native, learner, wrong-target Japanese, speed, pause, noise/RIR/codec, and special-mora panels.
6. Keep confidence/evidence tier in telemetry even if the C-end UI only shows a compact learner-friendly summary.
