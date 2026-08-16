# Free-Speech Four-Dimension Evidence v4

## Status

Branch: `free-speech-four-dimension-evidence-v4`

Base: `fluency-reliability-evolution-v3`

This branch is an evidence and validation evolution. It does **not** change the current learner-facing numeric ProductScore contract.

Frozen product score contract:

- `consumer_four_score_v2`

New evidence schema:

- `consumer_evidence_v3`

The main engineering question is:

> How can free/general Japanese produce useful four-dimensional evidence without pretending that target-independent ASR/F0 features are already validated human judgements?

The branch therefore separates three layers:

1. observed machine evidence;
2. provisional shadow score candidates for A/B and dispersion studies;
3. current ProductScore, which remains unchanged until human criterion validation passes.

---

## 1. Pre-change audit: free conversation was only superficially four-dimensional

A fresh audit of the v3 product path found a major C-end limitation.

In `transcript_assisted_light`:

- fluency had a changing score based mainly on transcript-derived mora rate and silent pause ratio;
- clarity had no target-independent measurement in the consumer component builder and fell back to `70`;
- rhythm expected a `details.fluency.rate_score` that this evaluator did not emit, and therefore fell back to `70`;
- intonation's broad fallback expected `details.tone.pitch_score`, but `transcript_assisted_light` did not emit `details.tone`, and therefore also fell back to `70`.

As a result, a typical free-speaking result could look approximately like:

- 明瞭さ: 70
- リズム: 70
- 流暢さ: changing
- 抑揚: 70

This is numerically stable but has poor C-end usefulness. Three visible dimensions can look like real measurements even though they are neutral product priors.

The correct response is **not** to replace those priors immediately with arbitrary moving numbers. The first requirement is to collect construct-matched evidence and make missingness/proxy status explicit.

---

## 2. Scientific construct boundaries retained

### 2.1 Clarity

The product criterion is broad clarity/comprehensibility:

> how easily the utterance can be understood as Japanese.

It is not:

- recording quality;
- native-likeness;
- strict phone correctness.

ASR recoverability may be useful evidence because poorly intelligible speech often reduces recognizer confidence or recognition accuracy. However, recognizer behavior also depends on:

- language-model predictability;
- model family;
- lexical frequency;
- decoding strategy;
- microphone/channel conditions.

Therefore v4 calls the new construct:

`machine_recoverability_proxy_not_human_comprehensibility`

and does not directly map it into learner-facing clarity.

### 2.2 Rhythm

Japanese rhythm is not treated as a rule that every mora must have identical duration.

Without a trusted target/reference alignment, v4 uses only a weak local-tempo descriptor derived from:

- ASR word timestamps;
- Japanese mora count inside each recognised word;
- duration per recognised mora at the word level;
- dispersion of log seconds-per-mora across words.

This is labelled:

`asr_word_level_local_tempo_structure_not_mora_isochrony`

It can reveal local timing irregularity, but expressive slowing, focus, phrase boundaries, vowel devoicing and ASR boundary error can also produce local variation.

### 2.3 Fluency

The existing v3 spontaneous-fluency shadow remains the main construct-separated source:

- speed;
- breakdown;
- repair.

v4 does not discard it or collapse fluency back into only speaking rate.

The current product fluency number remains unchanged in this branch.

### 2.4 Intonation

Without a trusted target or dialog-context interpretation, global F0 movement cannot prove that an utterance has context-appropriate Japanese intonation.

v4 therefore exposes descriptive evidence only:

- robust p90-p10 F0 range in semitones;
- median absolute local F0 step;
- terminal F0 slope when enough frames exist.

The construct is labelled:

`target_independent_f0_movement_not_contextual_intonation_correctness`

Lexical pitch accent remains outside the top-level construct and requires stronger target/alignment evidence.

F0 extraction failure is explicitly represented as **unavailable evidence**, not a low/flat intonation judgement.

---

## 3. No extra online model pass

The C-end latency constraint was reviewed before implementation.

v4 does not add:

- a second Whisper pass;
- ASR ensemble inference;
- online WavLM inference;
- online CTC/GOP inference;
- TTS generation;
- DTW reference comparison.

The free-speaking path already runs language-aware ASR and F0 extraction. v4 only preserves additional diagnostics already produced during that ASR call and derives lightweight statistics from existing outputs.

The faster-whisper wrapper now preserves per-segment:

- `avg_logprob`;
- `no_speech_prob`;
- `compression_ratio`;
- segment start/end.

Existing word timestamps already contain word probability.

Therefore the intended runtime cost is telemetry/statistics overhead rather than another acoustic-model inference pass.

---

## 4. New evidence schema

New module:

`src/jp_speech_eval/free_speech_evidence.py`

Top-level schema:

`free_speech_dimension_evidence_v1`

Stored under:

`details.shadow.free_speech_dimension_evidence`

### Clarity evidence

Stores:

- word probability distribution;
- segment average-log-probability distribution;
- segment no-speech probability;
- segment compression ratio;
- word-timestamp coverage;
- language probability, explicitly marked routing-only;
- a simple ASR recoverability index.

The recoverability index is deliberately a machine summary, not a calibrated human probability.

### Rhythm evidence

Stores:

- usable ASR word count;
- median seconds per mora at word level;
- median absolute deviation of log seconds per mora;
- p90-p10 spread of log seconds per mora;
- a small sample of word-level timing calculations for auditability.

### Fluency evidence

Links the existing:

`spontaneous_fluency_evidence_v1`

without remapping it.

### Intonation evidence

Stores:

- robust F0 range in semitones;
- local step descriptor;
- terminal slope;
- explicit unavailable state when F0 evidence is insufficient.

Every evidence block remains:

- `product_calibrated = false`;
- `user_facing = false`;
- not a formal educational measurement.

---

## 5. Shadow candidate surface

For product engineering, raw evidence alone is not enough to answer whether a future four-score UI would have useful score dispersion.

v4 therefore adds:

`details.shadow.free_speech_candidate_surface`

Policy:

`free_speech_candidate_surface_v1_shadow`

It creates provisional component candidates for:

- clarity;
- mora timing/rhythm;
- fluency;
- intonation.

These are **not** promoted scores.

### Conservative shrinkage

The unvalidated clarity/rhythm/intonation mappings are strongly shrunk toward the neutral anchor `70`.

This prevents weak proxies from immediately behaving like fully trusted 0-100 judgements.

Examples:

- clarity uses ASR recoverability only after strong shrinkage;
- rhythm converts local-tempo irregularity to a provisional monotonic candidate, then shrinks it;
- intonation uses a broad, forgiving robust-F0-range plateau, then shrinks it;
- fluency reuses the current product proxy only for comparison.

The shadow payload explicitly states:

- `user_facing = false`;
- `product_calibrated = false`;
- `product_score_changed = false`;
- human criterion + held acceptance are required before promotion.

The specific formulas are hypotheses for validation, not scientific conclusions.

---

## 6. C-end evidence-state metadata

A C-end score display should not present all numeric values as equally measured.

The consumer dimension payload now distinguishes:

- `unavailable`;
- `neutral_prior`;
- `broad_proxy`;
- `measured_proxy`.

It also exposes a `numeric_semantics` field.

For the current free-speaking ProductScore, the existing neutral values remain numerically unchanged, but the API can now identify that a value such as clarity `70` may be:

`neutral_anchor_not_direct_measurement`

This creates a safer path for future UI work, for example:

- normal solid score for validated/measured dimensions;
- softer or approximate presentation for rough proxy dimensions;
- explicit neutral/insufficient-evidence treatment rather than pretending every 70 is equally precise.

No visual design change is forced by this branch; it only makes the semantics available to the product layer.

---

## 7. Listener-to-evidence validation pipeline

The repository already had:

- four-construct listener rating schema;
- rating validator;
- generic machine-evidence analyzer.

However, the listener schema is a wide four-column format while the analyzer expects long-form:

`sample_id / criterion / human_rating`

v4 closes that gap.

New scripts:

- `scripts/normalize_consumer_ratings.py`
- `scripts/export_free_speech_v4_evidence.py`

The end-to-end path is now:

1. collect ratings using `consumer_rating_schema_v2.json`;
2. validate with `validate_consumer_ratings.py`;
3. normalize each assigned construct separately with `normalize_consumer_ratings.py`;
4. export already-stored v4 machine evidence with `export_free_speech_v4_evidence.py`;
5. analyze with `analyze_research_evidence.py`.

The normalizer never averages the four human constructs together.

The evidence exporter performs no model inference.

---

## 8. Frozen promotion protocol

New protocol:

`data/research_eval/free_speech_v4_promotion_protocol.json`

The protocol is frozen before criterion results are available.

Main held gate design:

- at least 60 held analyzable clips;
- at least 10 speakers;
- learner and native speakers both represented;
- target 5 ratings per analyzable clip;
- speaker-disjoint development vs held evaluation;
- spontaneous and controlled-dialogue tasks;
- channel-paired controls;
- English/Mandarin/silence/noise as eligibility/failure controls, not Japanese ability scores.

The minimum promotion checks include:

- construct-matched human association;
- candidate availability;
- speaker/task stability;
- score dispersion;
- paired-channel sensitivity;
- correct no-score/retry behavior for non-Japanese/unusable input;
- no F0-failure-as-bad-intonation behavior.

A shadow candidate that fails these checks remains shadow evidence even if it produces visually attractive score variation.

---

## 9. Why channel-paired controls are required

A target-independent system is particularly vulnerable to learning recording conditions.

For example, ASR word probability may drop because of:

- microphone distance;
- codec artifacts;
- background noise;
- low input level.

If human comprehensibility remains similar while the machine clarity candidate changes dramatically, the candidate is not sufficiently construct-specific for ProductScore.

Therefore channel sensitivity is treated as a promotion gate, not merely a diagnostic appendix.

This also preserves the product rule:

> recording analyzability may lower confidence/detail, but should not silently become learner pronunciation ability.

---

## 10. Current user-visible behavior intentionally remains unchanged

For valid Japanese free speech, this branch does **not** change:

- ProductScore weights;
- ProductScore display transform;
- current fluency thresholds;
- current clarity/rhythm/intonation numeric priors;
- no-score language eligibility behavior;
- feedback-detail gates.

Therefore this branch is safe to evaluate without creating a new generation of user history scores.

The score contract remains:

`consumer_four_score_v2`

Only the evidence schema becomes:

`consumer_evidence_v3`

A future product promotion must create a new score-contract version.

---

## 11. Remaining scientific limitations

### Clarity

ASR probability is decoder-dependent and can reward predictable wording. It needs direct validation against human clarity/comprehensibility.

A stronger future target-independent clarity system may combine:

- ASR recoverability/stability;
- phone-level confidence independent of the ASR language model;
- perception-oriented learned features.

But adding more models is not justified before testing whether the zero-extra-inference baseline already provides useful information.

### Rhythm

ASR word timing is not phone/mora forced alignment. The current measure is therefore intentionally weak.

For fixed reading/Deep Review, the stronger research direction remains SSL-DTW warp-path rhythm evidence with real reference panels.

For free speech, rhythm eventually needs better target-independent modelling or human-trained naturalness prediction.

### Fluency

The v3 repair evidence remains low confidence because general ASR may normalize fillers, repetitions and false starts.

### Intonation

Global F0 movement has no semantic/dialog-context model in this branch.

A later conversation system can provide context descriptors such as question/continuation/finality/focus, while the acoustic evaluator judges whether observed F0 movement is appropriate for that context.

That context layer should still remain separate from strict lexical pitch-accent diagnostics.

---

## 12. Literature basis for the engineering boundaries

Relevant sources reviewed for this evolution include:

- Geng, Saito & Minematsu, Interspeech 2025, perception-based L2 intelligibility assessment using native-rater shadowing and sequence-to-sequence voice conversion. This work motivates separating human-perceived intelligibility from conventional ASR/native-likeness measures.
- McIntosh et al., 2026, *Self-supervised Speech Comparison for L2 Phone, Rhythm, and Intonation Scoring*. This supports SSL comparison/DTW as a strong fixed-reference research direction and motivates local-tempo/warp-path rhythm evidence rather than simple equal-mora assumptions.
- Tsurutani & Ishihara, research on timing and pitch effects on perceived naturalness in L2 Japanese. This supports treating timing and F0 as distinct contributors rather than collapsing prosody into one number.
- Japanese spontaneous-speech rhythm work showing that spontaneous Japanese timing should not be reduced to a simplistic perfectly mora-timed rule.
- faster-whisper implementation/API, which exposes word probability and segment-level decoding diagnostics used here without another inference pass.

These sources motivate the evidence choices. They do **not** validate the v4 heuristic candidate mappings to `/100`; that is why those mappings remain shadow-only.

---

## 13. Next gate

The next useful operation is not another scoring formula.

It is to run the current free-speaking evaluator on a speaker-diverse, channel-controlled Japanese set, collect construct-matched listener ratings, export v4 evidence, and evaluate:

- which candidate actually tracks clarity;
- whether word-timing rhythm adds information beyond overall speaking rate;
- whether current fluency is too coarse;
- whether target-independent F0 range provides any stable ordering for perceived intonation;
- whether channel effects dominate any candidate;
- whether the candidate surface has enough C-end score dispersion to be worth promoting.

Only then should a future branch decide whether any of the three neutral priors can be replaced in ProductScore.
