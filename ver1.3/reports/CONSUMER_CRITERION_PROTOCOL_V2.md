# Consumer Criterion Protocol v2

## Status and purpose

This protocol defines human criterion data for the four **consumer practice constructs**. It does not replace `HUMAN_PRONUNCIATION_CALIBRATION_PROTOCOL.md`.

The existing pronunciation study remains the primary criterion for **pronunciation accuracy**. Its ratings must never be silently renamed `明瞭さ`.

This v2 protocol exists because the product's four visible dimensions are different constructs and need construct-matched validation before stronger `/100` claims are made.

## 1. Keep the constructs separate

For every clip, ratings are stored separately. They must not be averaged into one label during data collection.

### A. 明瞭さ — comprehensibility / ease of understanding

Primary question:

> この発話は、日本語としてどの程度楽に内容を理解できますか。

Suggested 1–7 scale:

- 1: 非常に理解しにくい
- 4: ある程度理解できる
- 7: 非常に楽に理解できる

This is not:

- recording quality;
- native-likeness/accentedness;
- strict phone correctness;
- ASR transcript agreement.

A secondary objective measure may be collected on a subset: listener transcription or keyword recovery. This objective recovery outcome remains separate from the 1–7 comprehensibility rating.

Rationale: recent L2 intelligibility work explicitly motivates perception-based evaluation because ASR and native-likeness measures do not necessarily capture listeners' perceived intelligibility.

### B. 流暢さ — utterance fluency

Primary question:

> この発話は、話す速さ、間の取り方、言い直しなどを含めて、どの程度流暢に聞こえますか。

Suggested 1–7 scale:

- 1: 非常に途切れがち
- 4: ある程度流暢
- 7: 非常に流暢

Raters are allowed to consider speed, breakdown, and repair phenomena. Pronunciation correctness and voice attractiveness should not dominate the rating.

The engineering analysis should separately retain:

- articulation/speech rate;
- silent pauses;
- pause position where available;
- filled pauses;
- repetitions/restarts/self-repair where available.

### C. リズム — timing/rhythm naturalness

Primary question:

> 音や拍の長さ、局所的な速さの変化を含めて、この発話のリズムはどの程度自然ですか。

Suggested 1–7 scale:

- 1: 非常に不自然
- 4: ある程度自然
- 7: 非常に自然

Raters must not be instructed that all morae should have equal duration. The criterion is perceived timing structure/naturalness.

This construct is used to validate current mora-timing evidence and new DTW warp-path rhythm shadows. It is not a direct phone-accuracy label.

### D. 抑揚 — phrase/sentence intonation naturalness

Primary question:

> 文・発話全体の音高の動きや句末の調子は、文脈に対してどの程度自然ですか。

Suggested 1–7 scale:

- 1: 非常に不自然
- 4: ある程度自然
- 7: 非常に自然

Strict lexical pitch-accent correctness is excluded from the headline criterion. When a separate verified lexical-accent task is studied, store that label independently.

## 2. Analyzability remains independent

Before the four ratings, collect:

`analyzable_yes_no`

If the recording cannot be judged as speech because of corruption, missing audio, severe clipping/noise, or another measurement problem, the rater may select `no` and leave construct ratings null.

Unanalyzable must never be converted to low clarity/fluency/rhythm/intonation.

## 3. Fixed reading and spontaneous speech are different strata

Do not pool fixed-reading and spontaneous/free-speaking clips without recording the task stratum.

Required field:

`task_mode = fixed_reading | spontaneous | controlled_dialogue`

Fixed reading may expose target text to the rater when the criterion requires it. For spontaneous comprehensibility/fluency, target text should normally not be shown as if there were a canonical sentence.

The primary report must include within-mode analyses before any pooled result.

## 4. Context for intonation

Intonation ratings require enough pragmatic context to interpret the utterance.

For dialogue clips, the rating UI may provide a minimal blinded context such as the immediately preceding turn or a short task description. Context must not expose speaker identity, model output, current score, or condition labels.

If context is unavailable, store:

`intonation_context_available = false`

and analyze those ratings separately.

## 5. Listener design

Recommended promotion-grade design:

- native Japanese listeners where possible;
- at least five independent ratings per clip;
- anonymized listener IDs;
- listener language/proficiency/training metadata;
- randomized order;
- hidden repeated clips for intra-rater consistency;
- no model score, native/learner label, L1, dataset source, channel condition, or speaker identity visible to raters.

Do not tell raters that a clip contains a known learner error. Do not force native anchors to receive the maximum rating.

## 6. Rating burden

Four full ratings on every long spontaneous clip may create fatigue. Use balanced assignment if necessary, but preserve enough overlap to estimate listener reliability for each construct.

A practical design is:

- analyzability on every presentation;
- clarity + fluency on the broadest panel;
- rhythm + intonation on a balanced overlapping subset;
- pronunciation accuracy retained in the separate frozen study for fixed-reading clips.

This is a sampling decision, not permission to collapse labels.

## 7. Analysis plan

For each construct independently report:

1. analyzability rate;
2. rating distribution;
3. inter-rater reliability;
4. hidden-duplicate intra-rater consistency;
5. speaker and target/task effects;
6. within-task-mode correlations with candidate machine evidence;
7. held-out-speaker validation;
8. held-out-target validation for fixed-reading evidence;
9. channel-condition stability where paired controls exist;
10. calibration error if a `/100` mapping is later fitted.

Candidate model/fusion selection must happen on development data only. The held-out speaker/target partition must not choose layers, weights, thresholds, normalizers, or mappings.

## 8. Construct-specific candidate evidence

### 明瞭さ

Compare, without assuming any is sufficient alone:

- ASR recoverability / transcript stability;
- mapped WavLM evidence after pronunciation study;
- phone-CTC/GOP evidence after Japanese-L2 validation;
- target-independent acoustic/SSL candidates for spontaneous speech.

The existing neutral clarity prior is a product fallback, not a scientific predictor.

### 流暢さ

Compare:

- rate;
- pause/breakdown measures;
- future pause-position features;
- repair features when available.

### リズム

Compare:

- current mora/special-mora timing proxy;
- global duration/tempo features;
- `rhythm_dtw_v1` tempo irregularity;
- interval distortion only after a validated interval classifier is available.

### 抑揚

Compare:

- normalized F0 contour evidence;
- phrase-level F0 movement;
- dialogue-context-conditioned candidates;
- lexical pitch-accent evidence only as a separate diagnostic.

## 9. Mapping policy

No construct receives a new `/100` mapping merely because it correlates with ratings on the same samples used for model selection.

Promotion requires:

- usable listener reliability;
- stable held-out-speaker behavior;
- target hold-out for target-relative methods;
- acceptable channel robustness;
- sensible distributions for ordinary native and learner Japanese;
- no systematic conversion of measurement failure into low learner ability.

If a candidate fails criterion validation, retain it as shadow evidence or remove it. Do not rescue it through cosmetic score transforms.

## 10. Relationship to pronunciation-accuracy protocol

The research stack therefore has two complementary human programs:

1. `HUMAN_PRONUNCIATION_CALIBRATION_PROTOCOL.md`
   - strict pronunciation accuracy;
   - validates phone/phonetic evidence.

2. `CONSUMER_CRITERION_PROTOCOL_V2.md`
   - clarity, fluency, rhythm, intonation;
   - validates the four C-end product constructs.

Their labels may be jointly analyzed as correlated outcomes, but they must never be averaged into a single ground-truth pronunciation score.

## References

- Geng, Saito, Minematsu et al. (Interspeech 2025), perception-based L2 intelligibility assessment work motivating separation of perceived intelligibility from ASR/native-likeness proxies.
- McIntosh, Smit, Saito, Minematsu, and Kamper (2026), *Self-supervised Speech Comparison for L2 Phone, Rhythm, and Intonation Scoring*, arXiv:2607.13721.
- Warner & Arai (2001), *The role of the mora in the timing of spontaneous Japanese speech*, JASA 109(3), 1144–1156.
