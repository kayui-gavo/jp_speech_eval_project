# UME-JRF expert-label criterion plan v1

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Status: **RESEARCH ACQUISITION / ADAPTER PREPARED — DATA NOT PRESENT IN REPOSITORY**
Product/runtime use: **PROHIBITED BY CORPUS LICENSE**
New user recording needed: **NO**

## Why this corpus is unusually valuable for the current Stage-0 problem

The current phone-CTC stack has passed model/API/frontend/native-anchor sanity checks, but it still lacks the criterion needed to interpret local phone evidence as pronunciation correctness. UME-JRF is much closer to that criterion than a new small convenience recording set.

The official NII Speech Resources Consortium page and database introduction state that UME-JRF was designed specifically for pronunciation-learning/education research and contains:

- learner speech aligned with Japanese pronunciation-education items;
- teacher/expert pronunciation ratings;
- native Japanese recordings of the same content;
- 141 international students, intermediate through advanced Japanese, covering 26 L1 backgrounds;
- 41 Tokyo/Kanto Japanese native speakers in the matched native portion;
- WAV, 16 kHz, 16-bit, mono.

Official sources:

- `https://research.nii.ac.jp/src/en/UME-JRF.html`
- `https://research.nii.ac.jp/src/UME-JRF.html`
- `https://research.nii.ac.jp/src/files/UME-JRF.pdf`
- data DOI: `https://doi.org/10.32130/src.UME-JRF`

The archived NII page notes that distribution is now through NII's Informatics Research Data Repository (IDR).

## Licensing boundary is non-negotiable

The official page says **“For research purpose only.”** The official introduction is even more explicit: the database was constructed for academic research and **commercial-purpose use is not permitted**.

Therefore:

- UME-JRF audio/labels must never be copied into the C-end runtime or product repository assets;
- they must not become a commercial training corpus unless separate permission is obtained;
- a model/artifact whose deployment rights depend on direct UME-JRF training must not be silently promoted to the commercial product;
- research conclusions about feature selection, construct validity and algorithm behavior may inform engineering decisions, but provenance must remain explicit;
- any future product model trained/fitted directly on UME-JRF requires a separate licensing review before deployment.

The adapter created for this project marks every UME-JRF criterion object `research_only_license=true` and `commercial_product_use_allowed=false`.

## Corpus structure relevant to our constructs

The official introduction defines four reading sets.

### A — phoneme-balanced sentences

ATR phonetically balanced reading sentences. Learners read one 50/53-sentence set. For every learner, the first five utterances of the assigned set were selected for expert rating.

Expert criterion:

- construct: broad pronunciation quality relative to an ideal Japanese speaker;
- scale: 1–5 absolute rating;
- useful for: utterance-level pronunciation evidence, not isolated-phone MDD.

### B — difficult-sound sentences

108 original sentences embedding difficult/minimal-pair items, split into two 54-sentence lists. Each learner reads one list; 28–29 utterances from that list were rated.

Expert criterion:

- item-specific question: whether the target difficult sound/minimal-pair word was pronounced correctly;
- scale: binary correct / incorrect;
- official example: `天気が悪いので、電気をつけた。` with the voiced/unvoiced contrast as the target;
- **highest-priority sentence-level criterion for local MDD/phone evidence**.

### C — prosody sentences

42 original dialogue-oriented sentences covering nine prosodic topics, including question intonation, prominence, branching structures, contrastive emphasis, sentence-final particles and fillers. Twelve utterances per learner were rated.

Expert criterion:

- item-specific prosodic target;
- scale: 1–5;
- official example: whether `誰` receives prominence in `誰がおどる？`;
- useful for the separate `抑揚`/prosody research line, **not for segmental clarity labels**.

### D — difficult-sound/minimal-pair words

115 words designed around difficult sounds/minimal pairs. Ten specific words were expert-rated for every learner:

`酸っぱい・全員・王座・通信・カミュ・廊下・友情・ビル・美容院・合唱`

Expert criterion:

- item-specific target-phone correctness;
- scale: 1–5;
- official example: for `酸っぱい`, whether the geminate/促音 is actually produced;
- **highest-priority isolated-word criterion for our phone/special-mora evidence**.

## Raters

The official introduction says the ratings were produced by **four Japanese-language-education experts with substantial experience in Japanese pronunciation education**. The public corpus page summarizes the supplement as grading lists by four native Japanese teachers.

We must preserve individual-rater raw labels when the corpus files make them available. Do not collapse them to `/100` on import.

## Priority order for this project

1. **D-rated 10 words** — local phone/special-mora criterion with simple isolated-word acoustics.
2. **B-rated difficult-sound sentences** — local phone correctness in sentence context.
3. **A-rated five sentences** — broad pronunciation-quality correlation with WavLM / sequence / clarity evidence.
4. **C-rated prosody sentences** — separate intonation/prosody study; never pool with segmental clarity labels.
5. Matched 41-speaker native data — native anchor distributions and false-alarm analysis, not a substitute for learner-error labels.

This ordering directly addresses the current Stage-0 blocker while minimizing construct mixing.

## Criterion schema before the actual corpus is obtained

The code now defines a strict research-only criterion object with:

- corpus = `UME-JRF`;
- set = `A | B | C | D`;
- construct = set-specific, not generic;
- scale = `ordinal_1_5` for A/C/D or `binary_correctness` for B;
- raw label preserved;
- pseudonymous rater ID;
- item/utterance ID;
- target-description field;
- `normalized_100 = null` by policy;
- `research_only_license = true`;
- `commercial_product_use_allowed = false`.

The importer deliberately **does not guess** how `FJlabel` files encode labels before the actual corpus documentation is inspected.

## Layout-probe-first policy

The official introduction points users to corpus-internal documentation such as:

- `${dvd}/Vol1/doc/FJcontent/description.txt`
- `${dvd}/Vol1/doc/FJlabel/description.txt`

Those are the only internal layout hints we treat as authoritative before acquisition. The new probe script:

- accepts a user-supplied/unpacked local UME-JRF root;
- searches for these documentation files and likely audio/label directories;
- hashes documentation files for provenance;
- emits a JSON layout report;
- performs **no network download**;
- performs **no label parsing** if the grading schema has not been inspected;
- refuses to call an unknown numeric column a pronunciation label.

Actual label/audio parsers should be added only after `FJlabel/description.txt` and the real directory tree are available.

## Frozen analysis plan once data are available

### D / B local correctness

For every rated target item:

- preserve all expert labels;
- report inter-rater agreement before producing an aggregate criterion;
- D 1–5: ordinal/mixed-effects treatment; do not pretend interval spacing is guaranteed;
- B correct/incorrect: rater-level binary criterion plus consensus/sensitivity analyses;
- evaluate phone feature vectors with leave-speaker-out and leave-item-out splits;
- report target-local false alarms on expert-correct learner speech;
- report detection on expert-incorrect speech;
- compare `{LPP, LPR, graph GOP, Occ}` against simpler baselines;
- keep phone-CTC backbone identity explicit; never average raw values across backbones.

### A broad pronunciation

Use as an utterance-level criterion for broad pronunciation/clarity evidence. Do not reinterpret its ideal-speaker 1–5 rating as intelligibility or recording quality.

### C prosody

Analyze separately with F0/prosodic features. Never use C labels to train a segmental clarity model.

## Promotion rule

UME-JRF can materially improve **research criterion validity**, but it cannot by itself authorize commercial deployment.

Phone evidence may move from `shadow` toward a C-end clarity component only if:

1. labeled B/D performance survives speaker- and item-held-out validation;
2. expert-correct learner false-alarm behavior is acceptable;
3. native anchors are not systematically penalized;
4. channel/gain robustness remains small relative to pronunciation effects;
5. any final product model is trained/calibrated on data whose commercial/product rights are compatible, or separate UME-JRF permission is obtained;
6. `/100` mapping is treated as a separate calibration problem.

Until then, UME-JRF is a research criterion source, not a product dataset.
