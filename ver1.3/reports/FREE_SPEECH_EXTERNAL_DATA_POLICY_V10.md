# Free-Speech External Data Policy v10

## Purpose

This note separates three different questions that must not be conflated:

1. Is a corpus scientifically relevant to the target construct?
2. Is the corpus legally/contractually usable for the intended research workflow?
3. May evidence derived from that corpus be used to change a commercial C-end ProductScore?

A corpus can be excellent for question 1 and still fail question 3.

This policy is conservative by design. It does not replace the actual corpus agreement or legal advice.

## I-JAS

Corpus: International Corpus of Japanese as a Second Language / 多言語母語の日本語学習者横断コーパス.

### Scientific fit

I-JAS is highly relevant to free-speech validation:

- 1,000 Japanese learners and 50 native Japanese speakers;
- multiple L1s and learning environments;
- objective learner proficiency tests/background metadata;
- spoken tasks include story telling, roughly 30-minute dialogue, role play, and picture description;
- speech audio is published online.

This makes I-JAS much closer to the v10 spontaneous/dialogue constructs than the repository's historical JANON isolated words or JVS scripted read speech.

### Current contractual boundary

The current official I-JAS online Terms of Service state that the permitted purpose is research as declared in the application. If research results are to be used for commercial purposes, separate consultation with the rights holder is required.

The current NINJAL Chunagon application guidance also states that commercial use is not accepted for the free service and that applicants wanting AI development / machine-learning use should consider the paid edition.

The I-JAS agreement also prohibits redistribution/transfer/sale/distribution of corpus data to third parties beyond the permitted scope.

### Repo policy

Until explicit permission covering the intended product-development use is obtained:

- I-JAS may be treated as a **research-only external shadow-validation candidate**, subject to the registered research purpose and all I-JAS terms;
- do not commit I-JAS audio or corpus data to this GitHub repository;
- do not redistribute I-JAS audio inside listener packs to people who are not authorized under the applicable agreement;
- do not use I-JAS to fit or tune a production score mapping;
- do not treat an I-JAS-only pass as sufficient authorization for a commercial ProductScore promotion;
- do not build an automatic importer until the actual authorized export/audio structure and intended use have been confirmed;
- if I-JAS evidence is used in an academic publication, follow the official citation/reporting requirements.

A useful future workflow, if access terms permit the specific research experiment, is:

`I-JAS -> external shadow benchmark -> compare candidate ordering / failure modes -> no ProductScore mutation`

The v10 self-collected, explicitly consented/provenanced held set remains the cleanest promotion evidence for the C-end product.

## UME-JRF

The current official distribution terms identify UME-JRF as research-purpose data. It is also a read-aloud learner corpus rather than spontaneous free speech.

Repo policy:

- research-only fixed-reading validation;
- useful for pronunciation/timing/reference-based research;
- not a substitute for spontaneous fluency/dialogue validation;
- not sufficient by itself for a commercial free-speech ProductScore promotion.

## JVS / JANON already in the project history

The v10 historical-reuse audit remains authoritative for the current project assets:

- JVS parallel read speech -> native fixed-reading/broad-mode regression;
- JANON isolated learner/native words -> fixed-reading regression;
- synthetic JVS edits -> engineering robustness;
- none are relabeled as held learner free-speech criterion data.

Scientific relevance does not change because an evaluator was run in a broad/free-speech mode.

## CSJ

CSJ is a strong native spontaneous-speech resource, but it is not L2 learner data. The current official NINJAL page states that commercial-purpose use is considered individually.

Repo policy:

- potentially useful for native spontaneous-speech research controls;
- cannot replace held learner validation;
- do not assume commercial permission;
- do not let native CSJ performance define the learner construct.

## I-JAS foreign-language task data / other negative controls

Real English/Mandarin speech matched to similar elicitation tasks would be preferable to TTS-only negative controls for the final language-routing check.

However, negative-control audio remains subject to its source agreement. A convenient corpus source is not automatically redistributable or commercially usable.

The simplest commercial-compatible final routing set remains newly collected or otherwise explicitly licensed real non-Japanese speech with clear provenance.

## Promotion evidence hierarchy

For the current C-end project, use the following hierarchy:

### Tier A — Product-promotion evidence

Self-collected or otherwise explicitly licensed data whose consent/provenance permits the intended product-development use.

This is the required foundation for a score-changing branch.

### Tier B — External research validation

Scientifically relevant corpora whose terms permit the specific research analysis but do not yet authorize commercial use of the resulting evidence.

Use for:

- external replication;
- failure-mode discovery;
- candidate stress testing;
- academic research.

Do not let Tier B alone change ProductScore.

### Tier C — Engineering regression / routing controls

Fixed reading, synthetic perturbations, demo recordings, and negative controls that test implementation behavior but do not establish construct validity.

Do not promote from Tier C.

## Why this matters

The goal is not to avoid useful public corpora. The goal is to avoid silently turning a research corpus into a commercial score-calibration asset, or turning read speech into spontaneous-speech validity evidence.

The v10 pipeline therefore keeps data provenance and construct provenance as first-class promotion boundaries, alongside model performance.
