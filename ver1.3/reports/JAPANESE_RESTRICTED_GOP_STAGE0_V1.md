# Japanese restricted-substitution GOP — Stage-0 v1

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Product impact: **NONE**
Human-recording gate: **BLOCKED**

## Why this branch exists

The current unrestricted segmentation-free CTC-GOP research path asks every
ordinary phone token to compete with a large backend inventory. Recent
alignment-free GOP work reports that phonologically restricted substitution
search can improve MDD efficiency/performance. That published positive evidence
is not Japanese-L2 validation, so the Japanese restricted search introduced
here is explicitly a **research hypothesis**.

The goal is not to make the C-end `明瞭さ` score larger or more stable by
construction. The goal is to test whether a smaller, auditable Japanese
alternative set produces more local and criterion-relevant phone evidence with
fewer native false alarms.

## Candidate policy

Implementation:

- `src/jp_speech_eval/japanese_phone_substitutions.py`
- policy id: `restricted_japanese_phonology_v1`
- status: `research_hypothesis_not_learner_validated`

Rules:

1. Candidate labels are **phone tokens**, never kana strings.
2. `pau`, `sil`, CTC blank and tokenizer special symbols cannot be segmental
   pronunciation alternatives.
3. `i/I` and `u/U` are logical allophone classes; normal high-vowel devoicing is
   not penalized merely because a backend uses upper/lowercase labels.
4. `N` and `cl` are special morae. In the restricted path they receive
   canonical-vs-deletion evidence only; their principal learner feedback still
   requires duration/context evidence and belongs primarily to rhythm/special-
   mora diagnostics rather than ordinary segmental clarity.
5. If an ordinary phone has no curated neighborhood in a backend vocabulary,
   an unrestricted segmental fallback is allowed only with explicit provenance.

The exact neighborhood table is not claimed to be an empirical Japanese
learner confusion matrix. It must be judged later against expert-labeled
Japanese learner speech.

## Restricted segmentation-free extractor

Implementation:

`src/jp_speech_eval/restricted_segmentation_free_gop.py`

Method:

`restricted_enumerated_fgop_ctc_sf_sd_v1`

For each canonical phone position, it computes exact CTC sequence evidence for:

- the canonical target;
- only that position's allowed substitution alternatives;
- deletion of that target phone.

It is phone-boundary-free. It does **not** yet model insertion in the restricted
path.

### Critical scale rule

The raw SD denominator changes with the candidate set. Therefore:

- raw RPS `gop_sf_sd` is a research feature;
- raw RPS values must not be compared directly across phones with different
  candidate sets;
- raw RPS and unrestricted SD-GOP values must not be subtracted as if they were
  on one common scale;
- clip/group means of heterogeneous raw RPS-GOP are not pronunciation-quality
  measurements.

The code now carries machine-readable guards:

- `phone_dependent_denominator = true`
- `candidate_count_affects_raw_denominator = true`
- `cross_phone_raw_gop_comparison_allowed = false`
- `rps_vs_ups_raw_gop_direct_comparison_allowed = false`

Controlled local tests should prefer a **target-specific substitution/deletion
LPR** for a fixed contrast, while criterion modeling may later learn from the
full feature vector.

## Legacy phoneme-confusion quarantine

`src/jp_speech_eval/phoneme_confusion.py` previously contained a research helper
that was unsafe for future pronunciation use. It has been quarantined because:

- a low Bhattacharyya coefficient was treated as if it meant high confusion,
  reversing the coefficient's overlap semantics;
- posterior columns over time were treated as calibrated phone distributions;
- kana strings, including an identity pair, were presented as "phoneme"
  confusion pairs;
- a faster-Whisper placeholder could be mistaken for a phone-CTC posterior
  extractor;
- an unverifiable project attribution appeared in its comments.

The generic probability-distribution utilities remain, with correct semantics.
The legacy detector now fails closed unless explicitly opted into and is marked
exploratory. Whisper-to-phone-posterior extraction now raises
`NotImplementedError`. No product scorer uses this path.

## Automatic tests before any new human recording

### A. Official JVS native false-alarm pressure test

`scripts/run_jvs_restricted_gop_preflight.py`

On the three existing official JVS native anchors, record:

- how often an RPS noncanonical alternative out-scores canonical;
- which phone/alternative positions cause that behavior;
- special-mora vs ordinary segmental behavior;
- candidate-count reduction vs the unrestricted inventory;
- any fallback positions.

A negative local LPR on native speech is **not** labeled a pronunciation error.
A high native rate instead blocks user-facing interpretation and points to the
candidate policy/backbone/feature formulation as the problem.

### B. Ephemeral controlled local phone-edit test

`scripts/run_controlled_tts_phone_edit_preflight.py`

OpenJTalk speech is generated only inside the test process and is not committed
or uploaded. Current controls include:

- `バスです。` → `パスです。` (`b -> p`)
- `かぎです。` → `かきです。` (`g -> k`)
- `つきです。` → `すきです。` (`ts -> s`)
- `しゃしんです。` → `さしんです。` (`sh -> s`)
- `ちずです。` → `しずです。` (`ch -> sh`)
- `かっこです。` → `かこです。` (`cl` deletion)
- `みんなです。` → `みなです。` (`N` deletion)

For substitution controls, the intended error phone must be present in the RPS
candidate set. For deletion controls, the target-vs-deletion LPR is checked.
The main sanity condition is directional: the target-specific LPR should fall
when the controlled error is synthesized. A positive-to-negative sign flip is
recorded separately but is not required as a universal learner threshold.

Synthetic TTS success would prove only implementation/localization sanity, not
Japanese-L2 criterion validity.

### C. Existing-data benchmark

`scripts/benchmark_restricted_gop_existing_data.py`

This consumes only audio already present near the research workspace. It does
not download JANON or any learner corpus and does not infer that non-native
speech is incorrect. Group summaries intentionally omit heterogeneous raw RPS
GOP means as a quality statistic.

## Criterion path

The strongest currently identified criterion path remains UME-JRF because its
expert labels separate broad pronunciation, difficult-sound correctness,
prosody and target-phone correctness. UME-JRF is research-only/non-commercial,
so it is an algorithm-validation source, not a product runtime/training asset.
The repository's adapter remains fail-closed until the real corpus-internal
`FJlabel` documentation is available; no label-table format is guessed.

For any eventual RPS/UPS criterion comparison, evaluate at minimum:

- target-specific expert-correct vs expert-incorrect discrimination;
- expert-correct learner false alarms;
- speaker-held-out splits;
- item/target-phone held-out stress tests where feasible;
- phone-specific calibration rather than one global threshold;
- RPS vs UPS criterion performance and computation, not raw score magnitude.

## Product boundary

None of this changes:

- C-end `明瞭さ`;
- the four-dimension consumer score;
- overall `/100`;
- human-recording eligibility.

Phone evidence stays shadow/research-only until Japanese-L2 criterion validity
and product-compatible data rights are demonstrated.
