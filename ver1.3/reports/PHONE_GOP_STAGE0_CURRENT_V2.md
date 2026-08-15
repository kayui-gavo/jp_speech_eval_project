# Phone-GOP Stage-0 current gate v2

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Product impact: **NONE**
Human-recording gate: **BLOCKED**

## Current objective

The current task is not to tune a learner-facing `明瞭さ /100`. It is to decide
whether the Japanese phone-CTC evidence stack is technically and scientifically
stable enough to justify spending human recording / annotation time.

A valid Stage-0 promotion requires automatic evidence for all of the following:

1. reproducible pinned phone-CTC backbones;
2. target frontend / phone inventory compatibility;
3. high-vowel allophone safety (`i/I`, `u/U`);
4. exclusion of pause/control tokens from segmental competition;
5. exact CTC sequence probability for substitution/deletion alternatives;
6. alignment-free local evidence that does not require external forced phone
   boundaries;
7. native-speech false-alarm pressure tests;
8. real Japanese-L2 domain sanity on existing public speech;
9. controlled local substitution/deletion directionality without new human
   recording;
10. explicit uncertainty / CTC-posterior peakiness diagnostics;
11. no hidden mapping of raw GOP/LPR/entropy/Occ(i) into a consumer `/100`.

## Automatic stack currently under test

### Backbones

- Beatrice Japanese HuBERT phone CTC, pinned revision;
- DistilHuBERT Japanese dual-CTC, pinned revision;
- WavLM Japanese dual-CTC research comparator, pinned revision.

Raw outputs from different backbones remain model-specific and are never
averaged.

### Feature families

The repository now keeps distinct research families instead of collapsing them:

- frame-local CTC support + mean/max logit margins;
- posterior GOP margin + entropy;
- alignment-free exact CTC canonical/substitution/deletion LPRs;
- enumerated SD-GOP features;
- Japanese phone-masked normalized SD alternative graph + `Occ(i)`;
- hybrid criterion-ready bundle that joins the above only as supervised model
  inputs, not as a score;
- CTC posterior peakiness / blank dominance / entropy diagnostics.

`Occ(i)` is graph occupancy, not phone duration. CTC Viterbi support frames are
not physical phone boundaries. Whole-sequence canonical log posterior is not a
phone-local correctness probability.

### Restricted Japanese substitution search

`restricted_japanese_phonology_v1` is a research hypothesis, motivated by
restricted phonological search work but not yet Japanese-L2 validated.

- ordinary phones use auditable Japanese phone-token neighborhoods;
- `N` and `cl` use canonical-vs-deletion evidence only in the restricted path;
- `pau`, `sil`, tokenizer controls and blank cannot be pronunciation
  alternatives;
- missing curated neighborhoods fall back only with explicit provenance;
- because the RPS denominator depends on candidate count, raw RPS-GOP values are
  not compared across phone types and are not directly subtracted from UPS-GOP.

Controlled tests prefer target-specific substitution/deletion LPRs for a fixed
contrast.

## Existing automatic speech anchors

### Native Japanese

The heavy preflight downloads the three official JVS public sample anchors
only for the job, then deletes the audio before artifact collection. They are
used to pressure-test native false alarms and candidate behavior, not to prove
learner-error detection.

### Real Japanese L2 speech

The heavy preflight also downloads the five public UME-JRF test-listening WAVs
from the official NII-SRC page. The page identifies these as examples by native
speakers of Chinese. The public page does not expose their four-teacher grades,
so they are **unlabeled learner-domain anchors only**.

The published minimal-pair samples `じぶつ / じんぶつ` provide an especially
useful real-speech `N` presence/absence contrast. The same-waveform target vs
partner sequence preference is recorded, but its sign is not treated as
pronunciation ground truth.

UME-JRF remains research-use only and cannot become a product runtime/training
asset merely because the public sample links are accessible.

### Synthetic local edit sanity

Ephemeral OpenJTalk controls test whether target-specific evidence moves in the
expected direction for:

- `b -> p`
- `g -> k`
- `ts -> s`
- `sh -> s`
- `ch -> sh`
- `cl` deletion
- `N` deletion

Generated audio is never committed or uploaded. Passing this test demonstrates
implementation/localization sanity only, not Japanese learner validity.

## Performance hardening

Restricted substitution/deletion evaluation originally used a scalar Python
CTC forward recurrence and made native preflight unnecessarily expensive. The
current branch introduces `rolling_vectorized_exact_ctc_v1`, regression-tested
against the earlier scalar implementation for:

- distinct label sequences;
- repeated labels;
- empty sequence;
- impossible repeated-label paths;
- target blank rejection.

The restricted extractor uses this exact vectorized recurrence for canonical,
substitution and deletion sequence likelihoods.

## Current test status before this heavy run

Ordinary CI at commit `ed5416f57ff931bcf1330ad7fd3acb4854da87d8`:

- **301 passed**
- **6 warnings**

The warnings are existing deprecation / audioread fallback warnings and are not
phone-GOP assertion failures.

This file intentionally triggers a fresh heavy model preflight on the current
Stage-0 tree. Promotion is decided only after inspecting its machine artifacts.

## Promotion rule

Even a completely green heavy preflight does **not** authorize direct C-end
clarity mapping. It can only promote the stack from "implementation not yet
trusted" to "ready for labeled criterion validation".

Before C-end `明瞭さ` can use phone-GOP as primary mapped evidence, require:

- Japanese-L2 criterion labels (ideally phone-level expert labels);
- speaker-held-out validation;
- expert-correct / native false-alarm calibration;
- local controlled-error discrimination;
- channel / speed robustness;
- phone-specific rather than one-global-threshold analysis;
- explicit uncertainty / abstention policy for phone-level feedback;
- final score mapping calibrated separately from the raw research features.

Until that happens:

- `score_mapped = false` for GOP research evidence;
- `product_calibrated = false`;
- current C-end four-score fallback remains unchanged;
- **no new human GOP recording is requested from the user**.
