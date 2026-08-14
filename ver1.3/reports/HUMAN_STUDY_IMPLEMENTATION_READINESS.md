# Human Pronunciation Study v1 — Implementation Readiness

## Delivered infrastructure

| Location | Purpose | Stage |
|---|---|---|
| `data/human_eval/pilot/pronunciation_listener_manifest_v1.csv` | Internal pilot master manifest | `pilot_seed` |
| `data/human_eval/pilot/pronunciation_listener_blind_v1.csv` | Rater-safe manifest | `pilot_seed` |
| `data/human_eval/pilot/listener_assignment_v1.csv` | Pilot-only incomplete-rater assignment | `pilot_seed` |
| `data/human_eval/pilot/human_rating_schema.json` | Human study contract | `pilot_seed` |
| `data/human_eval/pilot/human_rating_json_schema_v1.json` | Machine-validatable conditional schema | `pilot_seed` |
| `data/human_eval/final_template/*` | Empty final-manifest and assignment/recruitment templates | unpopulated |

The pilot and final artifacts are intentionally separated. No `final_calibration` assignment exists.

## Counts and gaps

| Measure | Current | Requirement / implication |
|---|---:|---|
| Pilot unique clips | 98 | Seed/usability-only, not final mapping validation. |
| Current same-target learner clean clips | 14 | 2 speakers × 7 targets. |
| Learner recruitment gap | 56 | 8 new speakers × 7 targets; final learner cohort = 70 clips / 10 speakers. |
| Projected unique clips after learner completion | ≈154 | Assumes current seed clips are retained. |
| Complete channel sets | 14 | Promotion protocol needs ≥15; gap = 1 real quartet. |
| Pilot base ratings | 490 | Five ratings/clip across an incomplete ten-slot rater design. |
| Pilot presentations | 500 | Ten total hidden repeats; one per listener slot is insufficient for stable repeatability. |
| Formal repeat plan | ≥5 / rater | Prefer 5–10% of workload; no formal assignment generated yet. |

## Statistical design decisions frozen before collection

- Primary label: ordinal 1–7 pronunciation accuracy, not naturalness or comprehensibility.
- Analyzability is independent; an unanalyzable clip may have a null accuracy rating.
- Primary rater cohort: native Japanese listeners. Near-native listeners receive a separate sensitivity analysis.
- Incomplete-rater agreement: Krippendorff alpha with ordinal distance.
- Average-rating reliability: generalizability-theory / mixed-effects variance components, with clip and rater effects and documented target effect treatment.
- Association/validation: raw and rater-normalized analyses, overall/within-target Spearman, leave-one-target-out, leave-one-speaker-out, speaker effects, and channel-set contrast analysis.

## Channel terminology and validation rule

The unit is a **channel set** (or channel quartet): clean, RIR, noise15dB, and codec. It provides three paired contrasts against clean. A quartet is complete only when all four conditions exist. This design directly tests whether WavLM changes more under channel perturbation than human pronunciation accuracy does.

## Scope and safety

`PronunciationCalibration.valid_target_scope` is now `janon_7target_isolated_word_validation_v1`; it is not a generic same-target or arbitrary-Japanese claim. Mapping remains `None`, production is disabled, ProductScore v2 is unchanged, and ProductScore v3 is not enabled.

## Gates

### PILOT READY: PASS

The seed manifest is explicitly labeled, rater-visible metadata is blind, independent analyzability is defined, pilot duplicate randomization is tested, and pilot ratings are excluded from final calibration data.

### FINAL HUMAN STUDY READY: BLOCK

The final 56 real learner recordings do not exist, the cohort has only 2 rather than 10 learner speakers, and 14 rather than 15 complete channel sets exist. The final manifest template is intentionally empty and the guarded final-assignment builder refuses the seed cohort.

### PRODUCT SCORE MAPPING READY: BLOCK

No formal listener ratings, reliability estimates, channel-bias results, or held-out target/speaker validation exist. No /100 mapping may be introduced.
