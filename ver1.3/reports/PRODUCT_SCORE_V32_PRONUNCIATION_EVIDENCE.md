# Product Score v3.2 — Pronunciation Evidence Validation

## Executive verdict

**PASS FOR CANDIDATE-ONLY RESEARCH; NOT READY FOR USER-FACING SCORE MAPPING.**

This validation leaves ProductScore v2 user-facing behavior intact.  The seven-target WavLM study shows useful same-target, multi-reference separation on most weak native/learner contrasts and very strong wrong-target separation, but it also exposes speaker/channel effects and one target (`バグ`) where a learner is not reliably farther than the native leave-one-out distribution.  There are no independent human pronunciation ratings, so the SSL distance remains an uncalibrated evidence index rather than a correctness or /100 score.

## 1. V2 user-facing parity audit

The same 45 real WAV/cache/config cases were evaluated from baseline `c610dab37992119a78dab25104bc4c590d3c1f57` and the v3.2 working tree.  The input set contains the existing 40 C-end cases plus five channel conditions (gain, RIR, 15 dB noise, bandlimit, codec), including fixed-reference and fallback-equal cases.

| Result | Count | Interpretation |
|---|---:|---|
| Identical user-facing result | 44 | Same display score, status, detail availability, alignment mode, and three legacy dimensions. |
| Explained availability change | 1 | `learner_enf1_i5` / target `バグ`: the pre-existing broad-Japanese fallback retains the Japanese ASR transcript and gives display score 78 instead of a baseline no-score. It is not caused by the `AlignmentResult` refactor or v3.2 SSL semantics. |
| User-facing regression | 0 | No blocker found. |

The detailed per-sample record is [v2_parity.csv](../outputs/product_score_v32/v2_parity.csv).  This is parity at the product surface; v3 remains shadow-only and does not replace `display_score`.

## 2. Eligibility and evidence semantics

`diagnostic_candidate_eligible` now means there are at least two evidence dimensions with coverage of at least 0.45.  It supports dimension research only.

`overall_product_score_candidate_eligible` is deliberately stricter: it needs pronunciation evidence, at least one independent dimension, coverage at least 0.65, and a **calibrated numeric** pronunciation value.  An uncalibrated SSL distance cannot create an overall product score.

| Telemetry population | Scope | Count | Overall A/B eligible |
|---|---|---:|---:|
| Previous 37 fixed-reference v3.1 panel, without SSL mapping | `delivery_prosody` | 29 | 0 |
| Previous 37 fixed-reference v3.1 panel, without SSL mapping | `continuity_only` | 8 | 0 |
| 21 complete-bank telemetry samples after raw SSL attachment | `full` | 21 | 0 |
| Codec example after raw global SSL attachment | `pronunciation_plus_delivery` | 1 | 0 |

The `full` label above describes the *available evidence scope*, not a released total score: the SSL pronunciation dimension has `value=null` and `score_mapped=false`.  All 22 attached-SSL records report `overall_product_score_candidate_eligibility_reason=pronunciation_evidence_not_score_mapped` in [v32_ssl_telemetry.csv](../outputs/product_score_v32/v32_ssl_telemetry.csv).

The fallback-equal rule remains explicit: local mora timing/rhythm and local special-mora evidence are unavailable when boundaries are synthetic.  A valid global WavLM comparison may still provide pronunciation evidence because it does not consume those boundaries.

## 3. SSL confidence is independent of alignment

`ssl_pronunciation_confidence` is a reliability field only.  It uses:

- native-reference count (four references is the current complete-bank target);
- dispersion among reference distances;
- SSL execution/distance stability;
- recording-quality reliability; and
- content verification plus valid audio as hard availability requirements.

It does **not** use mora/MFCC alignment confidence.  The codec case demonstrates why: MFCC local alignment is unavailable (`boundary_health_unstable`), while global WavLM distance is still obtained.  The current recording-quality heuristic remained 1.0 for these synthetic transformations, so its 10% confidence term did not reduce them; that is a limitation of this particular channel-control set, not evidence of channel invariance.

## 4. Seven-target multi-reference WavLM run

Data were restricted to the seven same-target banks: 4 human native recordings (`jpf1`, `jpf2`, `jpm1`, `jpm2`) per target, native leave-one-out comparisons (28), 2 learner recordings (`chf1`, `enf1`) per target (14), and one deliberately wrong-target native comparison per target (7).  Distances are cosine-DTW normalized cumulative distances from `microsoft/wavlm-large`, layers 12 and 24.  They are not listener ratings.

### Pooled raw distance distributions

Values are the simple mean of layer-12 and layer-24 aggregated distances; entries are median / SD.

| Aggregation | Native LOO (n=28) | Learner (n=14) | Wrong target (n=7) |
|---|---:|---:|---:|
| Median | 0.1979 / 0.0234 | 0.2327 / 0.0295 | 0.4868 / 0.0492 |
| Trimmed mean | 0.1999 / 0.0200 | 0.2327 / 0.0295 | 0.4868 / 0.0492 |
| Top-2 mean | 0.1849 / 0.0203 | 0.2154 / 0.0289 | 0.4781 / 0.0510 |
| Nearest | 0.1746 / 0.0202 | 0.2073 / 0.0328 | 0.4718 / 0.0525 |

Median aggregation is the conservative default research choice: it is less permissive than nearest while preserving the expected pooled native < learner < wrong-target ordering.  It is not a final product strategy.

### Per-target weak-label summary (median aggregation)

`P` is the fraction of that target's four native LOO distances strictly below the learner distance.  The learner/native effect is standardized only against this small native sample.  It is weak validation, not a claim of pronunciation correctness.

| Target | Native LOO median (SD) | chf1 distance / P | enf1 distance / P | Learner median effect |
|---|---:|---:|---:|---:|
| うっとうしい | 0.1892 (0.0172) | 0.2216 / 1.00 | 0.2374 / 1.00 | 2.34 |
| がっしり | 0.1860 (0.0238) | 0.2118 / 0.75 | 0.2584 / 1.00 | 2.06 |
| さっさと | 0.1889 (0.0088) | 0.2442 / 1.00 | 0.2921 / 1.00 | 8.98 |
| ばっちり | 0.2089 (0.0067) | 0.2069 / 0.50 | 0.2695 / 1.00 | 4.35 |
| オイル | 0.1980 (0.0229) | 0.2279 / 0.75 | 0.3083 / 1.00 | 3.07 |
| バグ | 0.2398 (0.0229) | 0.2263 / 0.25 | 0.2498 / 0.75 | -0.07 |
| 酸味 | 0.1892 (0.0153) | 0.2169 / 0.75 | 0.2186 / 0.75 | 1.87 |

The target-level artifact is [wavlm_7target_summary.csv](../outputs/product_score_v32/wavlm_7target_summary.csv); all per-comparison raw values are in [wavlm_7target_raw.csv](../outputs/product_score_v32/wavlm_7target_raw.csv).

## 5. Layer fusion and leave-one-target-out selection

Each of seven folds uses the other six targets for robust native median/MAD normalization and selects aggregation plus alpha in `D = alpha * L12_norm + (1-alpha) * L24_norm`.  The held target is not used for its own selection.

- `median`, alpha=1.00 (layer 12 only): 5/7 held-out targets.
- `nearest`, alpha=0.75: `オイル` only.
- `top2_mean`, alpha=1.00: `酸味` only.

Thus layer 12 with median aggregation is the modal robust choice, while no universal fusion is justified.  Layer 24's apparent wrong-target separation is useful research evidence but increases speaker variability; it should not be globally fused into a product score before listener-label calibration.  Fold-level distributions and normalizers are in [wavlm_7target_loto_fusion.csv](../outputs/product_score_v32/wavlm_7target_loto_fusion.csv).

## 6. Speaker effect

Native leave-one-out medians show residual speaker/channel identity.  At layer 24, `jpm2` has median 0.2293 and SD 0.0449, versus 0.2015–0.2038 and SD 0.0164–0.0323 for the other three speakers.  Layer 12 is more stable (speaker medians 0.1825–0.1940).  This supports the conservative layer-12/median research choice and blocks a speaker-agnostic /100 mapping today.  Full speaker statistics are in [wavlm_7target_speaker_effect.csv](../outputs/product_score_v32/wavlm_7target_speaker_effect.csv).

## 7. Channel robustness and speech-edit sensitivity

| Condition | MFCC local alignment | WavLM delta from clean | Interpretation |
|---|---|---:|---|
| Gain +6 dB | available | -0.0007 | Essentially stable. |
| Bandlimit | available | +0.0140 | Small change. |
| 15 dB noise | available | +0.0639 | Material channel sensitivity. |
| Mild RIR | available | +0.0935 | Largest channel sensitivity. |
| Codec | unavailable: `boundary_health_unstable` | +0.0536 | Global SSL still runs; local timing remains unavailable. |
| Consonant attenuation | available | +0.0061 | Weak response. |
| Vowel spectral tilt | available | -0.0003 | No useful response here. |
| Local delete-like edit | unavailable: `boundary_health_unstable` | +0.0315 | Modest global response; local timing correctly withheld. |

The exact fields, including reference dispersion and SSL confidence, are in [wavlm_jvs_channel_and_speech.csv](../outputs/product_score_v32/wavlm_jvs_channel_and_speech.csv).  The desired codec semantic holds, but overall channel robustness is **not yet strong enough** to call global WavLM a clean pronunciation-only measure.  The tested synthetic speech edits also do not establish robust local phonetic sensitivity; no thresholds were changed to force that conclusion.

## 8. Human-label gap and minimum next protocol

The observed native/learner ordering cannot validate pronunciation correctness because learner identity is only a weak label.  Before fitting any SSL-to-/100 mapping, collect a small independent listening study:

- at least 60 same-target clips, stratified across the seven targets, native references, learners, and channel conditions;
- at least 5 independent Japanese-proficient raters per clip;
- a single clearly defined segmental/pronunciation naturalness rating (not rhythm, fluency, or recording quality), with randomized presentation and hidden source identity;
- inter-rater reliability and rater-normalized score analysis; and
- a held-out target/speaker split for mapping selection.

That is the minimum evidence needed to decide whether a robust layer-12/median SSL index can become a calibrated candidate dimension.

## 9. Test and release state

Targeted semantic tests passed: **14 passed**.  Full repository suite: **173 passed, 6 warnings**.  Ordinary pytest does not instantiate or download WavLM; the model is loaded only by the explicit audit script with `local_files_only=True`.

## Recommendation

Continue v3.2 as internal telemetry only.  It is worthwhile to enter human-rating calibration planning, but **do not enable an overall ProductScore v3 A/B score**, do not map SSL distance to /100, and do not relax codec MFCC alignment standards.  A future candidate may use global SSL pronunciation evidence together with delivery dimensions only after the label study and a channel-robustness decision.
