# Product Score v3 candidate benchmark

## Executive finding

**Do not promote v3 to a feature-flag A/B yet.**  This branch adds an
evidence-aware, default-off candidate pipeline and demonstrates a viable
base-first content cascade.  It also makes `fallback_equal` explicit as
synthetic timing evidence rather than treating its near-zero mora-duration CV
as excellent speech.  The real-audio results identify two remaining promotion
blocks: channel perturbations still frequently flip alignment to
`fallback_equal`, and the current continuous fluency candidate is too
permissive when it is the only available dimension.  WavLM is executable, but
the available same-target learner panel has only one native reference per
word and does not support a product score mapping.

All v3 outputs are candidate telemetry.  `ProductScore v2` remains the only
source of learner-facing `display_score`.

## 1. Content cascade

The frozen hard panel contains 30 correct fixed targets and 50
duration-matched wrong Japanese targets.  Values below replay the actual
base/small/tiny ASR outputs from the same panel; no target label was changed.

| policy | wrong false verified | fixed-detail retention (correct) | broad fallback | small invoked | warm p50 / p90 s |
|---|---:|---:|---:|---:|---:|
| always small | 0.0% | 93.33% | 65.00% | 0.00% | 1.663 / 1.898 |
| always base | 0.0% | 83.33% | 68.75% | 0.00% | 0.454 / 0.503 |
| base first, direct broad | 0.0% | 83.33% | 68.75% | 0.00% | 0.658 / 0.752 |
| base first + selective small rescue | 0.0% | 90.00% | 66.25% | 5.00% | 0.660 / 1.229 |
| tiny first + small rescue | 0.0% | 93.33% | 65.00% | 72.50% | 2.074 / 2.306 |

The selective-rescue conflict band was determined on the frozen development
panel: base false mismatches for correct targets had kana similarity
0.475–0.710, whereas the 50 hard wrong targets were at most 0.220.  It is
implemented as an **off-by-default candidate** (`cascade_policy =
base_first_selective_small_rescue`), not a new production default.  Its cold
latency has not yet been measured as a full end-to-end cascade and must be
done before promotion.  The existing model cold measurements were base 1.229
s and small 2.492 s.

## 2. Language eligibility v3

Broad fallback now records separate speech-present, periodic-voice, ASR
language, script, and lexical evidence with `eligible`, `uncertain`, or
`ineligible` state.  Morphology is no longer a stand-alone rule that rejects
multi-token noun phrases.  Unit coverage explicitly accepts:

- `はい`, `いいえ`, `寿司`, `東京`, `ラーメン`, `コーヒー`, `ありがとうございます`
- `東京大学`, `新宿駅東口`, `日本語能力試験`, `人工知能研究`, `大学院入学試験`

English, Mandarin, silence, and white/pink noise controls remain rejected.
The observed forced-Japanese Mandarin hallucination is rejected only by a
joint ASR-language plus fragmented-token signal; a missing particle or verb
does not reject a Japanese noun phrase.

## 3. Fallback-equal integrity

`DimensionEvidence` now carries `value`, `available`, `confidence`, `source`,
and `reason`.  On `cached_dtw_fallback_equal`, v3 reports:

- `evidence_source = synthetic_equal_boundaries`
- rhythm unavailable; no mora-duration CV, local warp, or special-mora timing
  is consumed as high-quality evidence
- intonation unavailable because its target-local transition alignment is not
  reliable
- the aggregate re-normalizes only actually available dimensions; it does not
  inject a nominal 80 for missing dimensions

This prevents the previous equal-boundary timing proxy from masquerading as
near-perfect rhythm.  It does **not** make fallback_equal a scientifically
complete score: generic continuous fluency can still be available, and the
candidate is not eligible for promotion until that remaining limitation is
addressed.

## 4. Current v2 saturation vs v3 candidate distribution

The existing 40-case v2 acceptance panel had mean 89.825, median 90,
21/40 at least 90, and 7/40 exactly 100.  The v3 candidate was rescored on 37
runnable fixed-reference cases with content matching disabled solely to keep
this dimension benchmark independent of ASR latency:

| group | n | mean | SD | min–max | >=95 |
|---|---:|---:|---:|---:|---:|
| native | 17 | 82.15 | 9.11 | 65.58–100.00 | 2 |
| learner (JANON) | 14 | 81.17 | 8.11 | 67.66–94.01 | 0 |
| real general fallback | 2 | 72.71 | 0.00 | 72.71–72.71 | 0 |
| partial edits | 2 | 68.53 | 9.19 | 59.34–77.72 | 0 |
| all fixed runnable | 37 | 80.91 | 9.50 | 59.34–100.00 | 3 |

There are 35 distinct candidate values, so this is not merely a cosmetic
ceiling compression.  However one weak-reference sample and two native
fallback-equal samples still reach 100 through fluency alone.  Therefore the
candidate has better spread but fails the evidence-completeness requirement.

## 5. Reference-relative rhythm and continuous fluency

For non-synthetic boundaries v3 stores raw `rate_log_ratio`,
`warp_slope_cv`, and `warp_local_deviation` before mapping.  Rhythm uses a
modest global-speed term and primarily local normalized-warp deviation.
Fluency separately records continuous pause ratio, pause count, mean pause,
and speech-run duration; it does not use the old `4–7 mora/s => 100` plateau.

On the controlled ladder, ±10% speed did not need strict monotonicity:
0.9x produced 79.54 versus clean 76.06, while 1.1x fell back to synthetic
boundaries and produced 69.23.  The latter is an alignment robustness issue,
not evidence that a 10% speedup is inherently poor.  An inserted 0.8 s pause
fell to 59.84, but a local-delete edit became fallback_equal and only retained
generic fluency.  These findings block a rhythm/fluency A/B until alignment
robustness and the fallback evidence policy are improved.

## 6. Channel robustness and speech sensitivity

All audio below was an existing perturbation of the same real JVS sentence;
no corpus file was modified.

| condition | class | candidate | alignment | interpretation |
|---|---|---:|---|---|
| clean | baseline | 75.97 | cached DTW | baseline |
| gain +6 dB | channel | 76.06 | cached DTW | stable |
| 15 dB noise | channel | 78.14 | cached DTW | stable score, but not a confidence decrease |
| mild RIR | channel | 75.19 | cached DTW | stable |
| codec | channel | 68.23 | fallback equal | **undesired alignment loss** |
| bandlimit | channel | 72.49 | cached DTW | moderate collateral |
| consonant attenuation | speech | 76.40 | cached DTW | current non-SSL candidate is not sensitive enough |
| vowel spectral tilt | speech | 75.95 | cached DTW | current non-SSL candidate is not sensitive enough |
| local delete-like timing edit | speech | 77.72 | fallback equal | target-local evidence unavailable |
| long pause | speech | 59.34 | fallback equal | generic continuity responds |

The intended separation is therefore only partial: gain/RIR are stable, but
codec causes an alignment failure and local spectral edits do not yet affect a
pronunciation candidate.  This is exactly why WavLM remains candidate-only.

## 7. WavLM multi-reference pronunciation

`microsoft/wavlm-large` was run locally at 16 kHz with `AutoFeatureExtractor`,
attention-mask forwarding, and lazy loading.  The loader now uses
`local_files_only=True`, so ordinary tests/product calls cannot download the
checkpoint.  This run initially exposed missing optional `torch` and
`transformers`; they were installed only for the explicit benchmark.  A
cached checkpoint snapshot was then successfully loaded.

JVS uses 3 leave-one-out native references per sentence.  Robust scale
parameters were fit on odd-numbered sentences; even-numbered sentences were
held out.  Layer 24 (`alpha(layer12)=0`) had the largest held-out median
wrong-target separation: **35.50 normalized-distance units**, compared with
30.77 for layer 12.  Layer 12 remains preferable for native cross-speaker SD
in the earlier sweep, so no single layer is declared globally optimal.

The same-target JANON run completed for 7 words, each with one native `jpf1`
reference and two learners.  It shows real, nonconstant layer-12/24 distances,
but cannot establish multi-reference learner fairness: one learner's fused
distance is below the JVS native distribution while another is much larger.
No /100 SSL mapping is emitted or attached to `ProductScore v3`; that remains
unavailable until 3–5 same-sentence native references and held-out learner
validation are available.

## 8. Special mora, phrase intonation, and accent shadows

These remain default-off and user-facing false.  The carried-forward real
shadow checks found no negative ROI duration, out-of-audio ROI, NaN, or
constant payload.  Sokuon closure low-energy fraction was still uninformative
on the checked native samples; long-vowel periodicity and moraic-nasal context
remain insufficient for correctness claims.  Equal-boundary ROI outputs are
unavailable/low-confidence rather than precise local decisions.

Phrase intonation's existing +3-semitone check changed its candidate by only
+2.061 points on one native file, consistent with approximate global-pitch
invariance.  F0 flatten/final-rise removal/contour inversion still need a
multi-speaker rerun.  Accent nucleus stays `weak_target=true` for OpenJTalk
targets and makes no correct/incorrect claim without verified lexical
provenance.

## 9. Tests and promotion decision

Focused new tests cover dynamic weight renormalization, fallback-equal
unavailability, continuous local-vs-global timing behavior, SSL fusion, and
base-first rescue control flow.  Full regression completed: **163 passed**
(six pre-existing/dependency deprecation warnings).

**Promotion decision: NO A/B CANDIDATE YET.**  Preconditions not yet met are
channel-stable alignment, a non-saturated generic-fluency-only fallback
policy, multi-reference same-target learner validation, and complete
intonation/special-mora counterfactuals.  The branch is still useful as the
safe implementation and benchmark foundation; it never changes v2 output.

Raw reproducible artefacts are under `outputs/product_score_v3/`:
`content_cascade.csv`, `v3_candidate_real_audio.csv`,
`v3_candidate_ladder.csv`, `channel_and_speech_perturbations.csv`,
`wavlm_fusion.csv`, and `wavlm_janon_same_target.csv`.
