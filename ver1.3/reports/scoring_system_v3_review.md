# Scoring system v3 review

## Scope

This review targets the practice-score path and the score-display contract. It does not add an ASR/TTS provider, train a new speech model, or claim teacher-grade calibration.

## Problems found and corrected

| area | previous behavior | correction |
|---|---|---|
| Pronunciation clarity | Global mora-duration CV was treated as pronunciation. JVS native phone timing averaged 13.18, while equal fallback produced 100. | The visible score now uses recording quality, acoustic evidence coverage, alignment evidence and available content intelligibility evidence. Duration CV remains debug-only. |
| Rhythm | Natural duration variation was heavily penalized; equal boundaries were perfect by construction. | Reliable timing uses robust log-duration MAD and adjacent log-duration change. Equal fallback returns a narrow low-confidence estimate around the low 70s. |
| Fluency | A broad 4--7 mora/s interval gave exactly 100 to most read speech. | A continuous log-rate curve removes the plateau. Pause penalties apply only beyond utterance-length-aware allowance. |
| Overall score | The weak overall used pronunciation, pitch and rhythm but omitted the separately displayed fluency score. | The overall now aggregates the same four displayed dimensions: clarity 30%, rhythm 20%, fluency 25%, pitch 25%. |
| Score inflation | Fixed and weak display paths contained high-score floors that could keep an overall score above 85 despite a very low dimension. | Hidden floors were removed. A low visible dimension now actually lowers the overall. |
| Missing pitch | Missing mora-level F0 produced an empty fourth card even when enough frame-level F0 existed. | A capped 45--78 frame-F0 fallback supplies a low-confidence broad pitch-naturalness number. It is never used as accent correctness. |
| Zero handling | Python `or` and UI fallbacks could replace a legitimate score of 0 with another raw/debug score. | User-facing values use explicit `None` checks. Zero remains zero. |
| Special mora | The default renderer selected a limited candidate profile even though rollout reports still marked the feature as not production-ready. | Default behavior is safe/shadow. Long-vowel and moraic-nasal feedback requires explicit opt-in and reliable non-fallback evidence. Sokuon and yoon remain blocked. |
| Special-mora summary | A fallback/equal boundary could still contribute to a formal special-mora summary score. | Fallback, failed mapping and severe mapping warnings are excluded. |
| UI confidence | Weak-reference scores could inherit `high` from overall acoustic reliability. | Weak pronunciation/pitch are capped at medium confidence; fallback pronunciation/rhythm and coarse frame pitch are low confidence. |
| Fixed-reference fallback | A failed detailed alignment hid every score even when content, recording quality and broad acoustic evidence were usable. | Strict comparisons and special-mora details remain blocked, but the UI now shows four explicitly low-confidence practice proxies. Raw strict pitch never backfills the weak pitch value. |

## Objective audit

The engineering sanity audit uses 300 held-out JVS native recordings, paired timing/fluency perturbations, 60 synthetic channel degradations and 1,976 JANON sentence recordings as an external read-speech distribution. The synthetic AUC values only verify that the implementation reacts to those controlled degradations; they are not evidence of phoneme-error accuracy.

| metric | normal | control |
|---|---:|---:|
| Pronunciation clarity | 95.0 | 82.0 at 5 dB noise; 87.6 at low gain |
| Rhythm | 93.69 | 67.04 timing jitter; 72.77 equal fallback |
| Fluency | 92.11 | 80.77 slow; 80.71 fast; 52.93 hesitation |
| Pitch | 89.76 | existing pitch-v2 controls remain unchanged |
| Four-dimension overall | 92.68 | no hidden score floor |

JANON native and learner read-speech fluency means are 93.73 and 93.57. This is not treated as a failure: both groups read prepared scripts fluently. The important change is that neither distribution piles up at 100.

Detailed results are in `reports/practice_dimensions_v2_validation.md` and `data/calibration_candidates/practice_dimensions_v2_summary.csv`.

## Current display contract

1. A complete Japanese utterance that passes content and minimum-evidence checks receives four practice numbers.
2. Numeric score and confidence are separate. A number with low confidence is a coarse estimate, not a precise judgement.
3. Content mismatch, non-Japanese input, silence and extremely short input remain no-score states.
4. Raw strict-reference/debug metrics never backfill a missing user-facing value.
5. The overall is computed from the same four cards that the user sees.
6. Fixed-reference alignment fallback downgrades to four low-confidence practice proxies; it does not expose strict reference scores or special-mora corrections.

## Remaining scientific bottlenecks

### Pronunciation clarity

The current system can detect recording/evidence degradation, but not a cleanly recorded consonant or vowel substitution. Phone-level GOP, a calibrated ASR posterior feature, or human-labelled pronunciation errors are still needed before calling this phoneme accuracy.

### Rhythm and special mora

Broad rhythm now behaves well with reliable phone timing. Runtime arbitrary-sentence alignment is still commonly MFCC-DTW or equal fallback, so specific long-vowel, sokuon and moraic-nasal correction remains the main bottleneck. Sokuon has insufficient native coverage; yoon is not a duration-error task.

### Fluency

The score separates gross speed and hesitation controls. It still cannot distinguish a linguistically appropriate phrase pause from an inappropriate pause at the same acoustic duration without phrase-boundary information. Current allowance prevents obvious over-penalisation but is not a full phrasing model.

### Pitch

Pitch-v2 measures broad naturalness. The soft OpenJTalk/OJAD-style hint is a bounded mismatch penalty, not a ground-truth accent target. The frame-F0 fallback is deliberately low-confidence and capped. Wrong lexical-accent detection remains unresolved.

### Calibration

The scale is objectively sanity-checked but not human calibrated. JVS verifies native stability and paired perturbation sensitivity; JANON is an external learner trend, not an error label. The system remains a practice demo rather than an examination score.
