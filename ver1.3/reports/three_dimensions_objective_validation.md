# Pronunciation, Rhythm and Fluency Objective Validation

- JVS native rows: 300 (real audio transcripts plus phone-lab timing)
- JVS phone-to-mora mapping success: 300/300
- JANON sentence rows: 1976 (real audio; external descriptive audit only)
- No human ratings are introduced. JANON learners are not treated as bad-pronunciation labels.
- Runtime scoring is unchanged by this audit.

## Bottom Line

- **Pronunciation clarity is not stable yet.** Real native phone-lab timing receives a mean of only 13.18, while equal-boundary fallback produces 100. The score is dominated by the alignment representation and cannot detect segment substitutions.
- **Rhythm is not stable yet.** It responds to timing jitter with reliable boundaries, but equal fallback forces the score to 100 and removes the evidence needed for special-mora judgement.
- **Fluency is the strongest of the three, but still coarse.** It separates gross hesitation/fast speech; slow speech separation is weak, and readable JANON speech frequently hits 100.
- Therefore these dimensions can return numbers, but only fluency currently has useful coarse negative-control separation. None of the three has pitch-v2-level validation.

## Native / External Distributions

| group and dimension | n | mean | p10 | p50 | p90 | ceiling rate |
|---|---:|---:|---:|---:|---:|---:|
| JVS pronunciation timing proxy | 300 | 13.18 | 0.0 | 0.0 | 42.1 | 0.0 |
| JVS rhythm timing proxy | 300 | 53.4033 | 15.0 | 55.5 | 82.0 | 0.0 |
| JVS fluency | 300 | 90.8233 | 79.9 | 93.0 | 100.0 | 0.18 |
| JANON native fluency | 284 | 97.2042 | 89.0 | 100.0 | 100.0 | 0.5141 |
| JANON learner fluency (descriptive) | 1692 | 98.9397 | 97.0 | 100.0 | 100.0 | 0.8345 |

## Controlled Separation

| dimension | negative control | normal mean | control mean | delta | AUC | result |
|---|---|---:|---:|---:|---:|---|
| Pronunciation proxy | mora timing jitter | 13.18 | 0.4867 | 12.6933 | 0.7119 | weak separation; native already near floor |
| Pronunciation proxy | compressed special mora | 13.18 | 0.12 | 13.06 | 0.7306 | weak separation; floor-limited |
| Pronunciation proxy | segment substitution | 13.18 | 13.18 | 0.0 | 0.5 | FAIL: invisible to current formula |
| Rhythm | mora timing jitter | 53.4033 | 31.86 | 21.5433 | 0.732 | timing-sensitive with real boundaries |
| Rhythm | equal-boundary fallback | 53.4033 | 100.0 | -46.5967 | 0.0 | FAIL: fallback inflates score |
| Fluency | slow | 90.8233 | 84.8133 | 6.01 | 0.6092 | speed-sensitive |
| Fluency | fast | 90.8233 | 75.2667 | 15.5567 | 0.9309 | speed-sensitive |
| Fluency | hesitation | 90.8233 | 62.06 | 28.7633 | 0.9789 | pause-sensitive |

## Findings by Dimension

### Pronunciation clarity

The current score is `100 - 90 * mora-duration CV - special-mora penalties`. Natural phone-lab mora durations are not equal: devoicing, phrase structure and legitimate special-mora timing create substantial variation. All 300 JVS mappings succeeded, yet the native median is 0, so the low result is not explained by mapping failure alone. A phoneme substitution with unchanged timing also produces exactly the same score. This dimension is neither stable on native speech nor capable of consonant/vowel correctness judgement in its current form.

### Rhythm / special mora

With phone-lab boundaries, timing jitter is detectable, but the native mean is only 53.40. Under equal-mora fallback, every mora is assigned the same duration and rhythm becomes 100 by construction. The current result therefore changes more with the alignment backend than it should. A fallback result can remain numeric for UX, but it must be described as a coarse estimate with low confidence and cannot support specific special-mora correction.

### Fluency

Fluency responds strongly to fast speech and hesitation, but slow-control AUC is only 0.61. JANON learners remain a readable external population rather than a negative class, so high scores are not inherently wrong. However, 83% of learner sentence recordings and 51% of JANON native recordings score exactly 100, which shows a ceiling/resolution problem. The dimension also needs robustness checks for VAD and endpointing because those directly alter duration and pause ratio.

## Scientific Status

| dimension | numeric availability | native stability | controlled separation | current status |
|---|---|---|---|---|
| Pronunciation clarity | yes | **FAIL**: JVS mean 13.18, fallback 100 | weak timing separation; phoneme errors invisible | redesign required |
| Rhythm / special mora | yes | **FAIL**: JVS mean 53.40, fallback 100 | moderate with reliable boundaries | alignment-dependent; redesign required |
| Fluency | yes | partial: JVS high, JANON ceiling-heavy | strong for hesitation/fast, weak for slow | usable coarse proxy, not fine-grained |

## Next Step Without Human Ratings

1. Add waveform-level phoneme corruption controls and an acoustic/ASR posterior feature before claiming pronunciation separation.
2. Replace equal-boundary rhythm evidence with phone/alignment-derived timing when available; otherwise keep a coarse numeric score with low confidence.
3. Run gain/noise/codec/VAD perturbations and require limited score drift on clean native speech.
4. Keep JANON as an external readable-learner audit. Do not train a learner-vs-native classifier and call it pronunciation quality.
