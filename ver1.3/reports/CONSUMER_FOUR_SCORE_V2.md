# Consumer Four-Score Contract v2

## Status

This document supersedes the **product-score formula description** in `CONSUMER_FOUR_SCORE_V1.md`.

The executable single source of truth is:

`src/jp_speech_eval/score_contract.py`

Changing a public construct, component weight, display transform, or history-comparability rule requires a new `SCORE_CONTRACT_VERSION`. Documentation must not redefine these values independently.

This is a C-end practice contract, not a psychometric educational scale.

## 1. Public dimensions

The user-facing surface remains exactly four dimensions:

1. `流暢さ` — utterance fluency
2. `明瞭さ` — broad practice clarity
3. `リズム` — Japanese timing/rhythm structure
4. `抑揚` — phrase/sentence intonation

Lexical pitch accent is conditional detail evidence under prosody/intelligibility practice. It is not a fifth headline dimension.

### 明瞭さ

`明瞭さ` is broader than strict phone-level pronunciation accuracy. A phone-correctness model, WavLM distance, ASR transcript match, or recording-quality score must not silently redefine this construct.

Current evidence can include target-relative machine-intelligibility/acoustic support in fixed reading and lower-confidence target-independent evidence in broad/free speech. Missing strong evidence may fall back to a neutral product prior, but that prior must remain explicitly low confidence.

Future promotion requires a consumer-facing human criterion such as comprehensibility/ease of understanding. The existing frozen pronunciation-accuracy listener study remains a separate construct.

### リズム

`リズム` represents relative timing structure. It must not be interpreted as a rule that all Japanese morae should have equal duration.

The current legacy mora/special-mora timing evidence remains usable as one proxy. A new WavLM-DTW warp-path shadow (`rhythm_dtw_v1`) measures local tempo irregularity without treating global speaking-rate differences as the same error.

### 流暢さ

Current product evidence covers speed and breakdown fluency through rate and pause features. Repair fluency — filled pauses, repetition, restart, false start, self-repair — is not yet fully modeled and must not be claimed as solved.

### 抑揚

`抑揚` is phrase/sentence F0 movement and intonation. It is deliberately separate from strict lexical pitch-accent correctness. Low F0 coverage should reduce confidence/detail eligibility rather than be interpreted as flat or incorrect intonation.

## 2. Frozen ProductScore formula for contract v2

Internal component keys and weights are:

| Public construct | Internal component | Weight |
|---|---|---:|
| 明瞭さ | `clarity` | 0.30 |
| リズム | `mora_timing` | 0.25 |
| 流暢さ | `delivery_fluency` | 0.25 |
| 抑揚 | `intonation` | 0.20 |

Raw weighted score:

```text
raw = 0.30*clarity + 0.25*rhythm + 0.25*fluency + 0.20*intonation
```

Display transform:

```text
display = clip(70 + 1.08*(raw - 70), 0, 100)
```

The transform is a UX centering/stretch only. It is not an educational-measurement calibration.

The v1 report's statement that the consumer total was an equal-weight mean is historical and no longer authoritative. Code and documentation are now tied to the same versioned contract.

## 3. Always-score and no-score behavior

After basic Japanese/recording eligibility passes:

```text
strong evidence  -> score + stronger confidence/detail
partial evidence -> score + reduced confidence/detail
weak evidence    -> score + low-confidence fallback
```

True no-score remains appropriate for clear non-speech, non-Japanese/invalid content, or unusable recording.

Target mismatch alone is not pronunciation failure. Plausible Japanese that differs from a fixed prompt may be routed to general-Japanese scoring; target-relative local feedback must then be suppressed.

## 4. Reliability is not performance

The product contract separates:

- what the learner likely did;
- how confidently the system could measure it.

Alignment failure, F0 extraction failure, weak reference provenance, or moderate recording degradation should normally lower confidence and local-detail eligibility. They must not automatically become pronunciation/rhythm/intonation penalties.

Legacy raw evaluator caps remain audit fields until a controlled acceptance A/B shows they can be retired safely. The consumer layer must not introduce new reliability-as-performance penalties.

## 5. Progress/history contract

Every new product record stores:

- `score_contract_version`
- `evidence_schema_version`
- evaluation mode and mode family
- target text where relevant
- fixed-reference identity where relevant
- four component scores
- confidence/evidence tier metadata
- legacy raw scores separately for audit

A displayed total-score delta is allowed only when the records are comparable under `history_comparability()`.

The following are **not** directly comparable:

- legacy records without score-contract metadata;
- different score-contract versions;
- different evaluation-mode families;
- different fixed targets;
- fixed-reference records whose reference identity changed or is missing.

Observable acoustic quantities such as speech rate may still be compared across scoring generations when their measurement definition is stable.

## 6. Personal calibration

A user voice profile may summarize stable personal acoustics such as F0 range, speech rate, mora-duration proxy, and pause ratio. A cross-target calibration panel does **not** automatically establish a personal `/100` correctness baseline.

Accordingly, `consumer_baseline_scores` may be stored descriptively, while `direct_score_delta_allowed` remains false unless a future protocol explicitly validates that comparison.

## 7. Reference-dependency practice

Step 1 shadowing and Step 3 free production often use different evidence families. Their headline totals must not be blindly subtracted and called a `reference_dependency_gap`.

When total-score contexts are incompatible, the product reports no numeric score gap and instead inspects cross-step observables such as rate and pause changes. A future same-contract experimental design may re-enable a headline gap when comparability is explicit.

## 8. Research shadows

The following remain shadow/research evidence until criterion validation:

- WavLM pronunciation distance and layer/fusion candidates;
- WavLM-DTW warp-path rhythm metrics;
- CTC/GOP posterior/logit/sequence evidence;
- special-mora v2;
- strict phrase/accent diagnostics;
- future interval-distortion metrics requiring a validated vowel/consonant classifier.

No shadow becomes user-facing merely because it separates native from learner or wrong-target speech.

## 9. Next promotion gates

1. Finish reference-bank provenance migration.
2. Run held acceptance without threshold tuning.
3. Execute the frozen pronunciation-accuracy study for segmental evidence.
4. Execute `CONSUMER_CRITERION_PROTOCOL_V2.md` for the four product constructs.
5. Benchmark candidate evidence on speaker/target-disjoint data, including UME-JRF where licensing permits research use.
6. Only then fit or revise a user-facing `/100` mapping.
