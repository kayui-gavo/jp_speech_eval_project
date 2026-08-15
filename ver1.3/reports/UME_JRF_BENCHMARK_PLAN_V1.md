# UME-JRF Benchmark Plan v1

## Decision

Use the **UME Japanese Read speech corpus (UME-JRF)** as an external **research benchmark**, not as bundled product content and not as a production dependency.

The official NII corpus page states that the corpus is provided for research purposes. The repository must therefore store only code, manifests, derived aggregate results allowed by the license, and provenance metadata. Do not commit corpus audio or redistributed rating sheets unless the applicable license explicitly permits it.

As of the official NII description, UME-JRF contains Japanese read speech from 141 learners with 26 native-language backgrounds at intermediate-to-advanced levels, including:

- 503 ATR phonetically balanced sentences;
- 108 difficult-sound sentences;
- 42 prosody-focused sentences;
- 115 minimal-pair words;
- 16 kHz / 16-bit / mono WAV audio;
- evaluation/grading data from Japanese native teachers/educators.

The current NII page directs distribution through NII-IDR. Dataset access/terms acceptance is an external human/local step and must not be automated by ordinary repository tests.

## 1. Research questions

The benchmark should answer **which existing evidence is useful for which construct**, not simply which model best separates native and learner identity.

Primary questions:

1. Which evidence best tracks strict pronunciation accuracy?
2. Does WavLM-DTW tempo irregularity track human rhythm judgments better than the current mora-duration proxy?
3. Which evidence is robust across speakers, targets, and L1 groups?
4. Which metrics fail because they primarily encode speaker/channel/target identity?
5. Which signals should remain shadow-only?

No product threshold or `/100` mapping is selected from the final held-out evaluation folds.

## 2. Relevant JRF strata

McIntosh et al. (2026) use JRF subsets for Japanese L2 phone/rhythm/intonation evaluation. Their reported Japanese tasks make the corpus particularly relevant to this repository:

### Holistic sentence pronunciation

Use the sentence-level pronunciation subset for broad phonetic evidence comparisons.

Candidate evidence:

- WavLM layer 12 distance — frozen current baseline;
- WavLM final-layer distance — candidate;
- WavLM layer 24 distance — candidate/secondary;
- multi-reference aggregation strategies selected only on development folds;
- Beatrice phone-CTC posterior evidence;
- raw-logit competition evidence;
- restricted/alignment-free CTC sequence/GOP evidence;
- ASR recoverability;
- current reference-relative MFCC/DTW proxy for comparison only.

### Difficult sounds

Use the difficult-sound subset to examine phenomena such as gemination and long-vowel contrasts where the current special-mora stack needs learner criterion evidence.

Do not infer phone correctness from forced-alignment success alone.

### Prosody / intonation

Use the prosody-focused subset for phrase/sentence intonation candidates when the released human criterion is compatible with the target construct.

Strict lexical pitch-accent correctness remains a separate diagnostic unless the dataset provides a verified lexical-accent criterion for that item.

## 3. Rhythm candidates

The new research baseline should compare:

1. current mora-duration/rhythm proxy;
2. global duration ratio / global tempo difference;
3. WavLM-DTW `tempo_irregularity_rad` from `rhythm_dtw_v1`;
4. future interval-distortion evidence only after a validated frame-level vowel/consonant/silence classifier exists.

The key hypothesis is not that native-like global speed is always better. `tempo_irregularity` is designed to isolate local variation in the DTW warp path relative to the utterance's own average tempo.

## 4. Manifest boundary

The repository benchmark runner must consume a normalized local CSV/JSON manifest rather than hard-code proprietary corpus directory structure.

Required normalized fields should include at least:

- `sample_id`
- `audio_path`
- `speaker_id`
- `target_id`
- `target_text`
- `task`
- `criterion`
- `human_rating`

Recommended metadata where available:

- `l1`
- `proficiency`
- `rater_id`
- `subset`
- `condition`

One audio file can appear in multiple rows when it has multiple raters or criteria. The `(sample_id, criterion, rater_id)` key must be unique when `rater_id` is available.

The local corpus-to-manifest mapping is intentionally separate from model code so future corpus versions can be remapped without changing the evaluator.

## 5. Data splitting

At minimum report speaker-disjoint evaluation.

For target-relative evidence, also report target-disjoint evaluation where the subset size permits it.

Recommended process:

```text
train/development speakers/targets
    -> choose layer, reference aggregation, normalization, fusion, thresholds
held-out speakers/targets
    -> evaluate once without re-selection
```

If k-fold cross-validation is used, every model-selection operation must occur inside the training/development portion of each fold.

Native references used by a learner test item must not leak test-speaker information. Keep reference selection and aggregation reproducible.

## 6. Required outputs

For every candidate evidence family, export a tidy table containing:

- sample/target/speaker IDs;
- criterion and human aggregate;
- raw machine evidence;
- evidence availability/failure reason;
- model/layer/reference metadata;
- no `/100` score unless the value is already part of the frozen current product baseline.

Report:

- Spearman correlation;
- QWK/ordinal agreement where appropriate;
- MAE only for a predeclared mapped prediction task;
- within-target statistics;
- held-out-speaker statistics;
- held-out-target statistics where applicable;
- availability/failure rate;
- L1 and task subgroup diagnostics where sample size permits;
- channel/speaker sensitivity diagnostics.

Do not report only a pooled native-vs-learner AUC as evidence of pronunciation validity.

## 7. Product firewall

UME-JRF benchmark results do not automatically modify ProductScore.

A candidate may be promoted only after:

1. it has a construct-matched human relationship on development + held-out data;
2. its direction and scale are stable enough for C-end use;
3. product acceptance tests show ordinary Japanese retains useful score availability;
4. uncertainty/failure behavior does not become a hidden learner penalty;
5. a separate mapping/calibration decision is documented.

Research-only corpus terms must also remain separate from any commercial runtime asset decision.

## 8. Work split

### Can be implemented in the repository now

- normalized manifest schema/validator;
- benchmark runner interfaces;
- WavLM/CTC/rhythm feature export adapters;
- split logic and leakage guards;
- statistics/report generation;
- CI tests using synthetic manifests only.

### Requires local/human execution

- accepting NII-IDR terms and obtaining UME-JRF;
- mapping the downloaded corpus/rating files to the normalized manifest;
- running heavy WavLM/CTC models across the corpus if the model checkpoints are not available in CI;
- preserving any license-required provenance outside public artifacts.

These local steps should execute a frozen repository benchmark rather than redesigning the scoring system.

## References

- NII Speech Resources Consortium, UME Japanese Read speech corpus (UME-JRF), official corpus description/distribution page.
- McIntosh, Smit, Saito, Minematsu, and Kamper (2026), *Self-supervised Speech Comparison for L2 Phone, Rhythm, and Intonation Scoring*, arXiv:2607.13721.
