# Official JVS native phone-CTC anchor preflight v1

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Product score impact: **NONE**
New user recordings: **NONE**
Local phone-error labels: **NONE**

## Question

The bundled Stage-0 reference is Aivis TTS. Before treating any odd local phone evidence as a model/metric problem, test whether the same pinned Japanese phone-CTC backbones behave sensibly on real human native speech.

The official JVS corpus project page links three small samples for `jvs001`, `jvs002`, and `jvs003`. All three are the same `VOICEACTRESS100_001` text:

`また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。`

This preflight downloads those WAV files only for the CI run, validates the reviewed official source/file IDs plus WAV/audio semantics, evaluates them, and then deletes the audio before artifact collection.

## Reproducibility correction: transport bytes are not acoustic identity

The first successful run observed these raw HTTP/WAV representations:

| speaker | first observed bytes | first observed raw SHA-256 |
|---|---:|---|
| jvs001 | 778284 | `dc9fd6e4caefc6e1781ad225f0b41ca13153da4afe2fb92f39f175fa3d9d85a7` |
| jvs002 | 642764 | `d91e5199508d89b45d68f18473c013f90bbfd68c1940ad039f5ea0b183f61ae2` |
| jvs003 | 661004 | `7b164601457b27c8a89c6aaef971967e5a6c7cf9bf2d46c303e21d1e8e41ed15` |

A later official Google Drive response for `jvs001` changed to a much smaller raw WAV representation while preserving the expected ~8.621 s utterance. That exposed a flaw in the first reproducibility policy: a Google Drive transport/container byte hash is not a stable acoustic identity.

The old hard byte-hash gate has therefore been **removed rather than widened with arbitrary extra hashes**.

Current downloader policy verifies:

- the reviewed official JVS project-page Google Drive file ID;
- readable uncompressed mono PCM WAV;
- expected utterance duration within a tight tolerance;
- plausible PCM width and positive sample rate/frame count;
- downstream target-conditioned acoustic behavior after project-standard resampling.

Raw bytes, raw SHA-256, sample rate and sample width are still recorded as provenance. A raw-hash change alone is not called source failure. Duration/audio-semantic drift still fails closed and requires source inspection.

This is a more defensible reproducibility policy than pretending a mutable HTTP representation is an immutable corpus checksum.

## Control design

A different sentence can trivially change CTC sequence posterior because of phone-count/length differences. The native-anchor control therefore preserves exactly the same phone multiset and phone count but rotates the canonical phone order deterministically.

For each speaker/model we compare:

- canonical phone sequence posterior;
- same-length rotated phone-sequence posterior;
- canonical minus rotated margin per CTC frame.

For `jvs001`, the canonical audio is also evaluated after `0.8x` and `1.2x` amplitude gain. The gain delta is compared with the target-order margin.

This is an **acoustic/target consistency check**, not a pronunciation-correctness benchmark.

## Results

All 3 native speakers were available, and all 3 backbones preferred the known native target over the same-length rotated phone sequence for **all 9 speaker × model cases**.

| backbone | min canonical−rotated / frame | mean canonical−rotated / frame | cross-speaker SD of canonical LPP/frame | max |gain delta| / frame on jvs001 | min target margin / max gain delta |
|---|---:|---:|---:|---:|---:|
| Beatrice | 0.865371 | 0.887901 | 0.003264 | 0.00005066 | 17083× |
| DistilHuBERT dual CTC | 0.552010 | 0.629534 | 0.020749 | 0.00019146 | 2883× |
| WavLM dual CTC | 0.870016 | 0.906526 | 0.007222 | 0.00005639 | 15427× |

The ratios are not psychometric effect sizes and must not be compared as model-quality scores. They only show that, on these native anchors, a severe phone-order mismatch dominates mild amplitude change by several orders of magnitude within every backbone.

### Per-speaker margins

Canonical minus same-length rotated sequence, per CTC frame:

- Beatrice: `0.865371`, `0.898176`, `0.900156` for jvs001–003.
- DistilHuBERT: `0.552010`, `0.645254`, `0.691336`.
- WavLM: `0.870016`, `0.903181`, `0.946380`.

All are positive.

## What this resolves

The earlier bundled-Aivis result was not merely an artifact of the system being unable to model normal human native Japanese. On three independent official JVS native speakers, all three phone-CTC backbones strongly support the known target ordering over a same-length phone-order control and remain essentially invariant to mild gain.

This reduces the probability that the whole phone-CTC direction is unusable because of a TTS-only domain artifact.

## What this does **not** resolve

This result does not provide local phone-error labels. Therefore it cannot establish:

- that a particular learner phone is correct/incorrect;
- an LPR or `Occ(i)` threshold;
- phone-level false-alarm/recall;
- a Japanese learner MDD model;
- a C-end clarity `/100` mapping.

Native correct-target discrimination is necessary engineering evidence, not learner-error criterion validity.

## Decision

- **PASS** native-human target-consistency sanity gate.
- **PASS** mild-gain robustness sanity gate for these three samples.
- **CORRECTED** source reproducibility gate now uses reviewed source ID + acoustic semantics; transport-byte SHA is provenance only.
- Continue machine-only work with criterion-ready `{LPP, LPR, normalized graph GOP, Occ(i)}` bundles and already-existing learner/native data.
- Keep all phone evidence shadow-only.
- Keep the new-user-recording gate **BLOCKED** while existing expert-labeled corpora such as UME-JRF are pursued first.
