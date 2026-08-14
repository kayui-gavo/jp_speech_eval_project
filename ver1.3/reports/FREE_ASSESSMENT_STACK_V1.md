# Free Assessment Stack v1

## Decision

For the current product/research phase, the repository should use **free/local components only**. Paid pronunciation APIs are not part of the runtime plan and must not be required for tests, scoring, or fallback behavior.

This change is intentionally additive and shadow-first. It does **not** alter ProductScore v2, enable ProductScore v3, map SSL evidence to `/100`, or retire an existing local metric.

## What is already free and usable

### pyopenjtalk / OpenJTalk — keep and formalize

`pyopenjtalk` is already a base dependency and already powers the repository's Japanese text frontend. Upstream documents `g2p`, `run_frontend`, and `extract_fullcontext`; the wrapper is MIT-licensed and OpenJTalk is Modified BSD. The new `japanese_target_evidence.py` makes reading, phones, full-context labels, accent-source provenance, and optional marine output explicit.

A manual reading override is treated conservatively: it supplies the intended kana/phone target but **does not invent an automatic lexical-accent target**. This matters for names, place names, loanwords, and other items for which the default dictionary may choose the wrong reading.

Upstream: https://github.com/r9y9/pyopenjtalk

### faster-whisper — keep as the free ASR backbone

`faster-whisper` is MIT-licensed and is already a base dependency. It remains the product's local Japanese ASR/content-verification backbone. ASR recognition is evidence about spoken content; it is not reinterpreted as pronunciation correctness.

Upstream license: https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE

### WavLM — keep as pronunciation research evidence

The existing WavLM cosine-DTW work remains useful as a free/local pronunciation-similarity research backbone. The current repository correctly keeps the evidence unmapped and candidate-only. No normal test should download model weights.

Upstream project: https://github.com/microsoft/unilm/tree/master/wavlm

## Free candidates worth benchmarking, but not silently installing

### WhisperX Japanese alignment

WhisperX is BSD-2-Clause. Its current alignment code includes Japanese in the default alignment-model table and uses `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`; that Hugging Face model is marked Apache-2.0. This makes WhisperX a credible **word-timestamp/forced-alignment candidate**, particularly for free speaking and fixed-reading word localization.

It is not a pronunciation-correctness score, and the Japanese alignment model is large. Therefore v1 only discovers whether WhisperX is installed; it does not add it to base requirements or download weights automatically.

Sources:
- https://github.com/m-bain/whisperX
- https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese

### Montreal Forced Aligner (MFA)

MFA is a strong batch/research candidate for phone boundaries. Current MFA documentation provides a Japanese model path, and the 2026 MFA paper evaluates Japanese and reports mean boundary errors below 15 ms across the reported benchmark setting. The package is distributed as MIT; model-specific licenses/attribution still need to be preserved and checked for the exact model version used.

MFA is operationally heavy (Kaldi/conda/model provisioning), so it should first be used as an **offline reference/benchmark backend**, not a hidden dependency of a C-end request path.

Sources:
- https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
- https://montreal-forced-aligner.readthedocs.io/en/stable/user_guide/models/model_versions.html

### marine

`pyopenjtalk` supports optional DNN accent estimation through marine; upstream lists marine as Apache-2.0. It is useful as a second target-side accent opinion. It must remain shadow-only until compared with verified accent targets and must not be called automatically.

## Free tools deliberately *not* added to the product default

### openSMILE open-source edition

Do **not** add the open-source openSMILE package to the commercial/product default merely because it can be installed for free. Its upstream Research License explicitly restricts commercial product use without an additional commercial license. Research use may be possible under its terms, but that is different from a free product dependency.

Source: https://github.com/audeering/opensmile/blob/master/LICENSE

### praat-parselmouth

Parselmouth exposes Praat algorithms and is GPL-3.0-or-later. It can be scientifically useful for research cross-checks, but it is not added to the default product dependency set in this branch. Deployment/distribution obligations should be reviewed before deciding how to use it in a commercial product.

Source: https://github.com/YannickJadoul/Parselmouth

## Architecture policy

The free-only stack is:

| Need | Default free path | Optional research/shadow |
|---|---|---|
| Japanese target reading/phones | pyopenjtalk | marine accent shadow |
| Content verification | faster-whisper | — |
| Fixed-reading local timing | existing cached DTW | WhisperX word alignment; MFA phone alignment |
| Broad pronunciation evidence | existing WavLM shadow | multi-reference calibration research |
| Fluency | existing pause/rate evidence | WhisperX word timestamps |
| Phrase intonation | existing local F0 shadow | future validated target-relative model |
| Lexical pitch accent | OpenJTalk/verified target + local F0 shadow | marine target opinion |
| Special mora | existing local special-mora shadow | MFA phone-boundary benchmark |

The system must continue to distinguish **target-side linguistic evidence** from **learner-side acoustic evidence**. OpenJTalk saying that a target contains `/N/`, a long vowel, or an accent nucleus never proves that the learner realized it correctly.

## Rollback / merge policy

This work lives on `free-assessment-integration-v1`. Nothing should be merged to `main` merely because the code is additive. Before merge:

1. run the full repository test suite;
2. verify no ProductScore v2 output changes on the existing parity panel;
3. verify no optional model is downloaded by ordinary tests;
4. benchmark WhisperX/MFA separately before making either a runtime dependency.

If any product regression appears, the branch can be discarded or reset to its base commit; `main` is untouched.
