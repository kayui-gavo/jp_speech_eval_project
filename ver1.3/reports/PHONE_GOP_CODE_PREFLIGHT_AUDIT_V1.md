# Japanese phone-GOP code preflight audit v1

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Human recording gate: **BLOCKED**

## Decision

Do not ask a human speaker to record the 38-clip battery yet. The current Stage-A code is promising but not sufficiently hardened to justify spending human recording time.

The existing synthetic tests prove only that the basic CTC Viterbi and local competition features behave on toy logits. They do **not** yet prove that the chosen Japanese backend, target-phone frontend, allophone handling, deletion/insertion behavior, and end-to-end model plumbing are sound enough for human data collection.

## P0 blockers found in code/source audit

### P0-1. Target frontend mismatch is unresolved

The default phone model is `prj-beatrice/japanese-hubert-base-phoneme-ctc-v4`. Its model card states that training labels were generated with **pyopenjtalk-plus**. The project target evidence currently comes from the ordinary `pyopenjtalk` package.

Those two frontends are related but not identical; pyopenjtalk-plus ships a different/customized dictionary and explicitly documents phone/reading behavior differences. Before human recording, the GOP path must either:

1. use the same frontend family as the backend training labels; or
2. freeze a compatibility layer and prove that all pilot targets produce the intended phone sequence under both frontends.

The recording manifest must not be the first place where this mismatch is discovered.

### P0-2. `pau` and `sil` are currently eligible GOP competitors

The current competitor exclusion removes PAD/UNK/SOS/EOS/blank but does not exclude `pau` and `sil`, even though the Beatrice vocabulary explicitly contains both. A local phone GOP score must not report pause/silence as if they were ordinary segmental competitor phones.

Fix required: non-speech tokens must be excluded from segmental competitor sets and from top-level clarity interpretation.

### P0-3. Japanese high-vowel devoicing can be falsely treated as an error

The Beatrice vocabulary distinguishes `i/I` and `u/U`. The current extractor treats the uppercase and lowercase variants as competing labels.

That is unsafe for Japanese pronunciation assessment because devoiced high vowels are legitimate context-dependent realizations. A native-like realization must not lose clarity merely because the backend prefers `I` while the canonical frontend emits `i`, or vice versa.

Fix required: implement an allophone-equivalence policy for at least `i~I` and `u~U` in alignment and target-vs-competitor scoring. Keep a separate diagnostic if devoicing itself is studied later.

### P0-4. The current forced CTC path is not deletion/insertion-ready

`ctc_viterbi_align()` forces every canonical phone to receive support. This is useful for local feature extraction but means a truly deleted phone can still be assigned an artificial frame.

The manual battery currently contains many deliberate deletions (`cl`, long-vowel mora, `N`) and insertions (split yoon). It would be premature to spend human time recording those until the code has at least one deletion/insertion-sensitive sequence-level signal.

Fix required before recording:

- canonical CTC sequence log-probability;
- leave-one-phone-out deletion comparison or equivalent CTC-GOP deletion evidence;
- unconstrained/greedy phone sequence edit evidence for insertion/deletion localization;
- synthetic regression tests for deletion and insertion.

Full segmentation-free GOP can remain a later stage, but Stage A needs more than a forced canonical path.

### P0-5. Model revision is not pinned

The current backend loads the mutable Hugging Face model ID without a revision. The v4 repository currently points to commit `f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99` at the time of audit, but future upstream updates could silently change outputs.

Fix required: pin the research backend revision in code/report metadata and expose it in every JSON result.

### P0-6. No real end-to-end model smoke test exists yet

Current tests use synthetic logits and only check lazy/local-only loading. There is no automated test proving:

- the actual processor/model can load;
- tokenizer vocab size and model output dimension agree;
- blank/pad id is correct;
- expected sample rate is 16 kHz;
- canonical Japanese phones are covered;
- bundled Japanese reference audio produces finite GOP evidence;
- the same audio under an obviously wrong target produces worse sequence evidence;
- simple gain/channel perturbation does not change GOP more than a wrong target.

This preflight should use existing bundled/reference assets. It should not require new human recordings.

## P1 issues to harden before interpreting results

### P1-1. Max-competitor provenance bug

The code chooses the best competitor by mean logit and separately finds the best competitor by maximum logit, but stores only one competitor phone/id. Therefore `best_competitor_max_logit` can belong to a different phone than `best_competitor_phone`.

Fix required: store separate mean-competitor and max-competitor phone/id fields.

### P1-2. CTC support frames are not physical phone duration

`frame_count * frame_stride` is currently exposed as `duration_sec`. The module comments warn that CTC is peaky, but the field name can still be misread as a real phone duration.

Fix required: rename/label it as CTC support duration and keep real special-mora duration evidence in the timing/alignment subsystem.

### P1-3. Frame stride should come from the model frontend where possible

For the current HuBERT config, convolution strides multiply to 320 samples, i.e. 20 ms at the documented 16 kHz sample rate. The code currently estimates frame stride by `audio_duration / frame_count`.

Fix required: prefer model-config convolution stride; use duration/frame count only as an explicit fallback.

### P1-4. Backend input validation is too permissive

Before model inference, reject/flag empty, non-finite, near-silent, wrong-sample-rate and implausibly short waveforms. A shadow scorer must fail locally rather than poison a later batch report.

## Human-time protection gate

The 38-clip quick recording battery becomes eligible only after all of the following are true:

1. ordinary repository tests pass;
2. Japanese-specific GOP unit tests cover `pau/sil`, `i/I`, `u/U`, repeated phones, deletion and insertion behavior;
3. backend/model revision is pinned and reported;
4. target frontend compatibility is frozen for every quick-pilot phrase;
5. bundled-reference end-to-end preflight passes with no new recording;
6. wrong-target evidence is substantially worse than correct-target evidence on the same bundled audio;
7. harmless gain/resampling/channel controls do not cause larger degradation than wrong-target controls;
8. no product `/100` mapping is enabled;
9. manual protocol explicitly distinguishes CTC support frames from true phone duration;
10. an automatic batch report exists so the human is not asked to manually inspect dozens of raw JSON files.

Until these pass, the correct action is **code and automatic-data work, not human recording**.

## Next engineering order

1. harden target inventory/allophones/non-speech handling;
2. add deletion/insertion-sensitive sequence evidence;
3. pin and self-check Beatrice v4 backend;
4. add bundled-reference end-to-end preflight;
5. add batch comparison/report script;
6. only then reopen the recording gate.
